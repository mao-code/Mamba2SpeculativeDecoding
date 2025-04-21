import torch

import torch
import torch.nn.functional as F
from Mamba2.modeling_mamba2 import Mamba2ForCausalLM, Mamba2Cache
from typing import Tuple, List

def _commit_prefix(
    cache: Mamba2Cache,
    step_hist: List[List[torch.Tensor]],
    final_hist: Tuple[torch.Tensor, ...],
    m: int, K: int
) -> None:
    """
    Copy accepted SSM states from verifier to cache **in-place**.
    For m==K we can just take final_hist (already advanced K steps).
    """
    if m == 0:
        return
    src = final_hist if m == K else [h[:, m - 1] for h in step_hist]
    for l, st in enumerate(src):
        cache.ssm_states[l].copy_(st)   # O(1) pointer copy on GPU


# decoding_fast.py  (only the core loop shown)
@torch.inference_mode()
def mamba_spec_decode_seq(
    target: Mamba2ForCausalLM,
    draft : Mamba2ForCausalLM,
    prompt_ids: torch.Tensor,
    K: int = 3,
    max_new: int = 256,
):
    """
    q-distribution is the softmax of the logits from the draft model.
    p-distribution is the softmax of the logits from the target model.
    """
    
    # ensure models are in eval mode
    target.eval()
    draft.eval()

    device = prompt_ids.device
    gen_ids = prompt_ids.clone()               # (1, L0)
    
    # 1. Warm-up: run both models on the prompt to fill their caches
    tgt_cache = target(input_ids=prompt_ids, use_cache=True, return_dict=True).cache_params
    draft_cache = draft(input_ids=prompt_ids, use_cache=True, return_dict=True).cache_params

    total_accept_rate = 0
    runs = 0

    while gen_ids.size(1) < prompt_ids.size(1) + max_new:
        runs += 1
        seq_len = gen_ids.size(1)

        # Buffers for proposals and probabilities
        prop_buffer = torch.empty((1, K), dtype=torch.long, device=device)
        q_buffer    = torch.empty((1, K), dtype=torch.float, device=device)

        # -------- Draft proposes -----------------------------------------
        step_hist_layers = None           # we'll allocate on first step
        final_states_drf = None
        prop_buffer      = torch.empty(1, K, dtype=torch.long, device=device)

        last_tok = gen_ids[:, -1:]

        for i in range(K):
            drf_out = draft(
                inputs_embeds=draft.get_input_embeddings()(last_tok),
                cache_params=draft_cache,
                use_cache=True, 
                cache_fwd=True,
                return_dict=True,      
                cache_position=seq_len + i  # Position for the i-th proposed token
            )
            draft_cache = drf_out.cache_params

            logits_step = drf_out.logits[:, -1]      # (1, vocab_size)
            probs_step  = logits_step.softmax(-1)    # (1, vocab_size)

            next_tok     = logits_step.argmax(-1, keepdim=True)
            prop_buffer[:, i:i+1] = next_tok         # write in‑place buffer

            # record drafter probability q
            q_step = probs_step.gather(-1, next_tok).squeeze(-1)  # (1,)
            q_buffer[:, i] = q_step

            # allocate history tensors on first iteration
            if step_hist_layers is None:
                step_hist_layers = [
                    torch.empty((K, *s.shape), dtype=s.dtype, device=s.device)
                    for s in drf_out.final_states
                ]
            # store this step's hidden states
            for l, st in enumerate(drf_out.final_states):
                step_hist_layers[l][i] = st

            final_states_drf = drf_out.final_states
            last_tok = next_tok

        prop = prop_buffer # (1,K)
        step_hist_drf = step_hist_layers # list(L) of in‑place tensors

        # -------- Target verifies *once* ---------------------------------
        embeds = target.get_input_embeddings()(prop)
        tgt_out = target(
            inputs_embeds=embeds,
            cache_params=tgt_cache,
            use_cache=True,
            cache_fwd=True,
            return_dict=True,
            return_states=True,          
            return_final=True,
            cache_position=seq_len  # Starting position for the K tokens
        )
        logits   = tgt_out.logits # (1, K, vocab_size)
        probs    = logits.softmax(-1) # (1, K, vocab_size)

        # record target probabilities p
        p_buffer = probs.gather(-1, prop.unsqueeze(-1)).squeeze(-1)  # (1, K)

        # -------- Acceptance test ----------------------------------------
        # Rejection-sampling acceptance test: u <= p/q (u is from uniform distribution) 
        u = torch.rand_like(p_buffer)
        ratios = p_buffer / q_buffer
        good = u <= ratios
        m = good.cumprod(-1).sum().item()

        total_accept_rate += m / K
        # print("Acceptance rate: ", m/K)

        # -------- Commit + cache bookkeeping -----------------------------
        # Commit accepted tokens into gen_ids
        if m > 0:
            acc = prop[:, :m]
            gen_ids = torch.cat([gen_ids, acc], dim=1)

        # update both caches with accepted histories
        _commit_prefix(tgt_cache, tgt_out.step_states, tgt_out.final_states, m, K)
        _commit_prefix(draft_cache, step_hist_drf, final_states_drf, m, K)

        # -------- Divergence seed ----------------------------------------
        # 2-e. On first rejection, take target's own next token and advance cache
        if m < K:
            seed_tok = logits[:, m].argmax(-1, keepdim=True)
            gen_ids  = torch.cat([gen_ids, seed_tok], dim=1)

            nxt = target(
                inputs_embeds = target.get_input_embeddings()(seed_tok),
                cache_params = tgt_cache,
                use_cache = True,
                cache_fwd = True,
                return_dict = True,
                cache_position = seq_len + m,
            )
            tgt_cache = nxt.cache_params
        # loop

    avg_rate = total_accept_rate / runs if runs > 0 else 0.0
    print(f"Average acceptance rate: {avg_rate:.3f}")
    
    return gen_ids[:, prompt_ids.size(1):]     # new tokens only

@torch.inference_mode()
def mamba_vanilla_decode(
    target: Mamba2ForCausalLM,
    prompt_ids: torch.Tensor,
    max_new: int = 256
):
    gen_ids = prompt_ids.clone()  # (1, L0)

    # Warm-up forward on target (same as speculative)
    out = target(
        input_ids=prompt_ids,
        use_cache=True,
        return_dict=True
    )
    tgt_cache = out.cache_params  # Mamba2Cache

    # Generate tokens one by one
    current_pos = prompt_ids.size(1)  # Initial position after prompt
    for _ in range(max_new):
        last_tok = gen_ids[:, -1:]  # (1, 1)
        out = target(
            inputs_embeds=target.get_input_embeddings()(last_tok),
            cache_params=tgt_cache,
            use_cache=True,
            return_dict=True,
            cache_position=current_pos,  # Position for the new token

            cache_fwd=True
        )
        logits = out.logits[:, -1]  # (1, V)
        next_tok = logits.argmax(-1, keepdim=True)  # (1, 1)
        gen_ids = torch.cat([gen_ids, next_tok], dim=1)
        current_pos += 1  # Increment position
        tgt_cache = out.cache_params  # Update cache

    return gen_ids[:, prompt_ids.size(1):]  # New tokens only