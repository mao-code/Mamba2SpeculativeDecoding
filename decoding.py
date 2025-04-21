import torch

import torch
import torch.nn.functional as F
from Mamba2.modeling_mamba2 import Mamba2ForCausalLM, Mamba2Cache
from typing import Tuple

def _commit_prefix(cache, step_hist, final_hist, m, K):
    if m == 0:                       # nothing accepted
        return
    tgt = final_hist if m == K else [h[:, m-1] for h in step_hist]
    for l, st in enumerate(tgt):
        cache.ssm_states[l] = st     # in‑place, O(1)

# decoding_fast.py  (only the core loop shown)
@torch.inference_mode()
def mamba_spec_decode_seq(
    target: Mamba2ForCausalLM,
    draft : Mamba2ForCausalLM,
    prompt_ids: torch.Tensor,
    K: int = 8,
    max_new: int = 256,
    tau: float = 0.5
):

    device = prompt_ids.device
    gen_ids = prompt_ids.clone()               # (1, L0)
    # 1‑a. warm‑up forward on target
    out = target(
        input_ids=prompt_ids,
        use_cache=True, 
        return_dict=True
    )
    tgt_cache = out.cache_params               # Mamba2Cache

    # 1‑b. give the *same* prompt to draft so it has a cache too
    draft_out = draft(
        input_ids=prompt_ids,
        use_cache=True, 
        return_dict=True
    )
    draft_cache = draft_out.cache_params

    while gen_ids.size(1) < prompt_ids.size(1) + max_new:
        seq_len = gen_ids.size(1)  # Current sequence length

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
                return_dict=True,      # logits & final_states only
                cache_position=seq_len + i  # Position for the i-th proposed token
            )

            draft_cache = drf_out.cache_params
            logits_step  = drf_out.logits[:, -1]     # (1,V)
            next_tok     = logits_step.argmax(-1, keepdim=True)
            prop_buffer[:, i:i+1] = next_tok         # write in‑place buffer

            # allocate hist tensors once we know shape
            if step_hist_layers is None:
                step_hist_layers = [
                    # now each has shape (K, 24, 64, 128)
                    torch.empty((K, *s.shape), dtype=s.dtype, device=s.device)
                    for s in drf_out.final_states
                ]

            for l, st in enumerate(drf_out.final_states):
                step_hist_layers[l][i] = st       # in‑place

            final_states_drf = drf_out.final_states  # state after last tok
            last_tok = next_tok

        prop           = prop_buffer                 # (1,K)
        step_hist_drf  = step_hist_layers            # list(L) of in‑place tensors

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
        logits   = tgt_out.logits             # (1,K,V)
        probs    = logits.softmax(-1)

        # -------- Acceptance test ----------------------------------------
        conf_ok  = probs.gather(-1, prop.unsqueeze(-1)).squeeze(-1) > tau # if probability of proposed tokens in the target model is higher than tau
        eq_ok    = logits.argmax(-1).eq(prop) # two models' highest probs
        good     = conf_ok & eq_ok
        m        = good.cumprod(-1).sum().item() # number of accepted tokens (everything hits 0 becomes 0 after it)

        # -------- Commit + cache bookkeeping -----------------------------
        if m:
            acc_ids   = prop[:, :m]           # accepted tokens
            gen_ids   = torch.cat([gen_ids, acc_ids], dim=1)

        _commit_prefix(
            tgt_cache,
            tgt_out.step_states,      # tuple(L)
            tgt_out.final_states,     # tuple(L)
            m, K
        )

        _commit_prefix(
            draft_cache,   
            step_hist_drf,
            final_states_drf,
            m, K
        )

        # -------- Divergence seed ----------------------------------------
        if m < K:
            # append target's own token (ŷₘ) (the seed)
            next_tok = logits[:, m].argmax(-1, keepdim=True)
            gen_ids  = torch.cat([gen_ids, next_tok], dim=1)

            # advance cache by one real token
            nxt = target(
                inputs_embeds=target.get_input_embeddings()(next_tok),
                cache_params=tgt_cache,
                use_cache=True,
                cache_fwd=True,
                return_dict=True
            )
            tgt_cache = nxt.cache_params       # inplace but keep reference

        # loop

    return gen_ids[:, prompt_ids.size(1):]     # new tokens only

@torch.inference_mode()
def mamba_vanilla_decode(
    target: Mamba2ForCausalLM,
    prompt_ids: torch.Tensor,
    max_new: int = 256
):
    device = prompt_ids.device
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