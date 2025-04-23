import torch

import torch
import torch.nn.functional as F
from Mamba2.modeling_mamba2 import Mamba2ForCausalLM, Mamba2Cache
from typing import Tuple, List
from verification import VerificationStrategy, RatioSamplingStrategy, ExactMatchStrategy

def _commit_prefix(cache, step_hist, final_hist, m, K):
    if m == 0:
        # ---- Rewind to the state *before* any speculative step -----
        for l, st in enumerate(step_hist):           # st shape: (B, K+1, …) (last_ctx + K tokens)
            cache.ssm_states[l].copy_(st[:, 0])      # back to prefix
        return

    if m == K:
        src = final_hist
    else:
        # take the state *after* the (m‑th) accepted token
        src = [h[:, m - 1] for h in step_hist]

    for l, st in enumerate(src):
        cache.ssm_states[l].copy_(st)

# decoding_fast.py  (only the core loop shown)
@torch.inference_mode()
def mamba_spec_decode_seq(
    target: Mamba2ForCausalLM,
    draft : Mamba2ForCausalLM,
    prompt_ids: torch.Tensor,
    K: int = 3,
    max_new: int = 256,
    verification_strategy: VerificationStrategy = RatioSamplingStrategy(),
    log: bool = False
):
    """
    q-distribution is the softmax of the logits from the draft model.
    p-distribution is the softmax of the logits from the target model.
    """
    
    # ensure models are in eval mode
    target.eval()
    draft.eval()

    device = prompt_ids.device
    L0 = prompt_ids.size(1)  # length of the prompt
    
    # 1. Warm-up: run both models on the prompt to fill their caches
    # Prevent using the last token of the prompt (cache: 0...L0-2)
    tgt_cache = target(input_ids=prompt_ids[..., :-1], use_cache=True, return_dict=True).cache_params
    draft_cache = draft(input_ids=prompt_ids[..., :-1], use_cache=True, return_dict=True).cache_params
    gen_ids = prompt_ids.clone()

    tE = target.get_input_embeddings()
    dE = draft.get_input_embeddings()

    total_accept_rate = 0
    runs = 0

    while gen_ids.size(1) < L0 + max_new:
        runs += 1
        seq_len = gen_ids.size(1)
        corrected_k = min(K, L0 + max_new - seq_len)

        # Buffers for proposals and probabilities
        prop_buffer = torch.empty(1, corrected_k, dtype=torch.long, device=device)
        q_buffer = torch.empty(1, corrected_k, dtype=torch.float, device=device)

        # -------- Draft proposes -----------------------------------------
        last_tok = gen_ids[:, -1:]                       # start from context tail
        step_hist_layers, final_states_drf = None, None  # will be filled lazily

        for i in range(corrected_k):
            dr_out = draft(
                inputs_embeds=dE(last_tok),
                cache_params=draft_cache,
                use_cache=True,  
                cache_fwd=True,
                return_dict=True,
                cache_position=seq_len + i - 1
            )
            draft_cache = dr_out.cache_params

            logits_step  = dr_out.logits[:, -1]          # (1, V)
            probs_step   = logits_step.softmax(-1)
            next_tok     = logits_step.argmax(-1, keepdim=True)   # greedy draft
            prop_buffer[:, i : i + 1] = next_tok
            q_buffer  [:, i] = probs_step.gather(-1, next_tok).squeeze(-1)

            # store hidden‑state snapshots for *every* layer
            if step_hist_layers is None:
                batch, nheads, head_dim, dstate = dr_out.final_states[0].shape
                step_hist_layers = tuple(
                    torch.empty(
                        (batch, corrected_k, nheads, head_dim, dstate),
                        dtype=st.dtype,
                        device=st.device,
                    )
                    for st in dr_out.final_states # number of layers
                )
            for l, st in enumerate(dr_out.final_states):
                step_hist_layers[l][i] = st

            final_states_drf = dr_out.final_states
            last_tok = next_tok # feed the just‑generated token back in

        # -------- Target verifies *once* ---------------------------------
        ctx_last   = gen_ids[:, -1:]
        embeds_all = torch.cat([tE(ctx_last), tE(prop_buffer)], dim=1) # (1, 1+corrected_k, D)

        tgt_out = target(
            inputs_embeds=embeds_all,
            cache_params=tgt_cache,
            use_cache=True, 
            cache_fwd=True,
            return_dict=True, 
            return_states=True, 
            return_final=True,
            cache_position = seq_len - 1
        )
        tgt_cache = tgt_out.cache_params

        # step_states: (layers, st shape)
        # 64 layers (elements in a tuple), (1, 4, 80, 64, 128)

        if log:
            print("Length of the step state of the target model: ", len(tgt_out.step_states))
            print("Size of the first element state of the target model: ", tgt_out.step_states[0].size())
            print("Length of the final state of the target model: ", len(tgt_out.final_states))
            print("Size of the final sate of the target model: ", tgt_out.final_states[0].size())
            print("Size of the first layer hist states of the draft model (K,st shape): ", step_hist_layers[0].size())

            print("="*20)
        
        
        # The last token is the next prediction of the last token for the draf model. (we discard it)
        logits_prop  = tgt_out.logits[:, :-1, :]  # (1, k-1, V)      
        probs_prop   = logits_prop.softmax(-1)
        p_buffer     = probs_prop.gather(-1, prop_buffer.unsqueeze(-1)).squeeze(-1)  # (1, k)

        if log:
            print("Size of the raw output of the target model", tgt_out.logits.size())
            print("Size of logits_prop", logits_prop.size())
            print("Length of prop_buffer", prop_buffer.size(1))
            print("Length of p_buffer", p_buffer.size(1))
            print("Length of q_buffer", q_buffer.size(1))

        # -------- Acceptance test ----------------------------------------
        # Rejection-sampling acceptance test: u <= p/q (u is from uniform distribution) 
        good, m = verification_strategy.verify(prop_buffer, q_buffer, p_buffer, logits_prop)
        total_accept_rate += m / K

        if log:
            top_tokens = logits_prop.argmax(dim=-1)
            eq_mask = prop_buffer.eq(top_tokens)
            matched   = prop_buffer[0][eq_mask[0]].tolist()
            mismatched= prop_buffer[0][~eq_mask[0]].tolist()
            print(f"[Iter {runs:3d}] matched proposals: {matched} | mismatched: {mismatched} | target: {top_tokens[0].tolist()} | draft: {prop_buffer[0].tolist()}")

        # -------- Commit + cache bookkeeping -----------------------------
        # Commit accepted tokens into gen_ids
        if m:
            gen_ids = torch.cat([gen_ids, prop_buffer[..., :m]], dim=1)

        if gen_ids.size(1) < L0 + max_new:
            # update both caches with accepted histories
            _commit_prefix(tgt_cache, tgt_out.step_states, tgt_out.final_states, m, corrected_k)
            _commit_prefix(draft_cache, step_hist_layers, final_states_drf, m, corrected_k)

        # -------- Divergence seed ----------------------------------------
        # divergence step  (if <k accepted) take target's own token
        if gen_ids.size(1) < L0 + max_new and m < corrected_k:
            seed_tok  = logits_prop[:, m-1].argmax(-1, keepdim=True)
            gen_ids   = torch.cat([gen_ids, seed_tok], dim=1)

    avg_rate = total_accept_rate / runs if runs > 0 else 0.0
    if log:
        print(f"Average acceptance rate: {avg_rate:.3f}")

    return gen_ids[:, L0:]     # new tokens only

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