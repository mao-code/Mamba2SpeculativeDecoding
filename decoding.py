import torch

import torch
import torch.nn.functional as F
from Mamba2.modeling_mamba2 import Mamba2ForCausalLM, Mamba2Cache
from typing import Tuple, List

def _commit_prefix(cache, step_hist, final_hist, m, K):
    """
    Copy accepted SSM states from verifier to cache **in‑place**.
    Rewind to the prefix state when no token is accepted (m == 0).
    """
    if m == 0:
        # ---- Rewind to the state *before* any speculative step -----
        # time‑step 0 in step_hist is exactly the state after the
        # last committed token from the previous outer iteration.
        for l, st in enumerate(step_hist):           # st shape: (K, B, …)
            cache.ssm_states[l].copy_(st[0])         # back to prefix
        return

    if m == K:
        src = final_hist                             # already advanced K
    else:
        # take the state *after* the (m‑th) accepted token
        src = [h[m - 1] for h in step_hist]

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

    tE = target.get_input_embeddings()
    dE = draft.get_input_embeddings()

    total_accept_rate = 0
    runs = 0

    while gen_ids.size(1) < prompt_ids.size(1) + max_new:
        runs += 1
        seq_len = gen_ids.size(1)

        remaining      = prompt_ids.size(1) + max_new - seq_len
        corrected_k    = min(K, remaining)

        # Buffers for proposals and probabilities
        prop_buffer    = torch.empty(1, corrected_k, dtype=torch.long,   device=device)
        q_buffer       = torch.empty(1, corrected_k, dtype=torch.float,  device=device)

        # -------- Draft proposes -----------------------------------------
        last_tok = gen_ids[:, -1:]                       # start from context tail
        step_hist_layers, final_states_drf = None, None  # will be filled lazily

        for i in range(corrected_k):
            dr_out = draft(
                inputs_embeds=dE(last_tok),
                cache_params=draft_cache,
                use_cache=True,  cache_fwd=True,
                return_dict=True,
                cache_position=seq_len + i
            )
            draft_cache = dr_out.cache_params

            logits_step  = dr_out.logits[:, -1]          # (1, V)
            probs_step   = logits_step.softmax(-1)
            next_tok     = logits_step.argmax(-1, keepdim=True)   # greedy draft
            prop_buffer[:, i : i + 1] = next_tok
            q_buffer  [:, i] = probs_step.gather(-1, next_tok).squeeze(-1)

            # store hidden‑state snapshots for *every* layer
            if step_hist_layers is None:
                step_hist_layers = [
                    torch.empty((corrected_k, *s.shape), dtype=s.dtype, device=s.device)
                    for s in dr_out.final_states
                ]
            for l, st in enumerate(dr_out.final_states):
                step_hist_layers[l][i] = st

            final_states_drf = dr_out.final_states
            last_tok = next_tok      # feed the just‑generated token back in

        # -------- Target verifies *once* ---------------------------------
        # We must include the *last* real context token so that the model
        # produces logits for **each** proposal in positions
        #    [seq_len‑1 , … , seq_len+corrected_k‑1].
        #
        # We therefore send embeddings for  ( ctx_last ⊕ prop_buffer )
        ctx_last   = gen_ids[:, -1:]
        embeds_all = torch.cat([
            tE(ctx_last),
            tE(prop_buffer)
        ], dim=1)                                        # (1, 1+corrected_k, D)


        tgt_out = target(
            inputs_embeds = embeds_all,
            cache_params  = tgt_cache,
            use_cache=True, cache_fwd=True,
            return_dict=True, return_states=True, return_final=True,
            cache_position = seq_len - 1                 # start one step earlier
        )
        tgt_cache = tgt_out.cache_params

        # logits.shape = (1 , 1+corrected_k , V)
        #  └─ index 0   = ctx_last     (discard)
        #  └─ index 1+i = proposal i
        logits_prop  = tgt_out.logits[:, 1:]             # keep only proposals
        probs_prop   = logits_prop.softmax(-1)           # (1, corrected_k, V)
        p_buffer     = probs_prop.gather(-1, prop_buffer.unsqueeze(-1)).squeeze(-1)  # (1, k)

        # -------- Acceptance test ----------------------------------------
        # Rejection-sampling acceptance test: u <= p/q (u is from uniform distribution) 
        u = torch.rand_like(p_buffer)
        ratios = p_buffer / q_buffer
        good = u <= ratios
        m = good.cumprod(-1).sum().item()

        total_accept_rate += m / K

        # -------- Commit + cache bookkeeping -----------------------------
        # Commit accepted tokens into gen_ids
        if m:
            gen_ids = torch.cat([gen_ids, prop_buffer[:, :m]], dim=1)

        # update both caches with accepted histories
        _commit_prefix(tgt_cache, tgt_out.step_states, tgt_out.final_states, m, corrected_k)
        _commit_prefix(draft_cache, step_hist_layers, final_states_drf, m, corrected_k)

        # -------- Divergence seed ----------------------------------------
        # divergence step  (if <k accepted) take target's own token
        if m < corrected_k:
            seed_tok  = logits_prop[:, m].argmax(-1, keepdim=True)
            gen_ids   = torch.cat([gen_ids, seed_tok], dim=1)

            # advance both caches with the *same* seed token
            tgt_cache = target(
                inputs_embeds = tE(seed_tok),
                cache_params  = tgt_cache,
                use_cache=True, cache_fwd=True,
                return_dict=True,
                cache_position = seq_len + m
            ).cache_params

            draft_cache = draft(
                inputs_embeds = dE(seed_tok),
                cache_params  = draft_cache,
                use_cache=True, cache_fwd=True,
                return_dict=True,
                cache_position = seq_len + m
            ).cache_params

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