import torch

import torch
import torch.nn.functional as F
from Mamba2.modeling_mamba2 import Mamba2ForCausalLM, Mamba2Cache
from typing import Tuple, List
from verification import VerificationStrategy, RatioSamplingStrategy, ExactMatchStrategy

def _prune_cache(cache, step_hist, m, K):
    if m == 0:
        # ---- Rewind to the state *before* any speculative step -----
        for l, st in enumerate(step_hist):           # st shape: (B, K+1, …) (last_ctx + K tokens)
            cache.ssm_states[l].copy_(st[:, 0])      # back to prefix
        return

    elif 1 <= m and m < K:
        # take cache that has seen m tokens 
        src = [h[:, m-1] for h in step_hist]

        for l, st in enumerate(src):
            cache.ssm_states[l].copy_(st)

# decoding_fast.py  (only the core loop shown)
@torch.inference_mode()
def mamba_spec_decode_seq(
    target: Mamba2ForCausalLM,
    draft : Mamba2ForCausalLM,
    prompt_ids: torch.Tensor,
    pad_token_id: int = 0,
    eos_tokens_id: int | List[int] = 1,
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
    prompt_len = prompt_ids.size(1)  # length of the prompt
    max_seq_length = target.config.max_position_embeddings if hasattr(target.config, 'max_position_embeddings') else (target.config.max_context_length if hasattr(target.config, 'max_context_length') else 1024)
    total_len = min(max_seq_length, prompt_len + max_new)
    
    tgt_cache, draft_cache = None, None
    gen_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=target.device)
    gen_ids[0, :prompt_len] = prompt_ids.clone()

    cur_drft_start_pos, cur_drft_end_pos = 0, prompt_len
    cur_tgt_start_pos, cur_tgt_end_pos = 0, prompt_len

    # tE, dE = target.get_input_embeddings(), draft.get_input_embeddings()

    total_accept_rate, runs = 0.0, 0

    while cur_tgt_end_pos < total_len:
        runs += 1
        seq_len = cur_tgt_end_pos
        corrected_k = min(K, total_len - seq_len)

        # Buffers for proposals and probabilities
        draft_tok_buffer = torch.empty(1, corrected_k, dtype=torch.long, device=device)
        q_buffer = torch.empty(1, corrected_k, dtype=torch.float, device=device)

        # -------- Draft proposes -----------------------------------------
        # last_tok = gen_ids[:, -1:] # start from context tail
        step_hist_layers = None  # will be filled lazily

        for i in range(corrected_k):
            dr_out = draft(
                # inputs_embeds=dE(last_tok),
                input_ids=gen_ids[..., cur_drft_start_pos:cur_drft_end_pos],
                cache_params=draft_cache,
                use_cache=True,  
                cache_fwd=True,
                return_dict=True,
                cache_position=cur_drft_start_pos
            )
            draft_cache = dr_out.cache_params

            draft_logits  = dr_out.logits[..., -1, :] 
            draft_prob   = draft_logits.softmax(-1)
            next_tok     = draft_logits.argmax(-1, keepdim=True)   # greedy draft
            draft_tok_buffer[:, i : i + 1] = next_tok
            q_buffer  [:, i] = draft_prob.gather(-1, next_tok).squeeze(-1)

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
                step_hist_layers[l][:, i] = st

            gen_ids[0, cur_drft_end_pos+i] = next_tok.squeeze(-1)

            cur_drft_start_pos = cur_drft_end_pos
            cur_drft_end_pos += 1

        # -------- Target verifies *once* ---------------------------------
        tgt_out = target(
            input_ids=gen_ids[..., cur_tgt_start_pos:cur_tgt_end_pos+corrected_k],
            cache_params=tgt_cache,
            use_cache=True, 
            cache_fwd=True,
            return_dict=True, 
            return_states=True, 
            return_final=True,
            cache_position = cur_drft_start_pos
        )
        tgt_cache = tgt_out.cache_params

        # The last token is the next prediction of the last token for the draf model. (we discard it)
        target_logits  = tgt_out.logits[..., cur_tgt_end_pos-1:cur_tgt_end_pos+corrected_k-1, :]  # (1, k, V)    
        target_prob   = target_logits.softmax(-1)
        p_buffer     = target_prob.gather(-1, draft_tok_buffer.unsqueeze(-1)).squeeze(-1)  # (1, k)
        tgt_token_buffer = target_logits.argmax(-1)  # (1, k)

        # -------- Acceptance test ----------------------------------------
        # Rejection-sampling acceptance test: u <= p/q (u is from uniform distribution) 
        good, m = verification_strategy.verify(draft_tok_buffer, q_buffer, p_buffer, target_logits)
        total_accept_rate += m / K

        if log:
            # shapes: eq_mask [B, K]
            eq_mask    = draft_tok_buffer.eq(tgt_token_buffer)         # [B, K]

            # eq_mask[0] is a BoolTensor of length K; its cumprod will be 1s up to
            # first zero, then 0s thereafter. Summing gives exactly the prefix-match length.
            prefix_mask = eq_mask[0].cumprod(dim=0)         # [K], 1,1,…,0,0,0
            m = int(prefix_mask.sum().item())

            matched    = draft_tok_buffer[0, :m].tolist()
            mismatched = draft_tok_buffer[0, m:].tolist()

            print(f"[Iter {runs:3d}] prefix-length: {m} | matched: {matched} | "
                f"mismatched: {mismatched} | target: {tgt_token_buffer[0].tolist()} | "
                f"draft: {draft_tok_buffer[0].tolist()}")

        # -------- Commit + cache bookkeeping -----------------------------
        # Commit accepted tokens into gen_ids
        if m == corrected_k:
            last_tgt_token = tgt_out.logits[..., cur_tgt_end_pos+corrected_k-1, :].argmax(-1).view(-1, 1)
            gen_ids[0, cur_tgt_end_pos+corrected_k] = last_tgt_token
        else:
            # gen_ids = torch.cat([gen_ids, draft_tok_buffer[..., :m]], dim=1)
            gen_ids[0, cur_tgt_end_pos:cur_tgt_end_pos+m+1] = tgt_token_buffer[0, :m+1]
            gen_ids[0, cur_tgt_end_pos+m+1:cur_tgt_end_pos+corrected_k] = pad_token_id

        if cur_tgt_end_pos < total_len:
            cur_tgt_end_pos += m+1
            cur_tgt_start_pos = cur_tgt_end_pos-1
            
            cur_drft_start_pos = cur_tgt_start_pos
            cur_drft_end_pos = cur_tgt_end_pos

            _prune_cache(tgt_cache, tgt_out.step_states, m, corrected_k)
            _prune_cache(draft_cache, step_hist_layers, m, corrected_k)

    avg_rate = total_accept_rate / runs if runs > 0 else 0.0
    if log:
        print(f"Average acceptance rate: {avg_rate:.3f}")

    return gen_ids[:, prompt_len:]     # new tokens only

@torch.inference_mode()
def mamba_vanilla_decode(
    target: Mamba2ForCausalLM,
    prompt_ids: torch.Tensor,
    pad_token_id: int = 0,
    max_new: int = 256
):
    target.eval()
    gen_ids = prompt_ids.clone()  # (1, prompt_len)

    prompt_len = prompt_ids.size(1)  # length of the prompt
    max_seq_length = target.config.max_position_embeddings if hasattr(target.config, 'max_position_embeddings') else (target.config.max_context_length if hasattr(target.config, 'max_context_length') else 1024)
    total_len = min(max_seq_length, prompt_len + max_new)

    gen_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=target.device)  
    gen_ids[0, :prompt_len] = prompt_ids.clone()

    # Generate tokens one by one
    current_pos = 0
    for i in range(max_new):
        out = target(
            input_ids=gen_ids,
            use_cache=True,
            return_dict=True,
            cache_position=current_pos,

            cache_fwd=True
        )

        logits = out.logits[:, -1, :]  # (1, V)
        next_tok = logits.argmax(-1, keepdim=True)  # (1, 1)

        current_pos = prompt_len + i
        gen_ids[0, current_pos] = next_tok.squeeze(-1)

    return gen_ids[:, prompt_len:]  # New tokens only