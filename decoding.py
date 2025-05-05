import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
from Mamba2.modeling_mamba2 import Mamba2ForCausalLM, Mamba2Cache
from verification import VerificationStrategy, RatioSamplingStrategy, ExactMatchStrategy


def sample_token(
    logits: torch.Tensor,
    method: str = "greedy",
    top_k: int = 50,
    top_p: float = 0.9,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Utility to choose the next token according to a sampling strategy.

    Args:
        logits (torch.Tensor): Logits for the last position, shape (B, vocab).
        method (str): 'greedy', 'top_k', or 'top_p'.
        top_k (int): K value for top-K sampling (ignored if method != 'top_k').
        top_p (float): Cumulative probability threshold for top-p sampling (ignored if method != 'top_p').
        temperature (float): Softmax temperature (>0). 1 ⇒ no scaling.

    Returns:
        torch.Tensor: Chosen token ids, shape (B, 1).
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    if method not in {"greedy", "top_k", "top_p"}:
        raise ValueError("Unsupported sampling method: {}".format(method))

    # temperature scaling
    logits = logits / temperature

    if method == "greedy":
        return logits.argmax(dim=-1, keepdim=True)

    probs = F.softmax(logits, dim=-1)

    if method == "top_k":
        top_k = max(1, min(top_k, probs.size(-1)))
        values, indices = torch.topk(probs, k=top_k, dim=-1)
        # renormalize
        probs_top_k = values / values.sum(dim=-1, keepdim=True)
        next_token = torch.multinomial(probs_top_k, num_samples=1)
        return indices.gather(-1, next_token)

    # top‑p (nucleus) sampling
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = sorted_probs.cumsum(dim=-1)
    # mask tokens past the nucleus
    nucleus_mask = cumulative_probs <= top_p
    # ensure at least one token is kept
    nucleus_mask[..., 0] = True
    probs_nucleus = sorted_probs * nucleus_mask
    probs_nucleus = probs_nucleus / probs_nucleus.sum(dim=-1, keepdim=True)
    next_token = torch.multinomial(probs_nucleus, num_samples=1)
    return sorted_indices.gather(-1, next_token)


def snapshot_states(cache):
    """Deep-copy-safe snapshot of SSM/conv states for each layer."""
    ssm_snap = tuple(t.clone() for t in cache.ssm_states)
    conv_snap = tuple(t.clone() for t in cache.conv_states)
    return ssm_snap, conv_snap


def _prune_target_cache(cache, ssm_steps, conv_steps, num_tokens_to_prune):
    if num_tokens_to_prune <= 0:
        return
    for l, (ssm, conv) in enumerate(zip(ssm_steps, conv_steps)):
        cache.ssm_states[l].copy_(ssm[:, -num_tokens_to_prune, :, :, :])
        cache.conv_states[l].copy_(conv[:, -num_tokens_to_prune, :, :])


# ----------------------- Speculative decoding ----------------------------
@torch.inference_mode()
def mamba_spec_decode_seq(
    target: Mamba2ForCausalLM,
    draft: Mamba2ForCausalLM,
    prompt_ids: torch.Tensor,
    pad_token_id: int = 0,
    K: int = 3,
    max_new: int = 256,
    verification_strategy: VerificationStrategy = RatioSamplingStrategy(),
    draft_sampling: str = "greedy",  # 'greedy' | 'top_k' | 'top_p'
    draft_top_k: int = 50,
    draft_top_p: float = 0.9,
    draft_temperature: float = 1.0,
    log: bool = False,
    tokenizer=None,
):
    """Speculative decoding with optional sampling in the draft model."""
    target.eval(); draft.eval()

    device = prompt_ids.device
    prompt_len = prompt_ids.size(1)
    max_seq_length = getattr(target.config, "max_position_embeddings", 1024)
    total_len = min(max_seq_length, prompt_len + max_new)

    gen_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=device)
    gen_ids[0, :prompt_len] = prompt_ids.clone()

    cur_drft_start_pos, cur_drft_end_pos = 0, prompt_len
    cur_tgt_start_pos, cur_tgt_end_pos = 0, prompt_len
    draft_cache, tgt_cache = None, None

    # --- Warm‑up target cache on prompt−1 tokens
    tgt_out = target(
        input_ids=gen_ids[..., : prompt_len - 1],
        use_cache=True,
        return_dict=True,
        cache_position=torch.tensor([0], device=device),
    )
    tgt_cache = tgt_out.cache_params
    cur_tgt_start_pos, cur_tgt_end_pos = prompt_len - 1, prompt_len

    total_accept_rate, runs = 0.0, 0

    while cur_tgt_end_pos < total_len:
        runs += 1
        seq_len = cur_tgt_end_pos
        corrected_k = min(K, total_len - seq_len)

        draft_tok_buffer = torch.empty(1, corrected_k, dtype=torch.long, device=device)
        q_buffer = torch.empty(1, corrected_k, dtype=torch.float, device=device)

        ssm_hist: List[Tuple[torch.Tensor]] = []
        conv_hist: List[Tuple[torch.Tensor]] = []

        # ---------------- Draft proposes ------------------------------
        for i in range(corrected_k):
            dr_out = draft(
                input_ids=gen_ids[..., cur_drft_start_pos:cur_drft_end_pos],
                cache_params=draft_cache,
                use_cache=True,
                return_dict=True,
                cache_position=torch.tensor([cur_drft_start_pos], device=device),
            )
            draft_cache = dr_out.cache_params

            draft_logits = dr_out.logits[:, -1, :]
            next_tok = sample_token(
                draft_logits,
                method=draft_sampling,
                top_k=draft_top_k,
                top_p=draft_top_p,
                temperature=draft_temperature,
            )

            draft_prob = F.softmax(draft_logits, dim=-1)
            draft_tok_buffer[:, i : i + 1] = next_tok
            q_buffer[:, i] = draft_prob.gather(-1, next_tok).squeeze(-1)

            s, c = snapshot_states(draft_cache)
            ssm_hist.append(s)
            conv_hist.append(c)

            gen_ids[0, cur_drft_end_pos] = next_tok.squeeze(-1)
            cur_drft_start_pos = cur_drft_end_pos
            cur_drft_end_pos += 1

        if log and tokenizer is not None:
            print(f"[Iter {runs:3d}] Draft tokens: ", tokenizer.decode(draft_tok_buffer[0].tolist()))

        # ---------------- Target verifies once ------------------------
        tgt_out = target(
            input_ids=gen_ids[..., cur_tgt_start_pos : cur_tgt_end_pos + corrected_k],
            cache_params=tgt_cache,
            use_cache=True,
            cache_fwd=True,
            return_dict=True,
            cache_position=torch.tensor([cur_tgt_start_pos], device=device),
        )
        tgt_cache = tgt_out.cache_params

        target_draft_logits = tgt_out.logits[:, :corrected_k, :]
        target_draft_prob = F.softmax(target_draft_logits, dim=-1)
        p_buffer = target_draft_prob.gather(-1, draft_tok_buffer.unsqueeze(-1)).squeeze(-1)
        tgt_token_buffer = target_draft_logits.argmax(-1)

        good, m = verification_strategy.verify(
            draft_tok_buffer, q_buffer, p_buffer, target_draft_logits
        )
        total_accept_rate += m / K

        # ---------------- Commit tokens & cache bookkeeping ----------
        if m == corrected_k:
            if cur_tgt_end_pos + corrected_k < total_len:
                next_token_logits = tgt_out.logits[:, corrected_k, :]
                next_tgt_token = sample_token(next_token_logits, method=draft_sampling, temperature=draft_temperature)
                gen_ids[0, cur_tgt_end_pos + corrected_k] = next_tgt_token
        else:
            gen_ids[0, cur_tgt_end_pos + m] = tgt_token_buffer[0, m]
            gen_ids[0, cur_tgt_end_pos + m + 1 : cur_tgt_end_pos + corrected_k] = pad_token_id

            draft_cache.ssm_states = ssm_hist[-(corrected_k - m)]
            draft_cache.conv_states = conv_hist[-(corrected_k - m)]
            _prune_target_cache(
                tgt_cache, tgt_out.ssm_steps, tgt_out.conv_steps, corrected_k - m + 1
            )

        cur_tgt_end_pos += m + 1
        cur_tgt_start_pos = cur_tgt_end_pos - 1
        cur_drft_start_pos = cur_tgt_start_pos
        cur_drft_end_pos = cur_tgt_end_pos

    avg_rate = total_accept_rate / runs if runs else 0.0
    print(f"Average acceptance rate: {avg_rate:.3f}")

    return gen_ids[:, prompt_len:]


# ---------------------- Vanilla decoding -------------------------------
@torch.inference_mode()
def mamba_vanilla_decode(
    model: Mamba2ForCausalLM,
    prompt_ids: torch.Tensor,
    eos_id: int,
    max_new: int = 256,
    sampling: str = "greedy",  # 'greedy' | 'top_k' | 'top_p'
    top_k: int = 50,
    top_p: float = 0.9,
    temperature: float = 1.0,
):
    device = prompt_ids.device
    prompt_len = prompt_ids.size(1)

    # Prefill on prompt (except last token)
    if prompt_len > 1:
        out = model(
            input_ids=prompt_ids[:, :-1],
            use_cache=True,
            return_dict=True,
            cache_position=torch.tensor([0], device=device),
        )
        cache = out.cache_params
        cur_pos = prompt_len - 1
        next_input = prompt_ids[:, -1:]
    else:
        cache = None
        cur_pos = 0
        next_input = prompt_ids

    generated = []

    for _ in range(max_new):
        out = model(
            input_ids=next_input,
            cache_params=cache,
            use_cache=True,
            return_dict=True,
            cache_position=torch.tensor([cur_pos], device=device),
        )
        cache = out.cache_params
        logits = out.logits[:, -1, :]
        next_token = sample_token(
            logits, method=sampling, top_k=top_k, top_p=top_p, temperature=temperature
        )

        if next_token.item() == eos_id:
            break

        generated.append(next_token)
        next_input = next_token
        cur_pos += 1

    return torch.cat(generated, dim=1) if generated else torch.empty(1, 0, dtype=torch.long, device=device)
