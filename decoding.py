import torch

import torch
import torch.nn.functional as F

def mamba_spec_decode_seq(target, draft, prompt_ids, K=8, max_new=256):
    device   = prompt_ids.device
    generated = prompt_ids.tolist()[0]

    # 1. Initial target pass ------------------------------------------------
    with torch.no_grad():
        out = target(input_ids=prompt_ids, use_cache=True, return_dict=True)
    prev_cache = out.cache_params

    for _ in range(max_new):
        # 2. Draft K tokens -------------------------------------------------
        draft_ctx  = torch.tensor([generated[-1:]], device=device)
        draft_step = draft.generate(draft_ctx, max_new_tokens=K, do_sample=False) # greedy
        d_tokens   = draft_step[:, 1:] # remove seed token, shape: (batch, K)

        # 3. Target *verification* of the K‑token block ---------------------
        embeds = target.get_input_embeddings()(d_tokens)  # shape: (batch, K, vocab)
        cache_pos = torch.tensor([embeds.size(1)], dtype=torch.long, device=device)

        tgt_out = target(
            input_ids       = None,
            inputs_embeds   = embeds,
            cache_params    = prev_cache,
            use_cache       = True,
            cache_position  = cache_pos,
            cache_fwd       = True,
            return_dict     = True,
        )

        logits_blk = tgt_out.logits          # shape: (batch, K, vocab)
        pred_tok   = logits_blk.argmax(-1)   # shape: (batch, K)

        # 4. Exact‑match verification --------------------------------------
        neq_mask   = (pred_tok != d_tokens).squeeze(0) # shape: (K,)
        mismatch   = torch.where(neq_mask)[0]
        m = mismatch[0].item() if mismatch.numel() else K

        # 4‑a. Accept the matching prefix ----------------------------------
        if m > 0:
            accepted = d_tokens[:, :m] # shape: (batch, m)
            generated.extend(accepted.squeeze(0).tolist())

            # Re‑run cache_fwd **only for the accepted prefix** to keep cache
            # You can also save the cache from the first pass and reuse it here
            acc_emb    = embeds[:, :m, :]
            acc_out    = target(
                input_ids=None, inputs_embeds=acc_emb,
                cache_params=prev_cache, use_cache=True,
                cache_position=torch.tensor([m], device=device),
                cache_fwd=True, return_dict=True
            )
            prev_cache = acc_out.cache_params

        # 4‑b. Divergence handling -----------------------------------------
        # Append the last token generated from the target model (mth token) as new token for the draft model to predict next. 
        if m < K:
            # use target's token ŷₘ
            next_tok   = pred_tok[0, m].item()
            generated.append(next_tok)

            next_emb   = target.get_input_embeddings()(
                torch.tensor([[next_tok]], device=device)
            )
            nxt_out    = target(
                input_ids=None, inputs_embeds=next_emb,
                cache_params=prev_cache, use_cache=True,
                cache_position=torch.tensor([1], device=device),
                cache_fwd=True, return_dict=True
            )
            prev_cache = nxt_out.cache_params

    return generated


# TODO: Haven't tested this function yet. Do not use it.
# Tree‑based speculative decoding: State‑Tree‑Scan (STS)
def mamba_spec_decode_tree(target, draft, prompt_ids,
                           B=4, D=4, temperature=1.0, top_p=0.95):
    """
    Verifies a B^D‑node token tree via State‑Tree‑Scan in D cache_fwd calls.
    """
    device = prompt_ids.device
    tgt_out = target(input_ids=prompt_ids, use_cache=True, return_dict=True)
    states_lvl = tgt_out.cache_params.ssm_state        # (1,H,hdim,S)

    # draft tree -----------------------------------------------------------
    tree_tokens = []      # list of tensors with shape (B**lvl,)
    front = prompt_ids[:, -1:]                         # seed token
    for lvl in range(D):
        # sample B tokens for *each* frontier node
        logits = draft(front)[:, -1, :] / temperature
        probs  = torch.softmax(logits, -1)
        tokens = torch.multinomial(probs, B)           # (batch, B)
        tree_tokens.append(tokens.reshape(-1))         # flatten
        front = tokens.reshape(1, -1)                  # next frontier

    # verification ---------------------------------------------------------
    all_logits = []
    for lvl, toks in enumerate(tree_tokens):
        # replicate parent states to children
        states_lvl = states_lvl.repeat_interleave(B, dim=0)  # (B**lvl,...)
        embeds     = target.get_input_embeddings()(toks.unsqueeze(0)).squeeze(0)
        y, states_lvl = target.backbone.cache_fwd(
            embeds.unsqueeze(0), states_lvl
        )
        all_logits.append(y.squeeze(0))                # (B**lvl,1,H)

    # choose best branch ---------------------------------------------------
    # Example policy: accept longest prefix of top‑p branch
    path_scores = torch.stack(
        [torch.log_softmax(l, -1).gather(-1, t[:, None]).squeeze(-1)
         for l, t in zip(all_logits, tree_tokens)]
    )                                                   # (D, B**lvl)
    best_idx = path_scores.sum(0).argmax()              # highest log‑prob
    best_branch = []
    acc_state   = tgt_out.cache_params.ssm_state
    for lvl in range(D):
        tok = tree_tokens[lvl][best_idx % (B**(lvl+1)) // (B**lvl)]
        best_branch.append(tok.item())
        # update acc_state by replaying only accepted branch
        embed = target.get_input_embeddings()(tok.view(1,1))
        _, acc_state = target.backbone.cache_fwd(embed, acc_state)
    return best_branch, acc_state
