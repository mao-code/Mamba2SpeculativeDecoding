"""
Draft:
1. A prompt input. (P)
2. Generate the output and store the cache. (with a set length output)
3. Truncate and rewind the cache and generate the output again.
4. Compare the output with the original output to see if there are problems in draft model cache management. 

5. Store the proposed tokens in a list for the next target model test (PP)

Target:
1. A prompt input (P), a new prompt input (act as a the proposed token by the draft model) (PP)
2. Use cache_fwd to take the cache of "P" and process "PP" in parallel at once.
3. Compate the output with the output of the draft model to see the match rate.
4. Truncate and rewind the cache and generate the output again.
5. Compare the output with the original output to see if there are problems in target model cache management. 

All the above need to decode the result to see the qualitative result.
"""
import argparse, time, torch
from transformers import AutoTokenizer
from Mamba2.modeling_mamba2 import Mamba2ForCausalLM
from decoding import mamba_spec_decode_seq, mamba_vanilla_decode
from transformers import AutoConfig
from utils import set_random_seed

from decoding import _prune_target_cache, mamba_vanilla_decode, snapshot_states
import copy

def run_draft_test(draft, tok_drf, prompt_ids, gen_ids, max_new_tokens, device):
    print("Running draft test..."+ "="*20)
    prompt_len = prompt_ids.size(1)
    # ------------------------------------------------------
    # (A)  Prefill **up to the penultimate** prompt token
    # ------------------------------------------------------
    if prompt_len > 1:
        draft_out = draft(
            input_ids=prompt_ids[:, :-1],
            use_cache=True,
            return_dict=True,
            cache_position=torch.tensor([0], device=device)
        )
        draft_cache = draft_out.cache_params
        cur_pos = prompt_len - 1
        next_input = prompt_ids[:, -1:]         # the last prompt token
    else:                                       # 1-token prompt
        draft_cache = None
        cur_pos = 0
        next_input = prompt_ids

    dft_gen_ids = gen_ids.clone()                   # working buffer
    ssm_hist, conv_hist = [], []

    # ------------------------------------------------------
    # (B)  Incremental generation
    # ------------------------------------------------------
    for _ in range(max_new_tokens):
        draft_out = draft(
            input_ids=next_input,               # exactly one token
            cache_params=draft_cache,
            use_cache=True,
            return_dict=True,
            cache_position=torch.tensor([cur_pos], device=device)
        )
        draft_cache = draft_out.cache_params
        logits = draft_out.logits[:, -1, :]
        next_tok = logits.argmax(-1, keepdim=True)

        if next_tok.item() == tok_drf.eos_token_id:   # after you obtain next_tok
            dft_gen_ids[0, cur_pos] = next_input.item()
            break

        s, c = snapshot_states(draft_cache)
        ssm_hist.append(s)
        conv_hist.append(c)

        dft_gen_ids[0, cur_pos] = next_input.item()       # record token
        dft_gen_ids[0, cur_pos + 1] = next_tok.item()

        # set up next step
        next_input = next_tok
        cur_pos += 1

    # ------------------------------------------------------
    # (C)  Rewind the cache and generate the output again
    # ------------------------------------------------------
    rewind_gen_ids = gen_ids.clone()                 
    # Truncate and rewind the cache
    truncate_len = max_new_tokens // 2
    # The draft cache is from prompt_len to prompt_len + new_tokens_len - 1 (The last token is not included)
    
    draft_cache.ssm_states  = ssm_hist[-truncate_len]
    draft_cache.conv_states = conv_hist[-truncate_len]

    cur_drft_start_pos = prompt_len + (max_new_tokens - truncate_len)
    cur_drft_end_pos = cur_drft_start_pos + 1
    rewind_gen_ids[0,:cur_drft_start_pos+1] = dft_gen_ids[0,:cur_drft_start_pos+1]

    print("Length of draft ssm hist cache:", len(ssm_hist))
    print("Draft truncate length:", truncate_len)
    print("Rewind draft current start position:", cur_drft_start_pos)
    
    for i in range(truncate_len-1):
        pos = torch.tensor([cur_drft_start_pos], device=device)
        dr_out = draft(
            input_ids=rewind_gen_ids[..., cur_drft_start_pos:cur_drft_end_pos],
            cache_params=draft_cache,
            use_cache=True,  
            return_dict=True,
            cache_position=pos
        )
        draft_cache = dr_out.cache_params

        draft_logits  = dr_out.logits[:, -1, :] 
        next_tok     = draft_logits.argmax(-1, keepdim=True)   # greedy draft

        # draft_hist_caches.append(draft_cache)
        rewind_gen_ids[0, cur_drft_end_pos] = next_tok.squeeze(-1)

        cur_drft_start_pos = cur_drft_end_pos
        cur_drft_end_pos += 1

    # Compare the 2 results
    print("Compared draft output:")
    print("Original output:", tok_drf.decode(dft_gen_ids[0, :]))
    print("\nRewind output:", tok_drf.decode(rewind_gen_ids[0, :]))

    return dft_gen_ids[0,prompt_len:], rewind_gen_ids[0,prompt_len:]

def run_target_test(target, tok_tgt, prompt_ids, gen_ids, dft_proposed_ids, device):
    print("Running target test..." + "="*20)
    prompt_len = prompt_ids.size(1)
    new_token_len = len(dft_proposed_ids)

    tgt_gen_ids = gen_ids.clone()
    tgt_gen_ids[0, prompt_len:prompt_len + new_token_len] = dft_proposed_ids

    # ------------------------------------------------------
    # (A) Prefill up to penultimate prompt token
    # ------------------------------------------------------
    if prompt_len > 1:
        out = target(
            input_ids=prompt_ids[:, :-1],
            use_cache=True,
            return_dict=True,
            cache_position=torch.tensor([0], device=device)
        )
        tgt_cache = out.cache_params
        cur_pos = prompt_len - 1
    else:
        tgt_cache = None
        cur_pos = 0

    # ------------------------------------------------------
    # (B) Feed the whole "prompt + proposals" *once*
    #     (cache_fwd == True does the K-token scan)
    # ------------------------------------------------------
    pos = torch.tensor([cur_pos], device=device)
    org_out = target(
        input_ids=tgt_gen_ids[:, cur_pos: prompt_len + new_token_len],
        cache_params=tgt_cache,
        use_cache=True,
        cache_fwd=True,
        return_dict=True,
        cache_position=pos
    )
    tgt_cache = org_out.cache_params                 
    original_output_ids = org_out.logits.argmax(-1, keepdim=True).view(-1)

    # ------------------------------------------------------
    # (C) Rewind the cache and generate the output again
    # ------------------------------------------------------
    # Truncate and rewind the cache
    truncate_len = new_token_len // 2
    # The target cache is from prompt_len to prompt_len + new_tokens_len (The last token is included)
    _prune_target_cache(tgt_cache, org_out.ssm_steps, org_out.conv_steps, truncate_len + 1)

    cur_pos = prompt_len + (new_token_len - truncate_len)

    print("Length of target cache:", org_out.ssm_steps[0].size(1))
    print("Target truncate length:", truncate_len)
    print("Rewind target current position:", cur_pos)
    
    pos = torch.arange(cur_pos, prompt_len+new_token_len, device=device) 
    print("Rewind target input: ", tok_tgt.decode(tgt_gen_ids[0, cur_pos:prompt_len+new_token_len]))
    tgt_out = target(
        input_ids=tgt_gen_ids[..., cur_pos:prompt_len+new_token_len],
        cache_params=tgt_cache,
        use_cache=True, 
        cache_fwd=True,
        return_dict=True, 
        cache_position = pos
    )
    tgt_cache = tgt_out.cache_params
    rewind_output_ids = tgt_out.logits.argmax(-1, keepdim=True).view(-1)  # flatten all dimensions

    # Compare the 2 results
    print("Compared target output:")
    print("Original output:", tok_tgt.decode(original_output_ids))
    print("\nRewind output:", tok_tgt.decode(rewind_output_ids))

    return original_output_ids, rewind_output_ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target",  required=True, help="Path or HF-hub id of target model")
    ap.add_argument("--draft",   required=True, help="Path or HF-hub id of smaller draft model")
    ap.add_argument("--prompt",  default="I believe the meaning of life is")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda:0")

    # Set random seed
    set_random_seed(42)

    # ------------------- Load models -------------------
    print("Loading models...")
    target_config = AutoConfig.from_pretrained(
        args.target,
        trust_remote_code=True,    
    )
    target = Mamba2ForCausalLM.from_pretrained(
        args.target, 
        config=target_config, 
        torch_dtype=torch.float16, 
        trust_remote_code=True
    ).to(device).eval()
    target = target.to(torch.float32) # Triton donesn't support float16 on V100 yet

    draft_config = AutoConfig.from_pretrained(
        args.draft,
        trust_remote_code=True,
    )
    draft  = Mamba2ForCausalLM.from_pretrained(
        args.draft, 
        config=draft_config, 
        torch_dtype=torch.float16, 
        trust_remote_code=True
    ).to(device).eval()
    draft  = draft.to(torch.float32)

    tok_tgt = AutoTokenizer.from_pretrained(args.target)
    tok_drf = AutoTokenizer.from_pretrained(args.draft)
    assert tok_tgt.vocab_size == tok_drf.vocab_size, "Vocab size mismatch between target and draft models"

    encoding = tok_tgt(
        args.prompt,
        return_tensors="pt",
        padding=True,           # pad to longest sequence
        truncation=True,
        return_attention_mask=True
    )
    prompt_ids = encoding.input_ids.to(device)

    gen_ids = torch.full((1, len(prompt_ids[0]) + args.max_new_tokens), tok_tgt.pad_token_id, dtype=torch.long, device=target.device)
    prompt_len = len(prompt_ids[0])
    gen_ids[0, :prompt_len] = prompt_ids.clone()

    # ------------------- Testing -------------------
    org_dft_out, rew_dft_out = run_draft_test(draft, tok_drf, prompt_ids, gen_ids, args.max_new_tokens, device)

    org_tgt_out, rew_tgt_out = run_target_test(target, tok_tgt, prompt_ids, gen_ids, org_dft_out, device)

    print("Vanilla output with cache store: (target model)" + "="*20)
    vanilla_out = mamba_vanilla_decode(
        target, prompt_ids, eos_id=tok_tgt.eos_token_id, max_new=args.max_new_tokens
    )
    print(tok_tgt.decode(vanilla_out.view(-1)))

    print("Vanilla output without cache store: (target model)" + "="*20)
    no_cache_vanilla_output = ""
    no_cache_vanilla_output_ids = torch.empty(args.max_new_tokens, dtype=torch.long, device=device)
    input_ids = prompt_ids.clone()
    for i in range(args.max_new_tokens):    
        vanilla_out = target(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )

        logits = vanilla_out.logits[:, -1, :]

        # Sampling
        # probs = torch.nn.functional.softmax(logits, dim=-1)
        # next_token = torch.multinomial(probs, num_samples=1)

        # Greedy
        next_token = logits.argmax(-1, keepdim=True)   # greedy draft

        no_cache_vanilla_output_ids[i] = next_token.item()
        input_ids = torch.cat([input_ids, next_token], dim=1)
        no_cache_vanilla_output += tok_tgt.decode(next_token[0])

    print(no_cache_vanilla_output)

    print("Vanilla output with cache store: (draft model)" + "="*20)
    vanilla_out = mamba_vanilla_decode(
        draft, prompt_ids, eos_id=tok_drf.eos_token_id, max_new=args.max_new_tokens
    )
    print(tok_drf.decode(vanilla_out.view(-1)))

    print("Vanilla output without cache store: (draft model)" + "="*20)
    no_cache_vanilla_output = ""
    no_cache_vanilla_output_ids = torch.empty(args.max_new_tokens, dtype=torch.long, device=device)
    input_ids = prompt_ids.clone()
    for i in range(args.max_new_tokens):    
        vanilla_out = draft(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )

        logits = vanilla_out.logits[:, -1, :]

        # Sampling
        # probs = torch.nn.functional.softmax(logits, dim=-1)
        # next_token = torch.multinomial(probs, num_samples=1)

        # Greedy
        next_token = logits.argmax(-1, keepdim=True)   # greedy draft

        no_cache_vanilla_output_ids[i] = next_token.item()
        input_ids = torch.cat([input_ids, next_token], dim=1)
        no_cache_vanilla_output += tok_tgt.decode(next_token[0])

    print(no_cache_vanilla_output)

if __name__ == "__main__":
    main()

    """
    Example usage:
    python -m script.cache_decoding_debug \
        --target ./mamba2-2.7b_converted_weights \
        --draft  ./mamba2-130m_converted_weights \
        --prompt "I believe the meaning of life is" \
        --max_new_tokens 32
    """