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

from decoding import _prune_target_cache


def run_draft_test(draft, tok_drf, prompt_ids, gen_ids, max_new_tokens, device):
    print("Running draft test...")
    draft_hist_caches = []
    prompt_len = len(prompt_ids[0])
    cur_drft_start_pos, cur_drft_end_pos = 0, prompt_len
    dft_gen_ids = gen_ids.clone()
    rewind_gen_ids = gen_ids.clone()
    draft_cache = None

    # Generate token by token and store cache
    for i in range(max_new_tokens):
        pos = torch.arange(cur_drft_start_pos, cur_drft_end_pos, device=device)
        dr_out = draft(
            input_ids=dft_gen_ids[..., cur_drft_start_pos:cur_drft_end_pos],
            cache_params=draft_cache,
            use_cache=True,  
            return_dict=True,
            cache_position=pos
        )
        draft_cache = dr_out.cache_params

        draft_logits  = dr_out.logits[:, -1, :] 
        next_tok     = draft_logits.argmax(-1, keepdim=True)   # greedy draft

        draft_hist_caches.append(draft_cache)
        dft_gen_ids[0, cur_drft_end_pos] = next_tok.squeeze(-1)

        cur_drft_start_pos = cur_drft_end_pos
        cur_drft_end_pos += 1

    # Truncate and rewind the cache
    new_tokens_len = max_new_tokens - prompt_len
    truncate_len = new_tokens_len // 2
    # The draft cache is from prompt_len to prompt_len + new_tokens_len - 1 (The last token is not included)
    draft_cache = draft_hist_caches[-(truncate_len)]

    cur_drft_start_pos = prompt_len + (new_tokens_len - truncate_len)
    cur_drft_end_pos = cur_drft_start_pos + 1
    rewind_gen_ids[0,cur_drft_start_pos] = dft_gen_ids[0,cur_drft_start_pos]
    
    for i in range(truncate_len):
        pos = torch.arange(cur_drft_start_pos, cur_drft_end_pos, device=device)
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
    print("Original output:", tok_drf.decode(gen_ids[0, prompt_len:]))
    print("Rewind output:", tok_drf.decode(rewind_gen_ids[0, prompt_len:]))

    return dft_gen_ids[0,prompt_len:], rewind_gen_ids[0,prompt_len:]

def run_target_test(target, tok_tgt, prompt_ids, gen_ids, dft_proposed_ids, max_new_tokens, device):
    print("Running target test...")
    prompt_len = len(prompt_ids[0])
    new_token_len = len(dft_proposed_ids)
    tgt_gen_ids = gen_ids.clone()
    tgt_gen_ids[0, prompt_len:prompt_len+new_token_len] = dft_proposed_ids

    cur_tgt_start_pos, cur_tgt_end_pos = 0, prompt_len

    # Warm up the target model
    pos  = torch.arange(0, prompt_len-1, device=device)
    tgt_out = target(
        input_ids=tgt_gen_ids[..., :prompt_len-1],
        use_cache=True, 
        return_dict=True, 
        cache_position = pos
    )
    tgt_cache = tgt_out.cache_params
    cur_tgt_start_pos, cur_tgt_end_pos = prompt_len-1, prompt_len

    # Cache scan decoding
    pos = torch.arange(cur_tgt_start_pos, cur_tgt_end_pos+new_token_len, device=device) 
    tgt_out = target(
        input_ids=tgt_gen_ids[..., cur_tgt_start_pos:cur_tgt_end_pos+new_token_len],
        cache_params=tgt_cache,
        use_cache=True, 
        cache_fwd=True,
        return_dict=True, 
        cache_position = pos
    )
    tgt_cache = tgt_out.cache_params
    original_output_ids = tgt_out.logits.argmax(-1, keepdim=True).view(-1)  # flatten all dimensions

    # Truncate and rewind the cache
    new_tokens_len = max_new_tokens - prompt_len
    truncate_len = new_tokens_len // 2
    # The target cache is from prompt_len to prompt_len + new_tokens_len (The last token is included)
    _prune_target_cache(tgt_cache, tgt_out.ssm_steps, tgt_out.conv_steps, truncate_len + 1)

    cur_tgt_start_pos = prompt_len + (new_tokens_len - truncate_len)
    
    pos = torch.arange(cur_tgt_start_pos, cur_tgt_end_pos+new_token_len, device=device) 
    tgt_out = target(
        input_ids=tgt_gen_ids[..., cur_tgt_start_pos:cur_tgt_end_pos+new_token_len],
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
    print("Rewind output:", tok_tgt.decode(rewind_output_ids))

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

    org_tgt_out, rew_tgt_out = run_target_test(target, tok_tgt, prompt_ids, gen_ids, org_dft_out, args.max_new_tokens, device)

if __name__ == "__main__":
    main()

    """
    Example usage:
    python cache_decoding_debug.py \
        --target ./mamba2-2.7b_converted_weights \
        --draft  ./mamba2-130m_converted_weights \
        --prompt "I believe the meaning of life is" \
        --max_new_tokens 32
    """