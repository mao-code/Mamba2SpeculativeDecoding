"""
Vanilla Decoding v.s. Cache Scan Decoding
Use different input prompt and see the first token
"""

import argparse
from utils import set_random_seed
import torch
from transformers import AutoTokenizer, AutoConfig
from mamba2.modeling_mamba2 import Mamba2ForCausalLM
from mamba2.utils import snapshot_states

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  required=True, help="Path or HF-hub id of target model")
    args = ap.parse_args()

    device = torch.device("cuda:0")

    # Set random seed
    set_random_seed(42)

    # ------------------- Load models -------------------
    print("Loading models...")
    model_config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,    
    )
    model = Mamba2ForCausalLM.from_pretrained(
        args.model, 
        config=model_config, 
        torch_dtype=torch.float16, 
        trust_remote_code=True
    ).to(device).eval()
    model = model.to(torch.float32) # Triton donesn't support float16 on V100 yet
    tok = AutoTokenizer.from_pretrained(args.model)

    # ------------------- Testing -------------------
    prompts = [
        "I believe the meaning of life is",
        "The capital of France is",
        "The sun rises in the",
        "The quick brown fox jumps over the lazy dog",
        "In a galaxy far, far away",
        "Once upon a time in a land far away",
        "The future of AI is",
    ]

    with torch.no_grad():
        for prompt in prompts:
            print("Prompt:", prompt)
            encoding = tok(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = encoding.input_ids.to(device)

            # Warm up prompt_len-1 tokens
            pre_out = model(
                input_ids=input_ids[:, :-1],
                use_cache=True,
                return_dict=True,
                cache_position=torch.tensor([0], device=device),
            )
            cache = pre_out.cache_params
            last_idx = len(input_ids[0])-1
            orig_ssm, orig_conv = snapshot_states(cache)

            # Forward normally (vanilla auto-regressive decoding)
            print("Vanilla decoding...")
            vanilla_out = model(
                input_ids=input_ids,
                cache_params=cache,
                use_cache=True,
                return_dict=True,
                cache_position=torch.tensor([last_idx], device=device),
            )
            vanilla_out = vanilla_out.logits[:, -1, :].argmax(-1)
            print("Vanilla output:", tok.decode(vanilla_out[0].cpu().numpy()))

            # restore
            for l in range(len(cache.ssm_states)):
                cache.ssm_states[l].copy_( orig_ssm[l] )
                cache.conv_states[l].copy_( orig_conv[l] )
    

            # Forward with cache and parallel scan (cache scan kernel)
            print("Cache scan decoding...")
            cache_out = model(
                input_ids=input_ids,
                cache_params=cache,
                use_cache=True,
                cache_fwd=True,
                return_dict=True,
                cache_position=torch.tensor([last_idx], device=device),
            )
            cache_out = cache_out.logits[:, -1, :].argmax(-1)
            print("Cache scan output:", tok.decode(cache_out[0].cpu().numpy()))

if __name__ == "__main__":
    main()

    """
    Example usage:

    python -m script.cache_decoding_debug \
        --model ./mamba2-2.7b_converted_weights 
    """