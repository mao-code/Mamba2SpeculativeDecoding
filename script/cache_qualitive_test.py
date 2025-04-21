import argparse, time, torch
from contextlib import nullcontext
from transformers import AutoTokenizer
from Mamba2.modeling_mamba2 import Mamba2ForCausalLM
from decoding import mamba_spec_decode_seq, mamba_vanilla_decode
from transformers import AutoConfig

import numpy as np
import random
from utils import set_random_seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path",  required=True, help="Path or HF-hub id of target model")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    
    # Set random seed
    set_random_seed(42)

    print("Loading models...")
    config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=True,    
    )
    model = Mamba2ForCausalLM.from_pretrained(
        args.model_path, 
        config=config, 
        torch_dtype=torch.float16, 
        trust_remote_code=True
    ).to(device).eval()
    model = model.to(torch.float32)

    seed_prompt = "I believe the meaning of life is"
    tokenized = AutoTokenizer.from_pretrained(args.model_path)
    encoding = tokenized(
        seed_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = encoding.input_ids.to(device)
    max_new_tokens = args.max_new_tokens

    # Forward normally (vanilla auto-regressive decoding)
    print("Vanilla decoding...")
    vanilla_output = ""
    vanilla_output_ids = torch.empty(max_new_tokens, dtype=torch.long, device=device)
    for i in range(max_new_tokens):    
        vanilla_out = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )

        logits = vanilla_out.logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        vanilla_output_ids[i] = next_token.item()

        input_ids = torch.cat([input_ids, next_token], dim=1)
        vanilla_output += tokenized.decode(next_token[0])

    print("Vanilla output:")
    print("Input: ", seed_prompt)
    print("Output ids:", vanilla_output_ids)
    print("Output:", vanilla_output)

    print("=" * 20)

    # Forward with cache and parallel scan (cache scan kernel)
    print("Cache scan decoding...")
    new_input_len = 16
    new_input_ids = vanilla_output_ids[: len(seed_prompt) + new_input_len]
    new_input_emb = model.get_input_embeddings()(new_input_ids.unsqueeze(0))  # (1, L_new, d_model)
    L_prev = input_ids.shape[1]
    cache_position = torch.tensor([L_prev], dtype=torch.long, device=input_ids.device)
    cache = model(input_ids=input_ids, use_cache=True, return_dict=True).cache_params

    cache_out = model(
        inputs_embeds=new_input_emb,
        cache_params=cache,
        use_cache=True,
        cache_fwd=True,
        return_dict=True,
        return_states=True,          
        return_final=True,
        cache_position=cache_position
    )
    logits = cache_out.logits 
    probs = logits.softmax(-1) 
    cache_out_token = logits.argmax(-1, keepdim=True).view(-1)  # flatten all dimensions
    print("Cache out token ids:", cache_out_token)
    cache_output = tokenized.decode(cache_out_token)

    print("Cache scan output:")
    print("Seed Input: ", seed_prompt)
    print("New Input Ids: ", new_input_ids)
    print(f"Output:", cache_output)

    print("=" * 20)

    print("Compare the outputs:")

    print("Vanilla original output split:", vanilla_output.split())

    # SHIF RIGHT FOR 1 TOKEN, IT IS THE CULPRIT
    shift_factor = 1
    
    vanilla_output_cut = vanilla_output[: len(seed_prompt) + new_input_len+1] 
    print("Vanilla output cut:", vanilla_output_cut.split())
    print("Cache output split:", cache_output.split())

    paired = zip(vanilla_output_cut.split()[shift_factor:], cache_output.split())
    for i, (v, c) in enumerate(paired):
        if v != c:
            print(f"Mismatch at token {i}: vanilla: {v}, cache scan: {c}")
        else:
            print(f"Match at token {i}: vanilla: {v}, cache scan: {c}")

    print("=" * 20)
    print("Compare the logits only: ")
    vanilla_logits_prob_cut = vanilla_output_ids[shift_factor: len(seed_prompt) + new_input_len+1]
    for i, (v, c) in enumerate(zip(vanilla_logits_prob_cut, cache_out_token)):
        if not torch.allclose(v, c):
            print(f"Mismatch at token {i}: vanilla: {v}, cache scan: {c}")
        else:
            print(f"Match at token {i}: vanilla: {v}, cache scan: {c}")

if __name__ == "__main__":
    main()

    """
    python -m script.cache_qualitive_test \
        --model_path ./mamba2-2.7b_converted_weights \
        --max_new_tokens 128
    """