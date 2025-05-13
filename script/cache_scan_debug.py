"""
Vanilla Decoding v.s. Cache Scan Decoding
Measure and compare the latency of two decoding methods.
"""

import argparse
from utils import set_random_seed, snapshot_states
import torch
from transformers import AutoTokenizer, AutoConfig
from Mamba2.modeling_mamba2 import Mamba2ForCausalLM

def time_call(fn, *args, **kwargs):
    # Helper to time a single CUDA call
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    out = fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    return out, ms  # returns (output, elapsed_ms)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  required=True, help="Path or HF-hub id of target model")
    args = ap.parse_args()

    device = torch.device("cuda:0")

    set_random_seed(42)

    print("Loading model...")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = Mamba2ForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device).eval().to(torch.float32)
    tok = AutoTokenizer.from_pretrained(args.model)

    prompts = [
        "I believe the meaning of life is",
        "The capital of France is",
        "The sun rises in the",
        "The quick brown fox jumps over the lazy dog",
        "In a galaxy far, far away",
        "Once upon a time in a land far away",
        "The future of AI is",
        "The theory of relativity was proposed by",
        "The mitochondria is the powerhouse of the",
        "To be or not to be, that is the",
    ]

    vanilla_times = []
    cache_times = []
    cache_chunk_times = []

    with torch.no_grad():
        for prompt in prompts:
            print(f"\nPrompt: {prompt}\n" + "-"*40)
            encoding = tok(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = encoding.input_ids.to(device)

            # Warm up and capture cache
            pre_out = model(
                input_ids=input_ids[:, :-1],
                use_cache=True,
                return_dict=True,
                cache_position=torch.tensor([0], device=device),
            )
            cache = pre_out.cache_params
            orig_ssm, orig_conv = snapshot_states(cache)
            last_idx = input_ids.size(1) - 1
            last_token = input_ids[:, -1:].to(device)

            # --- Vanilla decoding timing ---
            def vanilla_step():
                out = model(
                    input_ids=last_token,
                    cache_params=cache,
                    use_cache=True,
                    return_dict=True,
                    cache_position=torch.tensor([last_idx], device=device),
                )
                return out.logits[:, -1, :].argmax(-1)

            vanilla_tok, vanilla_ms = time_call(vanilla_step)
            vanilla_times.append(vanilla_ms)
            print(f"Vanilla token: {tok.decode(vanilla_tok[0].cpu().item())} | time: {vanilla_ms:.2f} ms")

            # restore cache to original snapshot
            for l in range(len(cache.ssm_states)):
                cache.ssm_states[l].copy_(orig_ssm[l])
                cache.conv_states[l].copy_(orig_conv[l])

            # --- Cache‐scan decoding timing ---
            def cache_scan_step():
                out = model(
                    input_ids=last_token,
                    cache_params=cache,
                    use_cache=True,
                    cache_fwd=True,
                    return_dict=True,
                    cache_position=torch.tensor([last_idx], device=device),
                    chunk_size=1
                )
                return out.logits[:, -1, :].argmax(-1)

            cache_tok, cache_ms = time_call(cache_scan_step)
            cache_times.append(cache_ms)
            print(f"Cache scan token: {tok.decode(cache_tok[0].cpu().item())} | time: {cache_ms:.2f} ms")

            # --- Cache‐scan decoding timing (Chunk) ---
            def cache_scan_step():
                out = model(
                    input_ids=last_token,
                    cache_params=cache,
                    use_cache=True,
                    cache_fwd=True,
                    return_dict=True,
                    cache_position=torch.tensor([last_idx], device=device),
                )
                return out.logits[:, -1, :].argmax(-1)

            cache_chunk_tok, cache_ms = time_call(cache_scan_step)
            cache_chunk_times.append(cache_ms)
            print(f"Cache chunk scan token: {tok.decode(cache_chunk_tok[0].cpu().item())} | time: {cache_ms:.2f} ms")

    # Summary
    avg_vanilla = sum(vanilla_times) / len(vanilla_times)
    avg_cache   = sum(cache_times)   / len(cache_times)
    avg_cache_chunk = sum(cache_chunk_times) / len(cache_chunk_times)
    print("\n" + "="*40)
    print(f"Average Vanilla decoding time: {avg_vanilla:.2f} ms")
    print(f"Average Cache-scan decoding time: {avg_cache:.2f} ms")
    print(f"Average Cache-scan chunk decoding time: {avg_cache_chunk:.2f} ms")

if __name__ == "__main__":
    main()
