import argparse, time, torch
from contextlib import nullcontext
from transformers import AutoTokenizer
from Mamba2.modeling_mamba2 import Mamba2ForCausalLM
from decoding import mamba_spec_decode_seq
from transformers import AutoConfig

def vanilla_generate(target, prompt_ids, attention_mask, max_new=256):
    return target.generate(
        prompt_ids,  
        attention_mask=attention_mask,
        max_new_tokens=max_new,
        pad_token_id=target.config.eos_token_id,
    )[0][len(prompt_ids[0]):]

def timed(fn, *args, **kw):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    out   = fn(*args, **kw)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    dura  = time.perf_counter() - start
    return out, dura

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target",  required=True, help="Path or HF‑hub id of target model")
    ap.add_argument("--draft",   required=True, help="Path or HF‑hub id of smaller draft model")
    ap.add_argument("--device",  default="cuda:0")
    ap.add_argument("--prompt",  default="I believe the meaning of life is")
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--K",           type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    device = torch.device(args.device)

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
    prompt_ids      = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)

    # --- vanilla ----------------------------------------------------------
    _, t_vanilla = timed(
        vanilla_generate, target, prompt_ids, attention_mask,
        max_new=args.new_tokens, temperature=args.temperature
    )

    # --- speculative ------------------------------------------------------
    _, t_spec = timed(
        mamba_spec_decode_seq, target, draft, prompt_ids,
        K=args.K, max_new=args.new_tokens, temperature=args.temperature
    )

    # ----------------------------------------------------------------------
    tok_per_sec_van = args.new_tokens / t_vanilla
    tok_per_sec_spc = args.new_tokens / t_spec
    speedup         = t_vanilla / t_spec

    print("\n=== Results =====================================================")
    print(f"Prompt:               \"{args.prompt}\"")
    print(f"Target model:         {args.target}")
    print(f"Draft model:          {args.draft}")
    print(f"Block size (K):       {args.K}")
    print(f"Tokens generated:     {args.new_tokens}")
    print("---------------------------------------------------------------")
    print(f"Vanilla   : {t_vanilla:7.3f} s  |  {tok_per_sec_van:6.2f} tok/s")
    print(f"Spec-dec  : {t_spec:7.3f} s  |  {tok_per_sec_spc:6.2f} tok/s")
    print("---------------------------------------------------------------")
    print(f"Speed-up  : x{speedup:5.2f}")
    print("================================================================")

if __name__ == "__main__":
    main()

    """
    python -m script.seq_spec_dec_test \
    --target ./mamba2-2.7b_converted_weights \
    --draft  ./mamba2-130m_converted_weights \
    --prompt "I believe the meaning of life is" \
    --K 8 --new-tokens 256 --device cuda:0
    """