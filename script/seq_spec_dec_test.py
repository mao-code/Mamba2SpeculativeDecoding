import argparse, time, torch
from contextlib import nullcontext
from transformers import AutoTokenizer
from Mamba2.modeling_mamba2 import Mamba2ForCausalLM
from decoding import mamba_spec_decode_seq, mamba_vanilla_decode
from transformers import AutoConfig
from utils import set_random_seed
from verification import VerificationStrategy, RatioSamplingStrategy, ExactMatchStrategy

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
    ap.add_argument("--verification", type=str, default="ratio", choices=["ratio", "exact"])
    ap.add_argument("--log", action="store_true", help="Log the verification process")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Set random seed
    set_random_seed(42)

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
    prompt_ids      = encoding.input_ids.to(device)

    # --- vanilla ----------------------------------------------------------
    out_vanilla, t_vanilla = timed(
        mamba_vanilla_decode, target, prompt_ids, max_new=args.new_tokens
    )
    out_vanilla_text = tok_tgt.decode(out_vanilla.view(-1))
    print("Vanilla output:", out_vanilla_text)

    # --- speculative ------------------------------------------------------
    if args.verification == "ratio":
        verification_strategy = RatioSamplingStrategy()
    elif args.verification == "exact":
        verification_strategy = ExactMatchStrategy()
    else:
        raise ValueError(f"Unknown verification strategy: {args.verification}")

    out_spec, t_spec = timed(
        mamba_spec_decode_seq, target, draft, prompt_ids,
        K=args.K, max_new=args.new_tokens, verification_strategy=verification_strategy,
        log=args.log
    )
    out_spec_text = tok_tgt.decode(out_spec.view(-1))
    print("Speculative output:", out_spec_text)

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
    --K 3 --new-tokens 64 --device cuda:0 \
    --verification exact \
    --log
    """