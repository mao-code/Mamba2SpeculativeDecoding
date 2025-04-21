import argparse, time, torch
from contextlib import nullcontext
from transformers import AutoTokenizer
from Mamba2.modeling_mamba2 import Mamba2ForCausalLM
from decoding import mamba_spec_decode_seq, mamba_vanilla_decode
from transformers import AutoConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path",  required=True, help="Path or HF-hub id of target model")
    ap.add_argument("max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    device = torch.device(args.device)

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
    target = target.to(torch.float32)

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
    output = ""
    for i in range(max_new_tokens):    
        vanilla_out = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )

        logits = vanilla_out.logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        output += tokenized.decode(next_token[0])

    print("Vanilla output:")
    print("Input: ", seed_prompt)
    print("Output:", output)

    # Forward with cache and parallel scan (cache scan kernel)


if __name__ == "__main__":
    main()

    """
    python -m script.cache_qualitive_test \
        --model_path ./mamba2-2.7b_converted_weights \ 
        --max_new_tokens 128
    """