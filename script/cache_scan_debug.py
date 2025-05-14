"""
Vanilla Decoding v.s. Cache Scan Decoding
Measure and compare the latency of two decoding methods.
"""

import argparse
from utils import set_random_seed
from decoding import snapshot_states
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

    # model = torch.compile(model, backend="inductor", mode="max-autotune")

    # --- Global Warm-Up Phase ---
    print("Performing global warm-up...")
    dummy_prompt = "This is a dummy prompt for warm-up."
    dummy_encoding = tok(dummy_prompt, return_tensors="pt", padding=True, truncation=True)
    dummy_input_ids = dummy_encoding.input_ids.to(device)
    dummy_input_len = dummy_input_ids.size(1)
    K = 3
    dummy_pre_load_length = dummy_input_len - K

    # Warm-up forward pass to set up cache
    _ = model(
        input_ids=dummy_input_ids[:, :dummy_pre_load_length],
        use_cache=True,
        return_dict=True,
        cache_position=torch.tensor([0], device=device),
    )
    cache = _.cache_params

    # Warm-up Vanilla Decoding
    cache_pos = dummy_pre_load_length
    for i in range(K):
        token = dummy_input_ids[:, cache_pos : cache_pos + 1].to(device)
        out = model(
            input_ids=token,
            cache_params=cache,
            use_cache=True,
            return_dict=True,
            cache_position=torch.tensor([cache_pos], device=device),
        )
        cache = out.cache_params
        cache_pos += 1

    # Warm-up Cache Scan Decoding
    out = model(
        input_ids=dummy_input_ids[:, dummy_pre_load_length:],
        cache_params=cache,
        use_cache=True,
        cache_fwd=True,
        return_dict=True,
        cache_position=torch.tensor([dummy_pre_load_length], device=device),
        chunk_size=1
    )

    # Warm-up Cache Scan Chunk Decoding
    out = model(
        input_ids=dummy_input_ids[:, dummy_pre_load_length:],
        cache_params=cache,
        use_cache=True,
        cache_fwd=True,
        return_dict=True,
        cache_position=torch.tensor([dummy_pre_load_length], device=device),
    )
    print("Warm-up completed.")

    prompts = [
        "In a distant future where artificial intelligence governs most of human life, a small group of rebels begins to question the decisions made by machines. They live in the underbelly of a massive city, hidden from the eyes of drones and surveillance bots. Describe how this underground society functions, what kind of technologies they rely on to survive, and how their daily life differs from those who live under the AI regime. Include details about their philosophy, any rituals they follow, and how ",

        "You are tasked with writing a comprehensive guide for someone who has never cooked before but wants to make a delicious and healthy dinner using only ingredients commonly found in a typical kitchen. The guide should be broken into steps and include basic safety tips, suggestions for substitutions in case some ingredients are missing, and an encouraging tone to help the user feel confident. Be detailed in your instructions, including how to cut vegetables properly, how long to cook things, and how ",

        "Imagine a world where the sun never sets, and people must adapt to a life of perpetual daylight. Describe how this impacts their culture, sleeping habits, architecture, and society as a whole. How do they tell time? What kind of festivals or rituals have developed in response to this never-ending day? Include some fictional examples of how certain professions have evolved in this environment and what challenges people face when trying to find personal time or solitude. Also discuss how ",

        "You are designing a new city from scratch on a distant planet with a breathable atmosphere and gravity similar to Earth. The planet has different day/night cycles and a range of unfamiliar biomes. Write a detailed description of how you would plan the infrastructure, transportation, energy resources, and housing. Also explain how you would organize governance, education, and healthcare systems. Be sure to consider sustainability, adaptability to the alien environment, and how ",

        "Pretend you are writing a letter from the year 2124 to someone living in the year 2024. In your letter, describe the most significant technological advancements that have transformed daily life, such as transportation, communication, healthcare, and artificial intelligence. Share how people spend their time, what types of jobs exist, and what values society now holds most dear. Also mention how climate change was addressed and how political structures have evolved. Try to paint a vivid picture of what ",

        "Write a detailed summary of a fictional research paper titled 'Temporal Neural Dynamics in Synthetic Consciousness: A Study on Emergent Behavior in Large-Scale AI Agents.' The paper should include an abstract, background, methodology, experimental setup, results, and discussion. The focus of the research is on how synthetic agents display behaviors that resemble conscious thought when trained with large-scale, multimodal data and recursive feedback loops. Include insights about the limitations of the current study and potential future directions. This summary should be detailed enough that ",

        "You are interviewing an AI entity that has just passed a newly developed version of the Turing Test, which also includes emotional intelligence and situational reasoning. Draft a transcript of the interview where you ask the AI about its understanding of art, morality, free will, and its sense of self. Include the AI's responses, which should be insightful and nuanced, reflecting a high level of reasoning. Explore topics such as creativity, empathy, and decision-making. This transcript should reveal whether the ",

        "Write an editorial article discussing the societal impacts of humans gaining the ability to edit their memories. Include both sides of the argument: the benefits for trauma survivors, the ethical implications for historical truth, and the potential misuse in legal or political settings. Provide fictional case studies or scenarios where memory editing has caused both positive and negative outcomes. End the article with a personal reflection from the editor about whether they would choose to alter ",

        "Craft a fictional dialogue between a philosopher and a neuroscientist debating the nature of consciousness. The philosopher argues from a dualist perspective, suggesting that subjective experience cannot be reduced to physical processes alone. The neuroscientist counters with data from brain scans, lesion studies, and neural network simulations. The discussion should include references to famous theories like Integrated Information Theory, Global Workspace Theory, and philosophical thought experiments such as Mary's Room and the Chinese Room. Make sure both sides present compelling arguments, and ",

        "Create a narrative where an alien species tries to understand human behavior by analyzing fragments of our internet history. They encounter social media posts, news articles, memes, and videos from different eras. Describe their interpretations, misunderstandings, and the conclusions they draw about human priorities, emotions, and communication styles. Include scenes from their discussions and cultural context to show how their own beliefs shape the way they see us. This narrative should be humorous but also "
    ]


    vanilla_times = []
    cache_times = []
    cache_chunk_times = []

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            print(f"\nPrompt: {prompt}\n" + "-"*40)
            encoding = tok(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = encoding.input_ids.to(device)

            # Warm up and capture cache
            input_len = input_ids.size(1)
            K = 3
            pre_load_length = input_len - K
            pre_out = model(
                input_ids=input_ids[:, :pre_load_length],
                use_cache=True,
                return_dict=True,
                cache_position=torch.tensor([0], device=device),
            )
            cache = pre_out.cache_params
            orig_ssm, orig_conv = snapshot_states(cache)
            last_idx = pre_load_length
            new_input = input_ids[:, last_idx:].to(device)

            # --- Vanilla decoding timing ---
            def vanilla_step():
                cache_pos = pre_load_length
                cache = pre_out.cache_params
                for i in range(input_ids.size(1) - pre_load_length):
                    token = input_ids[:, cache_pos : cache_pos + 1].to(device)
                    out = model(
                        input_ids=token,
                        cache_params=cache,
                        use_cache=True,
                        return_dict=True,
                        cache_position=torch.tensor([cache_pos], device=device),
                    )

                    last_tok = out.logits[:, -1, :].argmax(-1)

                    cache = out.cache_params
                    cache_pos += 1

                return last_tok

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
                    input_ids=new_input,
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
                    input_ids=new_input,
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

    """
    Example usage:
    python -m script.cache_scan_debug --model ./mamba2-2.7b_converted_weights
    """
