"""
Run inference with the distilled model.

Usage:
    # Interactive chat
    python -m src.inference --model_path models/qwen3-1.7b-glm5-distill

    # Single prompt
    python -m src.inference --model_path models/qwen3-1.7b-glm5-distill \
        --prompt "Explain how Python generators work."

    # Benchmark speed
    python -m src.inference --model_path models/qwen3-1.7b-glm5-distill --benchmark
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

try:
    from src.persona import PERSONA_SYSTEM
except ImportError:
    PERSONA_SYSTEM = "You are a helpful, accurate assistant."


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer, device


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 1024, stream: bool = True):
    messages = [{"role": "system", "content": PERSONA_SYSTEM}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextStreamer(tokenizer, skip_special_tokens=True) if stream else None

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            streamer=streamer,
        )
    elapsed = time.perf_counter() - start

    generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    tokens_per_sec = generated_tokens / elapsed

    if not stream:
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(response)

    print(f"\n--- {generated_tokens} tokens in {elapsed:.1f}s ({tokens_per_sec:.1f} tok/s) ---")
    return generated_tokens, elapsed


def benchmark(model, tokenizer):
    prompts = [
        "先生，内卷到底有没有尽头？",
        "鲁迅先生怎么看现在的短视频？",
        "年轻人该不该躺平？",
    ]
    total_tokens = 0
    total_time = 0

    for i, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Benchmark {i+1}/{len(prompts)}: {prompt[:60]}...")
        print(f"{'='*60}")
        tokens, elapsed = generate(model, tokenizer, prompt, stream=False)
        total_tokens += tokens
        total_time += elapsed

    print(f"\n{'='*60}")
    print(f"Overall: {total_tokens} tokens in {total_time:.1f}s ({total_tokens/total_time:.1f} tok/s)")


def interactive(model, tokenizer):
    print("Interactive mode. Type 'quit' to exit.\n")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            break
        print("Assistant: ", end="", flush=True)
        generate(model, tokenizer, prompt, stream=True)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with distilled model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model_path)
    print(f"Model loaded on {device}")

    if args.benchmark:
        benchmark(model, tokenizer)
    elif args.prompt:
        generate(model, tokenizer, args.prompt)
    else:
        interactive(model, tokenizer)
