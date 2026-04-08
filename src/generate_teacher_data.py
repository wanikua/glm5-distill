"""
Generate training data by calling GLM-5.1 via Zhipu BigModel API.

Usage:
    python -m src.generate_teacher_data \
        --seed_file data/seeds/seed_prompts.jsonl \
        --output_file data/teacher_outputs/train.jsonl \
        --model glm-5.1 \
        --max_workers 4 \
        --num_samples 5000
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from zhipuai import ZhipuAI

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful, accurate, and thoughtful assistant. "
    "Provide clear, well-structured responses. "
    "For code questions, include working examples with explanations."
)


def call_glm(client: ZhipuAI, model: str, prompt: str, temperature: float = 0.7) -> str | None:
    """Call GLM-5.1 API and return the response text."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            print(f"Failed after 3 attempts for prompt: {prompt[:80]}... Error: {e}")
            return None


def load_seed_prompts(seed_file: str) -> list[dict]:
    """Load seed prompts from JSONL file. Each line: {"prompt": "...", "category": "..."}"""
    prompts = []
    with open(seed_file) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def generate_data(
    seed_file: str,
    output_file: str,
    model: str = "glm-5.1",
    max_workers: int = 4,
    num_samples: int | None = None,
    temperature: float = 0.7,
):
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        raise ValueError("ZHIPU_API_KEY not found in environment. Set it in .env file.")

    client = ZhipuAI(api_key=api_key)
    prompts = load_seed_prompts(seed_file)

    if num_samples and num_samples < len(prompts):
        prompts = prompts[:num_samples]

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing progress
    existing = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                data = json.loads(line)
                existing.add(data.get("prompt", ""))
        print(f"Resuming: {len(existing)} samples already generated")

    remaining = [p for p in prompts if p["prompt"] not in existing]
    print(f"Generating {len(remaining)} samples with {max_workers} workers...")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(call_glm, client, model, p["prompt"], temperature): p
            for p in remaining
        }
        with tqdm(total=len(remaining)) as pbar:
            for future in as_completed(futures):
                seed = futures[future]
                response = future.result()
                if response:
                    results.append({
                        "prompt": seed["prompt"],
                        "response": response,
                        "category": seed.get("category", "general"),
                        "model": model,
                    })
                pbar.update(1)

                # Flush every 100 samples
                if len(results) >= 100:
                    with open(output_path, "a") as f:
                        for r in results:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    results.clear()

    # Flush remaining
    if results:
        with open(output_path, "a") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(existing) + len(remaining)
    print(f"Done. Total samples: {total} -> {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate teacher data from GLM-5.1")
    parser.add_argument("--seed_file", type=str, default="data/seeds/seed_prompts.jsonl")
    parser.add_argument("--output_file", type=str, default="data/teacher_outputs/train.jsonl")
    parser.add_argument("--model", type=str, default="glm-5.1")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    generate_data(
        seed_file=args.seed_file,
        output_file=args.output_file,
        model=args.model,
        max_workers=args.max_workers,
        num_samples=args.num_samples,
        temperature=args.temperature,
    )
