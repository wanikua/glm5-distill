"""
Tacit Knowledge Distillation data generator.

Polanyi's insight: "We know more than we can tell."
Standard distillation only captures explicit I/O mappings.
This extracts the teacher's implicit reasoning via:
  1. CoT extraction — force tacit reasoning into explicit form
  2. Multi-temp sampling — capture uncertainty/confidence patterns
  3. Perspective probing — same question, different angles

Usage:
    python -m src.generate_teacher_data \
        --seed_file data/seeds/seed_prompts.jsonl \
        --output_dir data/teacher_outputs \
        --model glm-5.1 \
        --max_workers 8
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

# --- System prompts for different extraction modes ---

DIRECT_SYSTEM = (
    "You are a helpful, accurate assistant. "
    "Provide clear, well-structured responses."
)

COT_SYSTEM = (
    "You are a helpful assistant that thinks step by step. "
    "Before giving your answer, walk through your reasoning process in detail. "
    "Show your thought process, including what you considered, "
    "what tradeoffs you weighed, and why you chose your approach. "
    "Format: first your reasoning under '## Thinking', then your answer under '## Answer'."
)

PERSPECTIVE_TEMPLATES = [
    "You are an expert who explains things by first identifying common misconceptions. "
    "Start by addressing what most people get wrong about this topic, then give the correct explanation.",
    "You are a Socratic teacher. Instead of directly answering, "
    "guide the learner by asking and answering a series of progressive questions that build understanding.",
    "You are a practical engineer. Skip theory — focus on concrete examples, "
    "real-world usage, edge cases, and gotchas that matter in production.",
]

CONFIDENCE_SYSTEM = (
    "You are a calibrated assistant. After answering, add a section '## Confidence' where you: "
    "1) Rate your confidence (high/medium/low) 2) State what you're uncertain about "
    "3) Note what could make your answer wrong."
)


def call_api(client: ZhipuAI, model: str, system: str, prompt: str, temperature: float = 0.7) -> str | None:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=2048,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"FAIL: {prompt[:60]}... | {e}")
                return None


def generate_for_seed(client: ZhipuAI, model: str, seed: dict) -> list[dict]:
    """Generate all knowledge extraction variants for one seed prompt."""
    prompt = seed["prompt"]
    category = seed.get("category", "general")
    results = []

    # Phase 1: CoT — make tacit reasoning explicit
    cot = call_api(client, model, COT_SYSTEM, prompt, temperature=0.7)
    if cot:
        results.append({
            "prompt": prompt, "response": cot,
            "category": category, "phase": "cot", "mode": "explicit",
        })

    # Phase 2: Direct — the "polished" output (tacit knowledge internalized)
    direct = call_api(client, model, DIRECT_SYSTEM, prompt, temperature=0.7)
    if direct:
        results.append({
            "prompt": prompt, "response": direct,
            "category": category, "phase": "direct", "mode": "tacit",
        })

    # Phase 2b: Multi-temp — uncertainty awareness
    hot = call_api(client, model, DIRECT_SYSTEM, prompt, temperature=1.0)
    if hot:
        results.append({
            "prompt": prompt, "response": hot,
            "category": category, "phase": "direct_hot", "mode": "tacit",
        })

    # Phase 3: Confidence calibration — knowing what you don't know
    calibrated = call_api(client, model, CONFIDENCE_SYSTEM, prompt, temperature=0.7)
    if calibrated:
        results.append({
            "prompt": prompt, "response": calibrated,
            "category": category, "phase": "calibration", "mode": "meta",
        })

    # Phase 3: Perspective probing — subsidiary awareness
    # Use one random perspective to keep API costs reasonable
    import random
    perspective_sys = random.choice(PERSPECTIVE_TEMPLATES)
    perspective = call_api(client, model, perspective_sys, prompt, temperature=0.7)
    if perspective:
        results.append({
            "prompt": prompt, "response": perspective,
            "category": category, "phase": "perspective", "mode": "subsidiary",
        })

    return results


def load_seed_prompts(seed_file: str) -> list[dict]:
    prompts = []
    with open(seed_file) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def generate_data(
    seed_file: str,
    output_dir: str,
    model: str = "glm-5.1",
    max_workers: int = 8,
    num_samples: int | None = None,
):
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        raise ValueError("ZHIPU_API_KEY not set")

    client = ZhipuAI(api_key=api_key)
    seeds = load_seed_prompts(seed_file)
    if num_samples:
        seeds = seeds[:num_samples]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Output files by training phase
    files = {
        "phase1_explicit": out_path / "phase1_cot.jsonl",
        "phase2_tacit": out_path / "phase2_direct.jsonl",
        "phase3_meta": out_path / "phase3_meta.jsonl",
    }

    # Resume
    existing = set()
    for f in files.values():
        if f.exists():
            for line in open(f):
                d = json.loads(line)
                existing.add(f"{d['prompt']}|{d['phase']}")

    remaining = [s for s in seeds if s["prompt"] not in {k.split("|")[0] for k in existing}]
    print(f"Seeds: {len(seeds)}, Remaining: {len(remaining)}, ~{len(remaining)*5} API calls")

    def process_seed(seed):
        return generate_for_seed(client, model, seed)

    all_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_seed, s): s for s in remaining}
        for future in tqdm(as_completed(futures), total=len(remaining)):
            results = future.result()
            all_results.extend(results)

            # Flush periodically
            if len(all_results) >= 50:
                _flush(all_results, files)
                all_results.clear()

    if all_results:
        _flush(all_results, files)

    for name, f in files.items():
        count = sum(1 for _ in open(f)) if f.exists() else 0
        print(f"  {name}: {count} samples")


def _flush(results: list[dict], files: dict[str, Path]):
    for r in results:
        if r["mode"] == "explicit":
            target = files["phase1_explicit"]
        elif r["mode"] in ("tacit", "subsidiary"):
            target = files["phase2_tacit"]
        else:
            target = files["phase3_meta"]
        with open(target, "a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_file", default="data/seeds/seed_prompts.jsonl")
    parser.add_argument("--output_dir", default="data/teacher_outputs")
    parser.add_argument("--model", default="glm-5.1")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()
    generate_data(**vars(args))
