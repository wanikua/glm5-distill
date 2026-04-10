"""
Adversarial Distillation with Self-Play (ADSP).

Inspired by GAN dynamics + Polanyi's tacit knowledge:
  Generator  = Student model
  Discriminator = Teacher (GLM-5.1) as multi-dimensional judge
  Self-Play  = Student_N vs Student_{N-1} (SPIN-style)

Each round:
  1. Student generates K responses per prompt (rejection sampling)
  2. Teacher scores on 4 dimensions (accuracy, reasoning, clarity, utility)
  3. Build DPO pairs: best_of_K vs worst_of_K + teacher_preferred vs student
  4. Hard example mining: oversample prompts with largest gaps
  5. Teacher generates new harder prompts targeting student weaknesses
  6. DPO training with margin-weighted loss
  7. ELO tracking for convergence

Usage:
    python -m src.adversarial \
        --student_path outputs/qwen3-8b-distill/final \
        --rounds 5 \
        --prompts_file data/seeds/luxun_seeds.jsonl
"""

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from zhipuai import ZhipuAI

try:
    from src.persona import PERSONA_SYSTEM
except ImportError:
    PERSONA_SYSTEM = "You are a helpful, accurate assistant."

load_dotenv()

# ============================================================
# Judge prompts
# ============================================================

JUDGE_SYSTEM = """You are an expert evaluator for a persona AI model.
The model is supposed to embody a specific character. Score two responses.

For EACH response, provide scores (0-10) on five dimensions:
- persona: does it sound like the character? Voice, tone, habits, worldview.
- depth: insight quality, not superficial platitudes
- consistency: does it stay in character throughout? Any "AI assistant" slips?
- sharpness: is the language vivid, concise, memorable?
- utility: does it actually help the person asking?

A score of 0 on "consistency" means the response broke character (said "as an AI", listed bullet points like a chatbot, used emoji, etc).

Output ONLY valid JSON:
{
  "a": {"persona": <int>, "depth": <int>, "consistency": <int>, "sharpness": <int>, "utility": <int>},
  "b": {"persona": <int>, "depth": <int>, "consistency": <int>, "sharpness": <int>, "utility": <int>},
  "winner": "a" or "b" or "tie",
  "reason": "<one sentence>"
}"""

JUDGE_TEMPLATE = """Question:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Evaluate. Output JSON only."""

WEAKNESS_SYSTEM = """You are a curriculum designer for AI training.
Given a student model's weakest dimensions and example failures,
generate 5 NEW challenging prompts that specifically target these weaknesses.

Output ONLY a JSON array of objects: [{"prompt": "...", "category": "...", "targets": "..."}]"""

WEAKNESS_TEMPLATE = """The student model is weakest in: {weak_dims}

Example prompts where it scored poorly:
{examples}

Generate 5 harder prompts targeting these exact weaknesses. JSON array only."""


# ============================================================
# Core functions
# ============================================================

class ELO:
    """Track student vs teacher ratings across rounds."""

    def __init__(self, k=32):
        self.ratings = {"student": 1200.0, "teacher": 1500.0}
        self.history = []
        self.k = k

    def update(self, winner: str, loser: str):
        ea = 1 / (1 + 10 ** ((self.ratings[loser] - self.ratings[winner]) / 400))
        self.ratings[winner] += self.k * (1 - ea)
        self.ratings[loser] -= self.k * (1 - ea)

    def update_tie(self, a: str, b: str):
        ea = 1 / (1 + 10 ** ((self.ratings[b] - self.ratings[a]) / 400))
        self.ratings[a] += self.k * (0.5 - ea)
        self.ratings[b] -= self.k * (0.5 - ea)

    def record(self, round_num: int):
        self.history.append({"round": round_num, **self.ratings.copy()})

    def gap(self) -> float:
        return self.ratings["teacher"] - self.ratings["student"]


def api_call(client: ZhipuAI, model: str, system: str, prompt: str,
             temperature: float = 0.7, max_tokens: int = 2048) -> str | None:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return None


def student_generate(model, tokenizer, prompt: str, n: int = 1,
                     max_new_tokens: int = 1024) -> list[str]:
    """Generate n responses via rejection sampling."""
    messages = [{"role": "system", "content": PERSONA_SYSTEM}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    responses = []
    temps = [0.6, 0.7, 0.8, 0.9, 1.0][:n]
    for temp in temps:
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=temp, top_p=0.9,
            )
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        responses.append(resp)
    return responses


def multi_judge(client: ZhipuAI, model: str, prompt: str,
                resp_a: str, resp_b: str) -> dict | None:
    """Multi-dimensional scoring. Randomize positions to avoid bias."""
    swap = random.random() < 0.5
    if swap:
        resp_a, resp_b = resp_b, resp_a

    text = JUDGE_TEMPLATE.format(prompt=prompt, response_a=resp_a, response_b=resp_b)
    raw = api_call(client, model, JUDGE_SYSTEM, text, temperature=0.1, max_tokens=512)
    if not raw:
        return None

    try:
        if "{" in raw:
            raw = raw[raw.index("{"):raw.rindex("}") + 1]
        result = json.loads(raw)

        # Unswap
        if swap:
            result["a"], result["b"] = result["b"], result["a"]
            if result.get("winner") == "a":
                result["winner"] = "b"
            elif result.get("winner") == "b":
                result["winner"] = "a"

        return result
    except (json.JSONDecodeError, KeyError):
        return None


def compute_score(dims: dict) -> float:
    """Weighted aggregate — persona consistency is king for character models."""
    weights = {"persona": 0.30, "depth": 0.20, "consistency": 0.25, "sharpness": 0.15, "utility": 0.10}
    return sum(dims.get(k, 5) * w for k, w in weights.items())


def find_weaknesses(judgments: list[dict]) -> tuple[list[str], list[dict]]:
    """Identify student's weakest dimensions and worst prompts."""
    dim_scores = {"persona": [], "depth": [], "consistency": [], "sharpness": [], "utility": []}
    prompt_gaps = []

    for j in judgments:
        s_dims = j.get("student_dims", {})
        t_dims = j.get("teacher_dims", {})
        for d in dim_scores:
            if d in s_dims:
                dim_scores[d].append(s_dims[d])

        gap = j.get("gap", 0)
        prompt_gaps.append({"prompt": j["prompt"], "gap": gap})

    # Weakest dimensions
    dim_avgs = {d: sum(v) / len(v) if v else 5 for d, v in dim_scores.items()}
    weak_dims = sorted(dim_avgs, key=dim_avgs.get)[:2]

    # Hardest prompts
    prompt_gaps.sort(key=lambda x: x["gap"], reverse=True)
    hard_prompts = prompt_gaps[:5]

    return weak_dims, hard_prompts


def generate_harder_prompts(client: ZhipuAI, model: str,
                            weak_dims: list[str], hard_examples: list[dict]) -> list[dict]:
    """Teacher generates new prompts targeting student weaknesses."""
    examples_str = "\n".join(f"- {e['prompt']} (gap={e['gap']:.1f})" for e in hard_examples)
    prompt = WEAKNESS_TEMPLATE.format(weak_dims=", ".join(weak_dims), examples=examples_str)

    raw = api_call(client, model, WEAKNESS_SYSTEM, prompt, temperature=0.8)
    if not raw:
        return []

    try:
        if "[" in raw:
            raw = raw[raw.index("["):raw.rindex("]") + 1]
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []


# ============================================================
# Main adversarial loop
# ============================================================

def adversarial_round(
    model, tokenizer, client: ZhipuAI, teacher_model: str,
    prompts: list[dict], round_num: int, output_dir: str,
    elo: ELO, k_samples: int = 3, max_workers: int = 4,
) -> tuple[list[dict], list[dict], dict]:
    """Full adversarial round."""

    print(f"\n{'='*60}")
    print(f"  ROUND {round_num} | ELO: student={elo.ratings['student']:.0f} teacher={elo.ratings['teacher']:.0f}")
    print(f"{'='*60}")

    # --- Step 1: Student generates K responses per prompt ---
    print(f"\n[1/5] Student: {len(prompts)} prompts x {k_samples} samples...")
    student_outputs = {}
    for p in tqdm(prompts):
        student_outputs[p["prompt"]] = student_generate(model, tokenizer, p["prompt"], n=k_samples)

    # --- Step 2: Teacher generates reference responses ---
    print(f"\n[2/5] Teacher generating references...")
    teacher_outputs = {}

    def _tgen(prompt):
        return prompt, api_call(client, teacher_model,
                                PERSONA_SYSTEM, prompt)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_tgen, p["prompt"]): p for p in prompts}
        for f in tqdm(as_completed(futs), total=len(prompts)):
            prompt, resp = f.result()
            if resp:
                teacher_outputs[prompt] = resp

    # --- Step 3: Multi-dimensional judging ---
    print(f"\n[3/5] Multi-dimensional judging...")
    judgments = []
    dpo_pairs = []

    def _judge_prompt(p):
        prompt = p["prompt"]
        s_resps = student_outputs.get(prompt, [])
        t_resp = teacher_outputs.get(prompt)
        if not s_resps or not t_resp:
            return []

        local_pairs = []
        best_s_score = -1
        worst_s_score = 11
        best_s_resp = s_resps[0]
        worst_s_resp = s_resps[0]
        s_dims_best = {}

        # Judge each student response against teacher
        for s_resp in s_resps:
            result = multi_judge(client, teacher_model, prompt, s_resp, t_resp)
            if not result:
                continue

            s_score = compute_score(result.get("a", {}))
            t_score = compute_score(result.get("b", {}))

            if s_score > best_s_score:
                best_s_score = s_score
                best_s_resp = s_resp
                s_dims_best = result.get("a", {})
            if s_score < worst_s_score:
                worst_s_score = s_score
                worst_s_resp = s_resp

            # ELO update
            winner = result.get("winner", "tie")
            if winner == "a":
                elo.update("student", "teacher")
            elif winner == "b":
                elo.update("teacher", "student")
            else:
                elo.update_tie("student", "teacher")

        t_score_final = compute_score(result.get("b", {})) if result else 5
        gap = t_score_final - best_s_score

        judgments.append({
            "prompt": prompt, "gap": gap,
            "student_score": best_s_score, "teacher_score": t_score_final,
            "student_dims": s_dims_best, "teacher_dims": result.get("b", {}) if result else {},
        })

        # DPO pair: teacher vs best student (if teacher wins)
        if gap > 0.5:
            local_pairs.append({
                "prompt": prompt, "chosen": t_resp, "rejected": best_s_resp,
                "margin": gap, "type": "teacher_vs_student",
            })

        # DPO pair: best_of_K vs worst_of_K (self-play)
        if best_s_score - worst_s_score > 1.0 and best_s_resp != worst_s_resp:
            local_pairs.append({
                "prompt": prompt, "chosen": best_s_resp, "rejected": worst_s_resp,
                "margin": best_s_score - worst_s_score, "type": "self_play",
            })

        return local_pairs

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_judge_prompt, p) for p in prompts]
        for f in tqdm(as_completed(futs), total=len(prompts)):
            pairs = f.result()
            dpo_pairs.extend(pairs)

    # --- Step 4: Hard example mining ---
    # Oversample hard examples (prompts with large gaps)
    judgments.sort(key=lambda x: x.get("gap", 0), reverse=True)
    hard_pairs = [p for p in dpo_pairs if p.get("margin", 0) > 2.0]
    if hard_pairs:
        # Add hard examples again (2x weight)
        dpo_pairs.extend(hard_pairs)
        print(f"  Hard mining: +{len(hard_pairs)} oversampled pairs")

    # --- Step 5: Generate harder prompts for next round ---
    print(f"\n[4/5] Analyzing weaknesses & generating harder prompts...")
    weak_dims, hard_prompts = find_weaknesses(judgments)
    new_prompts = generate_harder_prompts(client, teacher_model, weak_dims, hard_prompts)
    print(f"  Weak dims: {weak_dims}, New prompts: {len(new_prompts)}")

    # --- Stats ---
    elo.record(round_num)
    avg_s = sum(j["student_score"] for j in judgments) / len(judgments) if judgments else 0
    avg_t = sum(j["teacher_score"] for j in judgments) / len(judgments) if judgments else 0
    n_tvs = sum(1 for p in dpo_pairs if p["type"] == "teacher_vs_student")
    n_sp = sum(1 for p in dpo_pairs if p["type"] == "self_play")

    stats = {
        "round": round_num,
        "avg_student": round(avg_s, 2), "avg_teacher": round(avg_t, 2),
        "gap": round(avg_t - avg_s, 2),
        "elo_student": round(elo.ratings["student"]),
        "elo_teacher": round(elo.ratings["teacher"]),
        "elo_gap": round(elo.gap()),
        "dpo_pairs_total": len(dpo_pairs),
        "dpo_teacher_vs_student": n_tvs,
        "dpo_self_play": n_sp,
        "weak_dims": weak_dims,
        "new_prompts": len(new_prompts),
    }

    print(f"\n[5/5] Round {round_num} summary:")
    print(f"  Score: student={avg_s:.1f} teacher={avg_t:.1f} gap={avg_t-avg_s:.1f}")
    print(f"  ELO:   student={elo.ratings['student']:.0f} teacher={elo.ratings['teacher']:.0f}")
    print(f"  DPO:   {n_tvs} teacher_vs_student + {n_sp} self_play = {len(dpo_pairs)} total")

    # Save
    rd = Path(output_dir) / f"round{round_num}"
    rd.mkdir(parents=True, exist_ok=True)
    with open(rd / "dpo_pairs.jsonl", "w") as f:
        for p in dpo_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(rd / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    if new_prompts:
        with open(rd / "new_prompts.jsonl", "w") as f:
            for p in new_prompts:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    return dpo_pairs, new_prompts, stats


def dpo_train(model, tokenizer, dpo_pairs: list[dict], output_dir: str,
              learning_rate: float = 5e-5, num_epochs: int = 1, batch_size: int = 2):
    if len(dpo_pairs) < 4:
        print("  Too few pairs, skip DPO")
        return model

    # Margin-weighted: shuffle but keep hard examples oversampled from mining step
    random.shuffle(dpo_pairs)

    records = [{
        "prompt": [{"role": "user", "content": p["prompt"]}],
        "chosen": [{"role": "assistant", "content": p["chosen"]}],
        "rejected": [{"role": "assistant", "content": p["rejected"]}],
    } for p in dpo_pairs]

    ds = Dataset.from_list(records)

    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            bf16=True,
            logging_steps=5,
            save_steps=100,
            save_total_limit=1,
            max_length=2048,
            max_prompt_length=512,
            report_to="none",
            gradient_checkpointing=True,
        ),
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print(f"  DPO training: {len(ds)} pairs, lr={learning_rate:.1e}")
    trainer.train()
    trainer.save_model(output_dir)
    return model


# ============================================================
# Entry point
# ============================================================

def run(
    student_path: str,
    teacher_model: str = "glm-4-plus",
    prompts_file: str = "data/seeds/luxun_seeds.jsonl",
    output_dir: str = "outputs/adversarial",
    rounds: int = 5,
    sample_size: int = 300,
    k_samples: int = 3,
    dpo_lr: float = 5e-5,
    dpo_epochs: int = 1,
    max_workers: int = 8,
    elo_convergence: float = 100,
):
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        raise ValueError("ZHIPU_API_KEY not set")
    client = ZhipuAI(api_key=api_key)

    # Load seed prompts, subsample for adversarial (5000 would take days)
    prompts = []
    with open(prompts_file) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    if sample_size and len(prompts) > sample_size:
        prompts = random.sample(prompts, sample_size)
        print(f"  Sampled {sample_size} prompts for adversarial")

    # Load student
    print(f"Loading student: {student_path}")
    tokenizer = AutoTokenizer.from_pretrained(student_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        student_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )

    # LoRA for DPO phase
    try:
        model = PeftModel.from_pretrained(model, student_path)
        print("  Loaded existing LoRA adapter")
    except Exception:
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64,
            lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ))
        print("  Created new LoRA adapter for DPO")

    elo = ELO()
    all_stats = []

    for r in range(1, rounds + 1):
        pairs, new_prompts, stats = adversarial_round(
            model, tokenizer, client, teacher_model,
            prompts, r, output_dir, elo, k_samples, max_workers,
        )
        all_stats.append(stats)

        # Convergence check
        if elo.gap() <= elo_convergence:
            print(f"\n  CONVERGED: ELO gap {elo.gap():.0f} <= {elo_convergence}")
            break

        # DPO
        dpo_dir = f"{output_dir}/round{r}/model"
        model = dpo_train(model, tokenizer, pairs, dpo_dir,
                          learning_rate=dpo_lr, num_epochs=dpo_epochs)

        # Inject harder prompts for next round
        if new_prompts:
            prompts.extend(new_prompts)
            print(f"  Prompt pool: {len(prompts)} (+{len(new_prompts)} new)")

        # Decay LR
        dpo_lr *= 0.7

    # Save final
    final_dir = f"{output_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Summary
    print(f"\n{'='*60}")
    print("  ADVERSARIAL DISTILLATION COMPLETE")
    print(f"{'='*60}")
    for s in all_stats:
        print(f"  R{s['round']}: gap={s['gap']:+.1f} elo={s['elo_student']:.0f}v{s['elo_teacher']:.0f} "
              f"pairs={s['dpo_pairs_total']} weak={s['weak_dims']}")
    print(f"\n  Final ELO gap: {elo.gap():.0f}")
    print(f"  Model -> {final_dir}")

    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump({"elo_history": elo.history, "rounds": all_stats}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_path", required=True)
    parser.add_argument("--teacher_model", default="glm-4-plus")
    parser.add_argument("--prompts_file", default="data/seeds/luxun_seeds.jsonl")
    parser.add_argument("--output_dir", default="outputs/adversarial")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--sample_size", type=int, default=300)
    parser.add_argument("--k_samples", type=int, default=3)
    parser.add_argument("--dpo_lr", type=float, default=5e-5)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--elo_convergence", type=float, default=100)
    args = parser.parse_args()
    run(**vars(args))
