"""
Progressive Tacit Knowledge Distillation trainer.

Polanyi's model of knowledge acquisition:
  Phase 1 (Explicit): Learn from CoT — teacher's reasoning made visible
  Phase 2 (Internalization): Learn from direct answers + perspectives —
           compress explicit reasoning into tacit competence
  Phase 3 (Calibration): Learn confidence/uncertainty —
           "knowing what you don't know"

Each phase uses decreasing learning rate, mimicking how experts
first learn rules explicitly, then internalize them, then develop
calibrated intuition.

Usage:
    python -m src.train_sft --config configs/distill_qwen3_8b.yaml
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

try:
    from src.persona import PERSONA_SYSTEM
except ImportError:
    PERSONA_SYSTEM = "You are a helpful, accurate assistant."


def load_config(config_path: str | None) -> dict:
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def load_phase_data(data_dir: str, phase: int, max_samples: int | None = None) -> Dataset:
    """Load data for a specific training phase."""
    data_dir = Path(data_dir)

    if phase == 1:
        files = [data_dir / "phase1_cot.jsonl"]
    elif phase == 2:
        files = [data_dir / "phase2_direct.jsonl"]
    elif phase == 3:
        files = [data_dir / "phase3_meta.jsonl"]
    else:
        # All phases combined (fallback)
        files = list(data_dir.glob("phase*.jsonl"))

    records = []
    for f in files:
        if not f.exists():
            print(f"  Warning: {f} not found, skipping")
            continue
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            records.append({
                "messages": [
                    {"role": "system", "content": PERSONA_SYSTEM},
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": item["response"]},
                ]
            })

    if max_samples:
        records = records[:max_samples]
    return Dataset.from_list(records)


def run_phase(
    model,
    tokenizer,
    phase: int,
    dataset: Dataset,
    output_dir: str,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_seq_length: int,
    bf16: bool,
    save_steps: int,
    logging_steps: int,
):
    phase_dir = f"{output_dir}/phase{phase}"
    phase_names = {1: "Explicit (CoT)", 2: "Internalization", 3: "Calibration"}

    if len(dataset) == 0:
        print(f"Phase {phase} ({phase_names[phase]}): no data, skipping")
        return model

    split = dataset.train_test_split(test_size=0.05, seed=42) if len(dataset) > 20 else None
    train_ds = split["train"] if split else dataset
    eval_ds = split["test"] if split else None

    print(f"\n{'='*50}")
    print(f"Phase {phase}: {phase_names[phase]}")
    print(f"  Samples: {len(train_ds)}, LR: {learning_rate}")
    print(f"{'='*50}")

    args = SFTConfig(
        output_dir=phase_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=bf16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=save_steps if eval_ds else None,
        max_seq_length=max_seq_length,
        report_to="none",
        seed=42,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(phase_dir)
    print(f"Phase {phase} done -> {phase_dir}")
    return model


def train(
    student_model: str = "Qwen/Qwen3-8B",
    data_dir: str = "data/teacher_outputs",
    output_dir: str = "outputs/qwen3-8b-distill",
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
    max_seq_length: int = 2048,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_target_modules: list[str] | None = None,
    max_samples: int | None = None,
    bf16: bool = True,
    save_steps: int = 200,
    logging_steps: int = 10,
    # Phase-specific LR multipliers (Polanyi progression)
    phase1_lr_mult: float = 1.0,    # Explicit: full LR
    phase2_lr_mult: float = 0.5,    # Internalization: gentler
    phase3_lr_mult: float = 0.25,   # Calibration: fine-tune
    **kwargs,
):
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    print(f"Student: {student_model}")
    tokenizer = AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        device_map, torch_dtype = "auto", torch.bfloat16 if bf16 else torch.float16
    elif torch.backends.mps.is_available():
        device_map, torch_dtype, bf16 = "mps", torch.float16, False
    else:
        device_map, torch_dtype, bf16 = "cpu", torch.float32, False

    model = AutoModelForCausalLM.from_pretrained(
        student_model, torch_dtype=torch_dtype, device_map=device_map,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, target_modules=lora_target_modules, bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # === Progressive training ===
    for phase in [1, 2, 3]:
        dataset = load_phase_data(data_dir, phase, max_samples)
        lr = learning_rate * [0, phase1_lr_mult, phase2_lr_mult, phase3_lr_mult][phase]
        model = run_phase(
            model, tokenizer, phase, dataset, output_dir,
            learning_rate=lr, num_epochs=num_epochs, batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_seq_length=max_seq_length, bf16=bf16,
            save_steps=save_steps, logging_steps=logging_steps,
        )

    # Save final
    final_dir = f"{output_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nAll phases complete -> {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--student_model", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config.update({k: v for k, v in vars(args).items() if v is not None and k != "config"})

    defaults = {
        "student_model": "Qwen/Qwen3-8B",
        "data_dir": "data/teacher_outputs",
        "output_dir": "outputs/qwen3-8b-distill",
        "num_epochs": 3, "batch_size": 4, "learning_rate": 1e-4,
        "max_seq_length": 2048, "lora_r": 64,
    }
    for k, v in defaults.items():
        config.setdefault(k, v)

    train(**config)
