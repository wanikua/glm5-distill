"""
LoRA SFT training for knowledge distillation.
Trains the student model on GLM-5.1 teacher-generated data.

Usage:
    python -m src.train_sft --config configs/distill_qwen3_1.7b.yaml

    # Or with CLI overrides:
    python -m src.train_sft \
        --student_model Qwen/Qwen3-1.7B \
        --data_file data/teacher_outputs/train.jsonl \
        --output_dir outputs/qwen3-1.7b-distill \
        --num_epochs 3 \
        --batch_size 4
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer


def load_config(config_path: str | None) -> dict:
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def load_training_data(data_file: str, max_samples: int | None = None) -> Dataset:
    """Load teacher-generated JSONL data into HuggingFace Dataset."""
    records = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            records.append({
                "messages": [
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": item["response"]},
                ]
            })

    if max_samples:
        records = records[:max_samples]

    return Dataset.from_list(records)


def train(
    student_model: str = "Qwen/Qwen3-1.7B",
    data_file: str = "data/teacher_outputs/train.jsonl",
    output_dir: str = "outputs/qwen3-1.7b-distill",
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_target_modules: list[str] | None = None,
    max_samples: int | None = None,
    bf16: bool = True,
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    logging_steps: int = 10,
    save_steps: int = 200,
    eval_ratio: float = 0.05,
):
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    print(f"Loading student model: {student_model}")
    tokenizer = AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect device
    if torch.cuda.is_available():
        device_map = "auto"
        torch_dtype = torch.bfloat16 if bf16 else torch.float16
    elif torch.backends.mps.is_available():
        device_map = "mps"
        torch_dtype = torch.float16  # MPS doesn't support bf16 well
        bf16 = False
    else:
        device_map = "cpu"
        torch_dtype = torch.float32
        bf16 = False

    model = AutoModelForCausalLM.from_pretrained(
        student_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    print(f"Loading training data from {data_file}")
    dataset = load_training_data(data_file, max_samples=max_samples)

    # Train/eval split
    if eval_ratio > 0 and len(dataset) > 20:
        split = dataset.train_test_split(test_size=eval_ratio, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset) if eval_dataset else 0}")

    # Training config
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        bf16=bf16,
        fp16=not bf16 and torch_dtype == torch.float16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=save_steps if eval_dataset else None,
        max_seq_length=max_seq_length,
        dataset_kwargs={"skip_prepare_dataset": False},
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA SFT distillation training")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--student_model", type=str, default=None)
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    # Load config file, then override with CLI args
    config = load_config(args.config)
    cli_overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    config.update(cli_overrides)

    # Set defaults for missing keys
    defaults = {
        "student_model": "Qwen/Qwen3-1.7B",
        "data_file": "data/teacher_outputs/train.jsonl",
        "output_dir": "outputs/qwen3-1.7b-distill",
        "num_epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "max_seq_length": 2048,
        "lora_r": 64,
    }
    for k, v in defaults.items():
        config.setdefault(k, v)

    train(**config)
