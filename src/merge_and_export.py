"""
Merge LoRA adapter weights into the base model and export for deployment.

Usage:
    python -m src.merge_and_export \
        --base_model Qwen/Qwen3-1.7B \
        --adapter_path outputs/qwen3-1.7b-distill \
        --output_dir models/qwen3-1.7b-glm5-distill \
        --push_to_hub your-username/qwen3-1.7b-glm5-distill
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_and_export(
    base_model: str,
    adapter_path: str,
    output_dir: str,
    push_to_hub: str | None = None,
):
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    if push_to_hub:
        print(f"Pushing to HuggingFace Hub: {push_to_hub}")
        model.push_to_hub(push_to_hub)
        tokenizer.push_to_hub(push_to_hub)
        print(f"Pushed to https://huggingface.co/{push_to_hub}")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and export")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--push_to_hub", type=str, default=None, help="HuggingFace Hub repo ID")
    args = parser.parse_args()

    merge_and_export(**vars(args))
