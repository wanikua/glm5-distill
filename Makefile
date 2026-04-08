.PHONY: install generate train-8b train-4b merge export-gguf inference benchmark clean

# ===== Setup =====
install:
	pip install -e ".[eval]"

# ===== Step 1: Generate teacher data from GLM-5.1 API =====
generate:
	python -m src.generate_teacher_data \
		--seed_file data/seeds/seed_prompts.jsonl \
		--output_file data/teacher_outputs/train.jsonl \
		--model glm-5.1 \
		--max_workers 4

# ===== Step 2: Train student model =====
train-8b:
	python -m src.train_sft --config configs/distill_qwen3_8b.yaml

train-4b:
	python -m src.train_sft --config configs/distill_qwen3_4b.yaml

# ===== Step 3: Merge LoRA and export =====
merge-8b:
	python -m src.merge_and_export \
		--base_model Qwen/Qwen3-8B \
		--adapter_path outputs/qwen3-8b-glm5-distill \
		--output_dir models/qwen3-8b-glm5-distill

merge-4b:
	python -m src.merge_and_export \
		--base_model Qwen/Qwen3-4B \
		--adapter_path outputs/qwen3-4b-glm5-distill \
		--output_dir models/qwen3-4b-glm5-distill

# ===== Step 4: Convert to GGUF for llama.cpp / Ollama =====
export-gguf:
	@echo "Install llama.cpp first: brew install llama.cpp"
	@echo "Then run: llama-quantize models/qwen3-8b-glm5-distill Q4_K_M models/qwen3-8b-glm5-distill-Q4.gguf"

# ===== Inference =====
inference:
	python -m src.inference --model_path models/qwen3-8b-glm5-distill

benchmark:
	python -m src.inference --model_path models/qwen3-8b-glm5-distill --benchmark

# ===== Utils =====
clean:
	rm -rf outputs/ models/ __pycache__ src/__pycache__
