.PHONY: install generate train-8b train-4b adversarial merge-8b merge-4b inference benchmark clean

# ===== Setup =====
install:
	pip install -e ".[eval]"

# ===== Step 1: Extract tacit knowledge from GLM-5.1 =====
generate:
	python -m src.generate_teacher_data \
		--seed_file data/seeds/seed_prompts.jsonl \
		--output_dir data/teacher_outputs \
		--max_workers 8

# ===== Step 2: 3-phase Polanyi training =====
train-8b:
	python -m src.train_sft --config configs/distill_qwen3_8b.yaml

train-4b:
	python -m src.train_sft --config configs/distill_qwen3_4b.yaml

# ===== Step 3: Adversarial refinement (GAN-style) =====
adversarial:
	python -m src.adversarial \
		--student_path outputs/qwen3-8b-glm5-distill/final \
		--rounds 5 \
		--k_samples 3 \
		--max_workers 8

# ===== Step 4: Merge & export =====
merge-8b:
	python -m src.merge_and_export \
		--base_model Qwen/Qwen3-8B \
		--adapter_path outputs/adversarial/final \
		--output_dir models/qwen3-8b-glm5-distill

merge-4b:
	python -m src.merge_and_export \
		--base_model Qwen/Qwen3-4B \
		--adapter_path outputs/qwen3-4b-glm5-distill/final \
		--output_dir models/qwen3-4b-glm5-distill

# ===== Inference =====
inference:
	python -m src.inference --model_path models/qwen3-8b-glm5-distill

benchmark:
	python -m src.inference --model_path models/qwen3-8b-glm5-distill --benchmark

clean:
	rm -rf outputs/ models/ __pycache__ src/__pycache__
