.PHONY: install generate train adversarial merge inference benchmark clean

# ===== Setup =====
install:
	pip install -e ".[eval]"

# ===== Step 1: Extract tacit knowledge from GLM-5.1 =====
generate:
	python -m src.generate_teacher_data \
		--seed_file data/seeds/luxun_seeds.jsonl \
		--output_dir data/teacher_outputs \
		--max_workers 8

# ===== Step 2: 3-phase Polanyi training =====
train:
	python -m src.train_sft --config configs/distill_qwen3_14b.yaml

# ===== Step 3: Adversarial refinement (GAN-style DPO) =====
adversarial:
	python -m src.adversarial \
		--student_path outputs/qwen3-14b-luxun/final \
		--prompts_file data/seeds/luxun_seeds.jsonl \
		--rounds 5 \
		--k_samples 3 \
		--max_workers 8

# ===== Step 4: Merge & export =====
merge:
	python -m src.merge_and_export \
		--base_model Qwen/Qwen3-14B \
		--adapter_path outputs/adversarial/final \
		--output_dir models/qwen3-14b-luxun

# ===== Inference =====
inference:
	python -m src.inference --model_path models/qwen3-14b-luxun

benchmark:
	python -m src.inference --model_path models/qwen3-14b-luxun --benchmark

clean:
	rm -rf outputs/ models/ __pycache__ src/__pycache__
