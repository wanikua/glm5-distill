# GLM-5.1 Distillation

将 [GLM-5.1](https://huggingface.co/zai-org/GLM-5.1)（754B MoE）的能力蒸馏到可在普通电脑上快速推理的小模型中。

## 方案概述

```
GLM-5.1 (754B, Teacher)  →  API 生成高质量数据  →  LoRA SFT 训练  →  小模型 (Student)
```

| 学生模型 | 参数量 | Q4 推理内存 | 适用场景 |
|---------|--------|------------|---------|
| Qwen3-8B | 8B | ~5GB | 推荐 - 8GB+ 内存电脑 |
| Qwen3-4B | 4B | ~2.5GB | 轻量 - 任何电脑/边缘设备 |

## 快速开始

### 1. 安装

```bash
git clone https://github.com/YOUR_USERNAME/glm5-distill.git
cd glm5-distill
pip install -e .
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的智谱 BigModel API Key
```

### 3. 生成教师数据

```bash
# 使用 GLM-5.1 API 生成训练数据
make generate

# 或自定义参数
python -m src.generate_teacher_data \
    --seed_file data/seeds/seed_prompts.jsonl \
    --output_file data/teacher_outputs/train.jsonl \
    --max_workers 8
```

> 建议准备 5000+ 条种子 prompt，覆盖代码、推理、数学、中英文等多个领域。
> `data/seeds/seed_prompts.jsonl` 中已包含 30 条示例。

### 4. 训练

```bash
# 默认: Qwen3-8B（需要 GPU，建议 A100/H100）
make train-8b

# 轻量版: Qwen3-4B
make train-4b
```

### 5. 合并权重 & 导出

```bash
make merge-8b   # 合并 LoRA 权重

# 推送到 HuggingFace Hub
python -m src.merge_and_export \
    --base_model Qwen/Qwen3-8B \
    --adapter_path outputs/qwen3-8b-glm5-distill \
    --output_dir models/qwen3-8b-glm5-distill \
    --push_to_hub your-username/qwen3-8b-glm5-distill
```

### 6. 推理

```bash
# 交互模式
make inference

# 单条推理
python -m src.inference \
    --model_path models/qwen3-8b-glm5-distill \
    --prompt "用 Python 实现一个简单的 HTTP 服务器"

# 速度测试
make benchmark
```

### 7. 部署到 Ollama（可选）

```bash
# 转换为 GGUF 格式
pip install llama-cpp-python
python -c "
from llama_cpp import Llama
# 或使用 llama.cpp 工具链转换
"

# 导入 Ollama
ollama create glm5-distill -f Modelfile
ollama run glm5-distill
```

## 自定义种子数据

编辑 `data/seeds/seed_prompts.jsonl`，每行一个 JSON：

```json
{"prompt": "你的问题或指令", "category": "code"}
```

支持的 category：`code`, `reasoning`, `math`, `chinese`, `chinese_code`, `general`

## 提高蒸馏质量的建议

1. **数据量**：至少 5000 条，越多越好
2. **多样性**：覆盖不同领域和难度等级
3. **数据质量**：可以对 GLM-5.1 的输出做人工筛选
4. **多轮对话**：扩展种子数据支持多轮对话格式
5. **温度调节**：用不同 temperature 生成多样化回复

## 项目结构

```
glm5-distill/
├── configs/                 # 训练配置
│   ├── distill_qwen3_8b.yaml
│   └── distill_qwen3_4b.yaml
├── data/
│   └── seeds/              # 种子 prompt
├── src/
│   ├── generate_teacher_data.py  # Step 1: 调用 GLM-5.1 API
│   ├── train_sft.py              # Step 2: LoRA SFT 训练
│   ├── merge_and_export.py       # Step 3: 合并权重
│   └── inference.py              # Step 4: 推理
├── Makefile
├── pyproject.toml
└── .env.example
```

## License

MIT
