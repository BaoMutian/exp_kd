# LLM Knowledge Distillation Framework

将经验策略内化到语言模型权重中的知识蒸馏框架。支持 SKD 和 SeqKD 两种蒸馏方法。

## 概述

本项目实现了两种知识蒸馏方法，用于将 Qwen3-8B 在有经验提示（Q+E）下的推理能力内化到模型权重中，使其在无经验提示（Q）时也能成功完成任务。

### 蒸馏方法

| 方法 | 原理 | 特点 |
|------|------|------|
| **SeqKD** | 在学生输入上对教师输出做 SFT | 简单高效，本质是标准 SFT |
| **SKD** | 最小化 Token 级 KL 散度 | 保留教师的概率分布信息（暗知识） |

## 安装

```bash
# 激活 conda 环境
conda activate icml26

# 安装依赖
pip install -r requirements.txt

# （可选）安装 Flash Attention
pip install flash-attn --no-build-isolation
```

## 项目结构

```
kd/
├── configs/
│   ├── base.yaml                 # 基础配置
│   ├── seqkd.yaml               # SeqKD 配置
│   ├── skd.yaml                 # SKD 配置
│   ├── accelerate_config.yaml   # 分布式训练配置
│   └── deepspeed_config.json    # DeepSpeed ZeRO-3 配置
├── src/
│   ├── data/
│   │   └── dataset.py           # 数据加载与预处理
│   ├── trainers/
│   │   └── skd_trainer.py       # SKD 自定义 Trainer
│   └── utils/
│       └── config.py            # 配置加载工具
├── scripts/
│   ├── run_seqkd.sh             # SeqKD 启动脚本
│   └── run_skd.sh               # SKD 启动脚本
├── train_seqkd.py               # SeqKD 训练入口
├── train_skd.py                 # SKD 训练入口
└── train_data/                  # 训练数据
    ├── alfworld_valid_train_*.jsonl        # 教师数据（带经验）
    └── noexp_alfworld_valid_train_*.jsonl  # 学生数据（无经验）
```

## 使用方法

### 1. 单 GPU 训练

```bash
# SeqKD
python train_seqkd.py --config configs/seqkd.yaml

# SKD
python train_skd.py --config configs/skd.yaml
```

### 2. 多 GPU 分布式训练（8x A100）

```bash
# 使用启动脚本（自动检测 GPU 数量）
bash scripts/run_seqkd.sh
bash scripts/run_skd.sh

# 或手动使用 accelerate
accelerate launch --config_file configs/accelerate_config.yaml \
    train_skd.py --config configs/skd.yaml
```

### 3. 命令行参数覆盖

```bash
# 覆盖配置文件中的参数
python train_skd.py --config configs/skd.yaml \
    --learning_rate 5e-6 \
    --temperature 4.0 \
    --alpha 0.5 \
    --output_dir outputs/skd_exp1
```

## 配置说明

### 基础配置 (`base.yaml`)

```yaml
model:
  student_path: "Qwen/Qwen3-8B"
  teacher_path: "Qwen/Qwen3-8B"
  enable_thinking: false  # Qwen3 non-thinking 模式
  torch_dtype: bfloat16

training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2e-5
  num_train_epochs: 3
```

### SKD 特定配置

```yaml
skd:
  temperature: 4.0       # 软标签温度
  alpha: 0.5             # KL损失权重
  kl_direction: reverse  # forward 或 reverse
```

## 关键参数说明

### SKD Alpha (α)

- `α = 1.0`: 只使用 KL 损失
- `α = 0.0`: 只使用 CE 损失
- `α = 0.5`: 混合（推荐）

### SKD Temperature

- 较高温度（如 4.0）使概率分布更平滑，便于学生学习
- 较低温度（如 1.0）保留更多概率分布的尖锐性

### KL Direction

- `forward`: KL(teacher || student)，mean-seeking
- `reverse`: KL(student || teacher)，mode-seeking（推荐）

## 内存优化

对于 8x A100 90GB 配置：

1. **使用 ZeRO-3**: 默认启用，参数和优化器状态分片
2. **梯度检查点**: 默认启用，降低激活内存
3. **BF16 混合精度**: 默认启用

如果仍然 OOM，可以：
- 减小 `per_device_train_batch_size`
- 增大 `gradient_accumulation_steps`
- 启用 LoRA（在配置中设置 `peft.enabled: true`）

## Qwen3 Non-Thinking 模式

本项目默认使用 Qwen3 的 non-thinking 模式：

```python
tokenizer.apply_chat_template(
    messages,
    enable_thinking=False,  # 关闭思考模式
)
```

推荐采样参数：
- Temperature: 0.7
- TopP: 0.8
- TopK: 20

## License

MIT License
