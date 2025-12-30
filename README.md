# LLM Knowledge Distillation Framework

将经验策略内化到语言模型权重中的知识蒸馏框架。支持 SKD、SeqKD 和 On-Policy GKD 三种蒸馏方法。

## 概述

本项目实现了三种知识蒸馏方法，用于将 Qwen3-8B 在有经验提示（Q+E）下的推理能力内化到模型权重中，使其在无经验提示（Q）时也能成功完成任务。

### 蒸馏方法

| 方法 | 原理 | 特点 |
|------|------|------|
| **SeqKD** | 在学生输入上对教师输出做 SFT | 简单高效，本质是标准 SFT |
| **SKD** | 最小化 Token 级 KL 散度 | 保留教师的概率分布信息（暗知识） |
| **GKD** | 学生自生成 + 教师反馈 | 解决 train-inference 分布不匹配 |

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
│   ├── gkd.yaml                 # GKD 配置
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
│   ├── run_skd.sh               # SKD 启动脚本
│   ├── run_gkd.sh               # GKD 启动脚本
│   └── run_all.sh               # 运行所有实验
├── train_seqkd.py               # SeqKD 训练入口
├── train_skd.py                 # SKD 训练入口
├── train_gkd.py                 # GKD 训练入口
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

# GKD
python train_gkd.py --config configs/gkd.yaml
```

### 2. 多 GPU 分布式训练（8x A100）

```bash
# 使用启动脚本（自动检测 GPU 数量）
bash scripts/run_seqkd.sh
bash scripts/run_skd.sh
bash scripts/run_gkd.sh

# 或手动使用 accelerate
accelerate launch --config_file configs/accelerate_config.yaml \
    train_gkd.py --config configs/gkd.yaml
```

### 3. 命令行参数覆盖

```bash
# 覆盖配置文件中的参数
python train_gkd.py --config configs/gkd.yaml \
    --learning_rate 5e-6 \
    --lmbda 0.8 \
    --beta 0.3 \
    --output_dir outputs/gkd_exp1
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

### GKD 特定配置

```yaml
gkd:
  lmbda: 0.5      # 0=监督, 1=on-policy
  beta: 0.5       # 0=前向KL, 1=反向KL
  temperature: 0.9
  max_new_tokens: 256
```

### SKD 特定配置

```yaml
skd:
  temperature: 2.0     # 软标签温度
  alpha: 0.7           # KL损失权重
  kl_direction: forward  # forward 或 reverse
```

## 关键参数说明

### GKD Lambda (λ)

- `λ = 0.0`: 纯监督蒸馏（类似 SKD）
- `λ = 1.0`: 纯 on-policy（学生生成所有数据）
- `λ = 0.5`: 混合模式（推荐）

### GKD Beta (β)

- `β = 0.0`: 前向 KL 散度（mean-seeking）
- `β = 1.0`: 反向 KL 散度（mode-seeking）
- `β = 0.5`: 广义 JSD（推荐）

### SKD Alpha (α)

- `α = 1.0`: 只使用 KL 损失
- `α = 0.0`: 只使用 CE 损失
- `α = 0.7`: 混合（推荐）

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

