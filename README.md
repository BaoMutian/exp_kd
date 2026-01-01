# LLM Knowledge Distillation Framework

将经验策略内化到语言模型权重中的知识蒸馏框架。支持 SKD、SeqKD 和 On-Policy KD 三种蒸馏方法。

## 概述

本项目实现了三种知识蒸馏方法，用于将 Qwen3-8B 在有经验提示（Q+E）下的推理能力内化到模型权重中，使其在无经验提示（Q）时也能成功完成任务。

### 蒸馏方法对比

| 方法             | 训练数据来源     | 原理                         | 特点                             |
| ---------------- | ---------------- | ---------------------------- | -------------------------------- |
| **SeqKD**        | 固定（教师输出） | 在学生输入上对教师输出做 SFT | 简单高效，本质是标准 SFT         |
| **SKD**          | 固定（真值序列） | 最小化 Token 级 KL 散度      | 保留教师的概率分布信息（暗知识） |
| **On-Policy KD** | 学生实时生成     | 学生采样 + 教师评分          | 解决训练-推理分布不匹配问题      |

### On-Policy KD 优势

On-Policy KD 基于 GKD (Generalized Knowledge Distillation) 论文思想：

1. **消除分布不匹配**：学生在自己生成的序列上学习，而非固定的真值数据
2. **从错误中恢复**：学生学会处理自己可能犯的错误，提高推理时的鲁棒性
3. **灵活的损失函数**：支持 JSD 损失，可在前向 KL 和反向 KL 之间插值

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
│   ├── base.yaml                   # 基础配置
│   ├── seqkd.yaml                  # SeqKD 配置
│   ├── skd.yaml                    # SKD 配置
│   ├── onpolicy_kd.yaml            # On-Policy KD 配置
│   ├── accelerate_config.yaml      # SKD/SeqKD 分布式配置 (ZeRO-3)
│   ├── accelerate_onpolicy_kd.yaml # On-Policy KD 分布式配置 (ZeRO-2)
│   ├── deepspeed_config.json       # DeepSpeed ZeRO-3 配置
│   └── deepspeed_zero2_config.json # DeepSpeed ZeRO-2 配置
├── src/
│   ├── data/
│   │   └── dataset.py           # 数据加载与预处理
│   ├── trainers/
│   │   ├── skd_trainer.py       # SKD 自定义 Trainer
│   │   └── onpolicy_kd_trainer.py # On-Policy KD 自定义 Trainer
│   └── utils/
│       └── config.py            # 配置加载工具
├── scripts/
│   ├── run_seqkd.sh             # SeqKD 启动脚本
│   ├── run_skd.sh               # SKD 启动脚本
│   └── run_onpolicy_kd.sh       # On-Policy KD 启动脚本
├── train_seqkd.py               # SeqKD 训练入口
├── train_skd.py                 # SKD 训练入口
├── train_onpolicy_kd.py         # On-Policy KD 训练入口
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

# On-Policy KD
python train_onpolicy_kd.py --config configs/onpolicy_kd.yaml
```

### 2. 多 GPU 分布式训练（8x A100）

```bash
# 使用启动脚本（自动检测 GPU 数量）
bash scripts/run_seqkd.sh
bash scripts/run_skd.sh
bash scripts/run_onpolicy_kd.sh

# 或手动使用 accelerate
accelerate launch --config_file configs/accelerate_config.yaml \
    train_onpolicy_kd.py --config configs/onpolicy_kd.yaml
```

### 3. 命令行参数覆盖

```bash
# SKD 参数覆盖
python train_skd.py --config configs/skd.yaml \
    --learning_rate 5e-6 \
    --temperature 4.0 \
    --alpha 0.5 \
    --output_dir outputs/skd_exp1

# On-Policy KD 参数覆盖
python train_onpolicy_kd.py --config configs/onpolicy_kd.yaml \
    --beta 0.5 \
    --generation_temperature 0.9 \
    --max_new_tokens 256 \
    --output_dir outputs/onpolicy_exp1
```

## 配置说明

### 基础配置 (`base.yaml`)

```yaml
model:
  student_path: "Qwen/Qwen3-8B"
  teacher_path: "Qwen/Qwen3-8B"
  enable_thinking: false # Qwen3 non-thinking 模式
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
  temperature: 4.0 # 软标签温度
  alpha: 0.5 # KL损失权重
  kl_direction: reverse # forward 或 reverse
```

### On-Policy KD 特定配置

```yaml
onpolicy_kd:
  temperature: 1.0 # 损失计算时的温度
  beta: 0.5 # JSD 插值系数 (0=前向KL, 0.5=对称JSD, 1=反向KL)
  max_new_tokens: 256 # 最大生成长度
  generation_temperature: 0.9 # 生成采样温度
  num_samples: 1 # 每个输入采样的回复数量
```

## 关键参数说明

### SKD Alpha (α)

- `α = 1.0`: 只使用 KL 损失
- `α = 0.0`: 只使用 CE 损失
- `α = 0.5`: 混合（推荐）

### SKD Temperature

- 较高温度（如 4.0）使概率分布更平滑，便于学生学习
- 较低温度（如 1.0）保留更多概率分布的尖锐性

### SKD KL Direction

- `forward`: KL(teacher || student)，mean-seeking
- `reverse`: KL(student || teacher)，mode-seeking（推荐）

### On-Policy KD Beta (β)

- `β = 0.0`: 前向 KL，mean-seeking，覆盖教师所有模式
- `β = 0.5`: 对称 JSD，平衡（推荐）
- `β = 1.0`: 反向 KL，mode-seeking，聚焦高概率区域

### On-Policy KD Generation Settings

- `generation_temperature`: 控制生成多样性（较高 = 更多样）
- `num_samples`: 每个输入采样多条回复可以提供更稳定的梯度

## 内存优化

对于 8x A100 90GB 配置：

1. **DeepSpeed ZeRO**：
   - SKD/SeqKD：使用 ZeRO-3（参数分片）
   - **On-Policy KD：使用 ZeRO-2**（因为 `generate()` 与 ZeRO-3 不兼容）
2. **梯度检查点**: 默认启用，降低激活内存
3. **BF16 混合精度**: 默认启用

如果仍然 OOM，可以：

- 减小 `per_device_train_batch_size`
- 增大 `gradient_accumulation_steps`
- 启用 LoRA（在配置中设置 `peft.enabled: true`）
- 对于 On-Policy KD，减小 `max_new_tokens`

> ⚠️ **重要**: On-Policy KD 在训练过程中调用 `model.generate()`，这与 DeepSpeed ZeRO-3 的参数分片机制不兼容。因此 On-Policy KD 必须使用 ZeRO-2 配置（`accelerate_onpolicy_kd.yaml`）。

## On-Policy KD 训练流程

```
学生模型 (输入: Q)
    │
    ▼
采样生成 N 条回复 A₁, A₂, ..., Aₙ
    │
    ▼
教师模型 (输入: Q+E + Aᵢ) → 计算 P(Aᵢ|Q+E)
    │
    ▼
计算 JSD(P_teacher, P_student) 损失
    │
    ▼
更新学生模型权重
```

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
