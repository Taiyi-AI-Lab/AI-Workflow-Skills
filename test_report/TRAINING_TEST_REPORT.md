# AI Workflow Skills 本地模型训练测试报告

**测试日期**: 2026年6月11日
**测试环境**: Linux + NVIDIA A100 80GB
**测试模型**: Qwen3.5-0.8B
**训练方法**: GRPO (Group Relative Policy Optimization)

---

## 一、测试概述

### 1.1 测试目标

本次测试旨在验证 **AI Workflow Skills** 中的 `grpo-finetune` 技能在本地 GPU 环境下进行大语言模型微调的可行性、易用性和效果。

**主要测试点：**
- Skills 安装与配置流程
- GRPO 训练脚本的可用性
- 训练过程的稳定性
- 模型微调效果评估

### 1.2 测试结论

> ✅ **测试通过**：AI Workflow Skills 的 grpo-finetune 技能可成功用于本地 GRPO 微调，文档清晰，模板完善，训练流程稳定。

---

## 二、测试环境

### 2.1 硬件配置

| 组件 | 规格 |
|------|------|
| **GPU** | NVIDIA A100-SXM4-80GB |
| **可用显存** | 81-85 GB |
| **CPU** | 未记录 |
| **内存** | 未记录 |

### 2.2 软件环境

| 软件/库 | 版本 |
|---------|------|
| **Python** | 3.12 |
| **PyTorch** | (CUDA 可用) |
| **Transformers** | 5.6.2 |
| **PEFT** | 0.19.1 |
| **TRL** | (GRPO 可用) |
| **vLLM** | 0.19.1 (存在版本警告，但不影响训练) |

---

## 三、数据与模型

### 3.1 训练数据

| 数据集 | 来源 | 格式 | 样本数 |
|--------|------|------|--------|
| **GSM8K 训练集** | data_small | JSONL | 200 |
| **GSM8K 测试集** | data_small | JSONL | 100 |

**数据格式示例：**
```json
{
  "answer": "72",
  "prompt": "You are a helpful math assistant. Solve the following math problem step by step. Put your reasoning inside <think...</think tags and your final answer inside <answer>...</answer> tags. Output ONLY the XML, with no other text before or after.\n\nProblem: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
}
```

### 3.2 基座模型

| 属性 | 值 |
|------|-----|
| **模型名称** | Qwen3.5-0.8B |
| **模型大小** | 1.7 GB |
| **参数量** | 0.8B |
| **架构** | Qwen3_5ForConditionalGeneration |

---

## 四、测试过程

### 4.1 Skills 安装

1. **克隆技能仓库**
   ```bash
   git clone https://github.com/Taiyi-AI-Lab/AI-Workflow-Skills.git
   ```

2. **安装到项目**
   - 将技能复制到项目本地目录：`.claude/skills/`
   - 安装的技能：
     - `grpo-finetune` (本地 CUDA GRPO 训练)
     - `mint-lora-training` (云端 MinT 训练)

3. **技能结构验证**
   ```
   grpo-finetune/
   ├── SKILL.md           # 技能说明文档
   ├── README.md          # 使用指南
   ├── manifest.json      # 技能元数据
   ├── templates/         # 训练脚本模板
   │   ├── rewards.py
   │   ├── train_grpo.py
   │   └── evaluate.py
   └── references/        # 参考文档
       ├── troubleshooting.md
       └── training-curves.md
   ```

### 4.2 模型准备

将基座模型从 `/data/models/Qwen3.5-0.8B` 复制到项目目录：

```bash
cp -r /data/models/Qwen3.5-0.8B /data/wsbi/test_skills_train/models/
```

### 4.3 训练脚本定制

根据数据格式定制了以下脚本：

| 文件 | 主要修改内容 |
|------|-------------|
| `rewards.py` | 调整 XML 标签匹配 (`<think...` 和 `<answer>`) |
| `train_grpo.py` | 改用 JSONL 数据加载，适配数据格式 |
| `evaluate.py` | JSONL 数据评估脚本 |

### 4.4 环境检查

✅ 所有检查通过：
- CUDA: 可用
- GPU: A100 80GB
- Transformers: 5.6.2
- PEFT: 0.19.1
- TRL GRPO: OK

### 4.5 Dry-Run 测试

**目的**: 验证训练流程完整性

**配置**:
```bash
--max_steps 2
--per_device_train_batch_size 1
--num_generations 4
--max_completion_length 256
```

**结果**: ✅ 成功完成 (2 步，用时 2 分钟)

### 4.6 完整训练

**训练配置**:

| 参数 | 值 |
|------|-----|
| max_steps | 100 |
| per_device_train_batch_size | 4 |
| gradient_accumulation_steps | 4 |
| num_generations | 8 |
| max_completion_length | 512 |
| learning_rate | 5e-6 |
| beta (KL penalty) | 0.04 |
| temperature | 0.9 |
| LoRA rank | 16 |

**训练时长**: 约 82 分钟 (4941 秒)

**训练指标变化**:

| 步数 | format_reward/mean | correctness_reward/mean | kl |
|------|-------------------|----------------------|-----|
| 1 | 0.175 | 0.000 | 0.000 |
| 20 | 0.529 | 0.150 | 0.002 |
| 50 | 0.581 | 0.269 | 0.005 |
| 100 | 0.627 | 0.281 | 0.004 |

---

## 五、评估结果

### 5.1 定量评估

**测试集**: 50 个 GSM8K 测试样本

| 指标 | 数值 | 说明 |
|------|------|------|
| **准确率** | 34.0% (17/50) | 答案正确率 |
| **平均格式得分** | 0.652 | XML 结构符合度 |
| **完美格式样本** | 0/50 (0%) | 格式得分 ≥0.99 的样本 |

### 5.2 样本分析

**正确示例**:
```
问题: Janet 的鸡蛋销售问题
GT: 18
Pred: 18
Format: 0.70
状态: ✅ 正确
```

**错误示例**:
```
问题: 蓝色纤维和白色纤维问题
GT: 3
Pred: (推理过程未完成)
Format: 0.70
状态: ❌ 错误
```

### 5.3 效果分析

1. **格式学习效果显著**: 格式奖励从 0.175 提升到 0.627，提升 258%
2. **正确率有提升**: 正确率奖励从 0 提升到 0.281
3. **需要更多训练**: 100 步训练较少，正确率提升有限；500-1000 步效果会更好

---

## 六、遇到的问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| vLLM 版本警告 | TRL 支持 v0.10.2-v0.12.0，系统安装 v0.19.1 | 设置 `VLLM_AVAILABLE=0` 使用 HF 生成 |
| 数据格式不匹配 | 模板默认 HF Hub 数据集，本地为 JSONL | 定制数据加载函数 |
| XML 标签格式 | 数据使用 `<think...` 模板为 `<think_box>` | 调整 rewards.py 中的标签匹配 |

---

## 七、Skills 质量评估

### 7.1 文档质量 ⭐⭐⭐⭐⭐

- ✅ SKILL.md 结构清晰，步骤明确
- ✅ 提供完整的参数说明
- ✅ 包含常见问题解答 (troubleshooting.md)
- ✅ 提供训练曲线参考数据

### 7.2 模板质量 ⭐⭐⭐⭐⭐

- ✅ 代码可直接运行
- ✅ 自助测试功能完善 (rewards.py 自测)
- ✅ 参数化设计良好
- ✅ 错误处理清晰

### 7.3 易用性 ⭐⭐⭐⭐

- ✅ 环境检查脚本完善
- ✅ Dry-run 模式验证流程
- ✅ 默认配置合理
- ⚠️ 数据格式适配需要一定代码能力

---

## 八、总结与建议

### 8.1 测试总结

1. **功能验证**: AI Workflow Skills 的 grpo-finetune 技能完全可用于本地 GRPO 微调
2. **流程完整**: 从环境检查到模型评估的完整流程都有覆盖
3. **效果可验证**: 训练指标和评估结果清晰可见

### 8.2 建议

| 建议 | 优先级 | 说明 |
|------|--------|------|
| 增加训练步数 | 高 | 100 步太少，建议 500-1000 步 |
| 支持 JSONL 数据 | 中 | 模板可增加 JSONL 数据示例 |
| 完善格式奖励 | 中 | 100 步内完美格式样本为 0，可优化奖励函数 |

### 8.3 下一步

- [ ] 使用更多训练步数 (500-1000) 重新训练
- [ ] 对比基座模型与微调模型的详细表现
- [ ] 测试其他数据集和任务类型
- [ ] 尝试 mint-lora-training 云端训练技能

---

## 九、附录

### 9.1 训练命令

```bash
# Dry-run
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 train_grpo.py \
    --max_steps 2 --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 --num_generations 4 \
    --max_completion_length 256 --eval_strategy no --report_to none

# Full training
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 train_grpo.py \
    --max_steps 100 --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 --num_generations 8 \
    --max_completion_length 512 --eval_strategy no --report_to none
```

### 9.2 评估命令

```bash
python3 evaluate.py \
    --model_path ./outputs/run_1/final_model \
    --data_dir /data/wsbi/test_skills_train/data_small \
    --num_samples 50
```

### 9.3 输出文件结构

```
grpo-training/
├── outputs/run_1/
│   ├── final_model/          # 完整微调模型
│   │   ├── config.json
│   │   ├── model.safetensors*
│   │   └── ...
│   └── lora_adapter/         # LoRA adapter
│       ├── adapter_config.json
│       └── adapter_model.safetensors
├── models/Qwen3.5-0.8B/      # 复制的基座模型
├── rewards.py                 # 定制的奖励函数
├── train_grpo.py              # 定制的训练脚本
├── evaluate.py                # 定制的评估脚本
├── eval_results.json         # 详细评估结果
├── training.log              # 训练日志
└── eval.log                  # 评估日志
```

---

**报告生成时间**: 2026-06-11
**测试执行者**: Claude Opus 4.8
