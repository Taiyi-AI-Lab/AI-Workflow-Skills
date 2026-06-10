---
name: grpo-finetune
description: Use when users need local CUDA GRPO fine-tuning for LLMs, including reward function design, TRL GRPOTrainer setup, LoRA training templates, dry runs, evaluation, training-curve interpretation, and troubleshooting.
version: 2.0.0
author: AI Workflow Skills Maintainers
license: MIT
platforms: [linux]
---

# GRPO Fine-Tuning: End-to-End Pipeline

Use this skill when the user wants to fine-tune an LLM using GRPO (Group Relative Policy Optimization) to learn a specific output format or improve task performance. This skill is a complete, self-contained pipeline that an agent can follow step-by-step from environment check to final evaluation.

## What This Skill Provides

- **7-step numbered workflow** from project setup to evaluation
- **Ready-to-run code templates** for rewards, training, evaluation
- **All known pitfalls** with fixes (TRL API, OOM, vLLM, batching)
- **Parameterized** — user specifies: model, dataset, output format; everything else has proven defaults

## Prerequisites

- Python 3.10+, PyTorch with CUDA, TRL >= 0.29, Transformers >= 5.0, PEFT, Datasets
- NVIDIA GPU with at least 4GB free VRAM (LoRA mode)
- HuggingFace access (or mirror via `HF_ENDPOINT`)

## When to Use

- User says "train/fine-tune with GRPO", "RL fine-tuning", "teach model to output X format"
- User wants structured output (XML, JSON, specific tags) with task correctness
- User has a dataset and wants to start training immediately

Don't use for: SFT (use SFTTrainer), DPO (use DPOTrainer), PPO (use PPOTrainer).

---

## Step 0: Gather User Requirements

Before writing any code, confirm these with the user (use defaults if they don't care):

| Parameter | Example | Default |
|-----------|---------|---------|
| Model name | `Qwen/Qwen2.5-0.5B-Instruct` | `Qwen/Qwen2.5-0.5B-Instruct` |
| Dataset | `gsm8k` (HF hub name) | (must specify) |
| Output format | XML with `...` and `...` | XML with think + answer tags |
| Answer extraction | How to get the ground truth answer | `text.split("####")[-1].strip()` for GSM8K |
| Project directory | `/data/my-grpo-project` | `/data/grpo-finetune` |
| Training steps | 500 | 500 |
| GPU VRAM available | ~5GB free | Auto-detect |

If user only specifies dataset, use all defaults.

---

## Step 1: Environment Check

Run these checks first. Stop and fix any failures before proceeding.

**STOP if macOS.** GRPO training requires NVIDIA CUDA GPU. macOS (including Apple Silicon MPS) is NOT supported. Check OS first:

```bash
# Check OS — must be Linux
uname -s
# Expected: Linux
# If "Darwin" (macOS): STOP and tell user "GRPO training requires NVIDIA GPU with CUDA. macOS is not supported. Use a Linux server with NVIDIA GPU."
```

```bash
# Check GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}, Free: {(torch.cuda.mem_get_info()[0]/1e9):.1f}GB')"

# Check packages
python3 -c "import trl; print(f'TRL: {trl.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import peft; print(f'PEFT: {peft.__version__}')"

# Verify GRPO import (may take 30-120s due to lazy loading)
python3 -c "
import os; os.environ['VLLM_AVAILABLE']='0'
from trl import GRPOTrainer, GRPOConfig
print('GRPO OK')
"

# Check GRPOConfig API (version-sensitive!)
python3 -c "
import os; os.environ['VLLM_AVAILABLE']='0'
from trl import GRPOConfig
import inspect
sig = inspect.signature(GRPOConfig.__init__)
params = [n for n in sig.parameters if n not in ('self','kwargs')]
print('Available params:', params)
" 2>&1 | grep -v 'UserWarning\|deprecated\|Modular'
```

**If TRL import hangs:** vLLM version mismatch. Set `VLLM_AVAILABLE=0` and retry with 120s timeout.
**If GRPOConfig missing params:** See [references/troubleshooting.md](references/troubleshooting.md) for version-specific fixes.

---

## Step 2: Create Project Structure

```bash
mkdir -p /data/grpo-finetune
cd /data/grpo-finetune
```

Create these files in order (or copy from templates/):
1. `rewards.py` — Reward functions (format + correctness)
2. `train_grpo.py` — Training script
3. `evaluate.py` — Evaluation script

---

## Step 3: Write Reward Functions

Create `rewards.py`. This is the most critical file — it defines what the model learns.

### Architecture: Two-Component Reward

Use two reward functions passed to GRPOTrainer:
- **format_reward**: Checks output structure (0.0–1.0, smooth gradient)
- **correctness_reward**: Checks answer accuracy (0.0 or 1.0, sparse signal)

### Template

Copy `templates/rewards.py` and customize the three marked sections:

**Section A — Output format tags:** Change `...` / `...` / `<answer>` / `</answer>` to your desired format.

**Section B — Answer extraction:** Change `extract_answer()` to parse your format. Default extracts from `<answer>...</answer>`.

**Section C — Answer normalization:** Change `normalize_answer()` for your comparison logic. Default handles numbers (commas, $, %, fractions).

### Key Rules for Reward Functions

```python
# Rule 1: First param is always 'completions: list[str]'
# Rule 2: Other params = dataset column names (TRL passes them as kwargs)
# Rule 3: Return list[float], one per completion
# Rule 4: **kwargs catches unused columns

def format_reward(completions: list[str], **kwargs) -> list[float]:
    """Returns 0.0–1.0 based on output structure compliance."""
    ...

def correctness_reward(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    """Returns 0.0 or 1.0 based on answer accuracy.
    'answer' param name MUST match dataset column name."""
    ...
```

### Self-Test

```bash
python3 rewards.py
# Should show:
# Good XML: 1.00, No XML: 0.00, Correct: 1.0, Incorrect: 0.0
```

### Format Reward Scoring (for XML pattern)

| Check | Score | Purpose |
|-------|-------|---------|
| Opening + closing think tags | 0.15 | Tags exist |
| Think content non-empty | 0.15 | Has reasoning |
| Opening + closing answer tags | 0.20 | Tags exist |
| Answer content non-empty | 0.20 | Has final answer |
| Think before answer (order) | 0.15 | Correct structure |
| No extra text outside tags | 0.15 | Clean output |
| **Total** | **1.00** | |

---

## Step 4: Write Training Script

Create `train_grpo.py`. Copy from `templates/train_grpo.py` and customize:

### What to Customize

**A. Dataset loading** — Change `load_dataset("gsm8k", "main")` to your dataset. Write a `format_example()` that:
- Creates a `prompt` column using the model's chat template
- Creates columns matching your reward function kwargs (e.g., `answer`)

**B. System prompt** — Change to describe your task and output format.

**C. LoRA target modules** — For Qwen/Llama models: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

**D. Reward functions list** — Change `reward_funcs=[format_reward, correctness_reward]` to your functions.

### VRAM-Based Config Selection

Before training, check free VRAM and select config:

| Free VRAM | batch | gen | accum | max_len | Notes |
|-----------|-------|-----|-------|---------|-------|
| 4-8 GB | 1 | 2 | 1 | 256 | 0.5B model only |
| 8-20 GB | 2 | 4 | 2 | 384 | 0.5B-1.5B model |
| 20-40 GB | 4 | 8 | 4 | 512 | 1.5B-3B model |
| 40+ GB | 4 | 8 | 4 | 512 | 7B+ model with LoRA |

Always use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

---

## Step 5: Dry-Run (CRITICAL — Do Not Skip)

Run 2 steps to verify the entire pipeline works before committing hours to training:

```bash
cd /data/grpo-finetune

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_ENDPOINT=https://hf-mirror.com \
python3 train_grpo.py \
    --max_steps 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_generations 4 \
    --max_completion_length 256 \
    --eval_strategy no \
    --report_to none \
    2>&1 | tee dry_run.log
```

**Expected output:**
```
100%|██████████| 2/2
{'rewards/format_reward/mean': '0.4...', 'rewards/correctness_reward/mean': '0.0', 'kl': '0'}
```

**If it fails:** See [references/troubleshooting.md](references/troubleshooting.md). Common first-run errors:
1. `got an unexpected keyword argument 'max_prompt_length'` → Remove it from GRPOConfig
2. `generation_batch_size must be divisible by num_generations` → Set `generation_batch_size=num_generations`
3. `report_to=None is not supported` → Use `report_to="none"` (string)
4. `CUDA out of memory` → Reduce batch/gen/max_completion_length
5. Import timeout → Set `VLLM_AVAILABLE=0`, wait 120s

**Do not proceed to Step 6 until dry-run completes with exit code 0.**

---

## Step 6: Full Training

```bash
cd /data/grpo-finetune

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_ENDPOINT=https://hf-mirror.com \
python3 train_grpo.py \
    --max_steps 500 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --num_generations 4 \
    --max_completion_length 384 \
    --learning_rate 5e-6 \
    --beta 0.04 \
    --temperature 0.9 \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_strategy no \
    --output_dir ./outputs/run_1 \
    --report_to none \
    2>&1 | tee training.log
```

Run in background if the agent supports it. Monitor progress:

```bash
# Check current step
grep -oP "\d+%\|[^|]+\| \d+/\d+" training.log | tail -1

# Check latest metrics
grep "'reward'" training.log | tail -1
```

### What to Watch

| Metric | Healthy | Unhealthy |
|--------|---------|-----------|
| `rewards/format_reward/mean` | Rising 0.4→0.9+ | Stuck below 0.3 after 100 steps |
| `rewards/correctness_reward/mean` | Gradual upward trend | Always 0 after 200 steps |
| `kl` | Under 0.05 | Spiking above 0.1 |
| `entropy` | Gradual decrease | Rapid drop to 0 (mode collapse) |
| `loss` | Near zero | Spiking or NaN |

### Training Duration Estimates

| Model | Steps | A100 | 4090 | V100 |
|-------|-------|------|------|------|
| 0.5B | 500 | ~1.5h | ~2h | ~4h |
| 1.5B | 500 | ~3h | ~5h | ~8h |
| 3B | 500 | ~6h | ~10h | N/A |

---

## Step 7: Evaluate

Create `evaluate.py` from `templates/evaluate.py`. Run after training completes:

```bash
python3 evaluate.py \
    --model_path ./outputs/run_1/final_model \
    --num_samples 200 \
    --temperature 0.6 \
    --output_file eval_results.json \
    2>&1 | tee eval.log
```

### Key Metrics to Report

| Metric | How to Calculate |
|--------|-----------------|
| Format Score | Average of `format_reward()` over test set |
| Perfect Format % | Fraction with format_score >= 0.99 |
| Accuracy | Fraction of correct answers |
| Sample outputs | Show 3-5 examples (correct and incorrect) |

### Interpreting Results

- **High format + low accuracy**: Model learned format but not task. Increase steps, try larger model, or add process rewards.
- **Low format**: Format reward too hard or contradicting correctness reward. Check reward function self-test.
- **High KL (>0.1)**: Model drifting too far. Increase `beta` (KL penalty).
- **Mode collapse (entropy→0)**: Reduce `beta`, increase `temperature`, or add entropy bonus.

---

## Files in This Skill

| File | Purpose |
|------|---------|
| `templates/rewards.py` | Reward function template (format + correctness) with self-test |
| `templates/train_grpo.py` | Complete training script with LoRA, chat template, configurable args |
| `templates/evaluate.py` | Evaluation script: load model, generate, score, save JSON results |
| `references/troubleshooting.md` | All known errors and fixes organized by training phase |
| `references/training-curves.md` | Real training data for expected behavior reference |
| `references/reward-patterns.md` | Reward function design patterns and edge cases |

---

## Quick Reference: GRPOConfig Parameters (TRL 0.29.1)

```python
GRPOConfig(
    output_dir="./outputs/run_1",
    # --- Training ---
    max_steps=500,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_steps=10,                    # NOT warmup_ratio (deprecated)
    bf16=True,
    gradient_checkpointing=True,
    # --- GRPO ---
    num_generations=4,                  # Completions per prompt
    num_generations_eval=1,             # 1 for fast eval
    generation_batch_size=4,            # MUST == num_generations
    max_completion_length=384,
    beta=0.04,                          # KL penalty (higher = more conservative)
    temperature=0.9,                    # Sampling temperature
    use_vllm=False,                     # Avoid version conflicts
    scale_rewards="group",              # Normalize within group
    loss_type="grpo",
    generation_kwargs={"do_sample": True, "top_p": 0.95},
    # --- Logging ---
    logging_steps=10,
    report_to="none",                   # STRING, not None
    save_steps=100,
    save_total_limit=3,
    # --- Dataset ---
    remove_unused_columns=False,
    dataloader_num_workers=0,
)
```

## Quick Reference: GRPOTrainer API (TRL 0.29.1)

```python
trainer = GRPOTrainer(
    model=model,                        # Pre-loaded model (NOT model name string if using peft_config)
    reward_funcs=[fn1, fn2],            # List of reward callables
    args=GRPOConfig(...),
    train_dataset=train_ds,             # Must have 'prompt' column
    eval_dataset=test_ds,
    processing_class=tokenizer,         # NOT 'tokenizer='
    peft_config=lora_config,            # Pass LoRA config directly, don't wrap model
)
```

## Verification Checklist

After completing all steps, verify:

- [ ] `rewards.py` self-test passes with correct scores
- [ ] Dry-run (2 steps) completes without errors
- [ ] Full training runs to completion (exit code 0)
- [ ] `final_model/` directory exists with model weights
- [ ] Evaluation produces accuracy + format scores
- [ ] Sample outputs show the model using the target format
- [ ] Training log shows format reward rising and KL staying low
