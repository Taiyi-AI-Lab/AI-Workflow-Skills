# GRPO Training Troubleshooting Guide

All known errors organized by training phase. Based on real debugging sessions with TRL 0.29.1.

---

## Phase 1: Import & Setup

### TRL import hangs or times out

**Symptom:** `from trl import GRPOTrainer` hangs for >30 seconds.
**Cause:** TRL lazily loads vLLM, which has version incompatibilities.
**Fix:**
```python
import os
os.environ["VLLM_AVAILABLE"] = "0"
from trl import GRPOTrainer, GRPOConfig
```
Give it up to 120 seconds on first import.

### vLLM version mismatch warning

**Symptom:** `vLLM 0.19.1 is not compatible with TRL (expected 0.10-0.12)`
**Fix:** Set `use_vllm=False` in GRPOConfig and `VLLM_AVAILABLE=0` env var. HF generate works fine for small models.

### HuggingFace Hub unreachable

**Symptom:** `ConnectionError` when loading model or dataset.
**Fix:**
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
Qwen models may auto-download via ModelScope in some regions.

### GSM8K dataset not found

**Symptom:** `Dataset 'openai/gsm8k' not found`
**Fix:** Use `load_dataset("gsm8k", "main")` (without the `openai/` prefix).

---

## Phase 2: GRPOConfig Errors

### `got an unexpected keyword argument 'max_prompt_length'`

**Cause:** `max_prompt_length` is NOT a GRPOConfig parameter in TRL 0.29+.
**Fix:** Remove it. Prompt length is controlled by the model's max position embeddings.

### `got an unexpected keyword argument 'tokenizer'`

**Cause:** GRPOTrainer uses `processing_class`, not `tokenizer`.
**Fix:**
```python
# WRONG
trainer = GRPOTrainer(tokenizer=tokenizer, ...)

# CORRECT
trainer = GRPOTrainer(processing_class=tokenizer, ...)
```

### `None is not supported` for report_to

**Cause:** `report_to=None` crashes. Must be a string.
**Fix:** `report_to="none"` (string, not None)

### `warmup_ratio is deprecated`

**Cause:** Transformers 5.x deprecated `warmup_ratio`.
**Fix:** Use `warmup_steps=10` instead.

### `generation_batch_size must be divisible by num_generations`

**Cause:** generation_batch_size must be an exact multiple.
**Fix:** Set `generation_batch_size=num_generations` (same value).

### `eval batch size must be divisible by num_generations_eval`

**Cause:** Same divisibility rule for evaluation.
**Fix:** Set `num_generations_eval=1`.

---

## Phase 3: OOM (Out of Memory)

### Training OOM at step 3-5

**Cause:** GPU has insufficient free VRAM for the batch + generation size.
**Fix 1:** Reduce parameters:
```
per_device_train_batch_size: 4 → 2 → 1
num_generations: 8 → 4 → 2
gradient_accumulation_steps: 4 → 2 → 1
max_completion_length: 512 → 384 → 256
```

**Fix 2:** Set environment variable:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_grpo.py ...
```

**Fix 3:** Check what's using VRAM:
```bash
python3 -c "import torch; print(f'Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB')"
nvidia-smi
```

**VRAM budget for 0.5B model (bf16 + LoRA):**
| Config | VRAM needed |
|--------|-------------|
| batch=1, gen=2 | ~2 GB |
| batch=2, gen=4 | ~4 GB |
| batch=4, gen=8 | ~8 GB |

### First run OOMs but second run works

**Cause:** Zombie GPU allocations from previous failed runs.
**Fix:** Kill stale processes, or just retry — PyTorch may reclaim memory.

---

## Phase 4: Training Issues

### Format reward stuck below 0.3

**Cause:** Reward function may be too strict or have a bug.
**Fix:**
1. Run `python3 rewards.py` self-test
2. Check that `rfind()` doesn't return -1 (the slicing bug):
   ```python
   # BUG: rfind returns -1, text[-1+9:] slices wrong
   post_end = text.rfind("</answer>")
   # FIX: check for -1
   if post_end != -1:
       tail = text[post_end + len("</answer>"):].strip()
   ```
3. Ensure at least some sub-scores are achievable early (not all-or-nothing)

### Correctness reward always 0

**Cause:** Parameter name mismatch between reward function and dataset column.
**Fix:** If dataset has column `"answer"`, the reward function MUST have parameter `answer`:
```python
# Dataset: {"prompt": ..., "answer": "72"}
# Reward function parameter name must match:
def correctness_reward(completions, answer, **kwargs):  # 'answer' matches column name
```

### KL divergence spiking above 0.1

**Cause:** Model diverging too fast from reference.
**Fix:** Increase `beta` (KL penalty): `0.04 → 0.1 → 0.2`

### Entropy drops to near 0 (mode collapse)

**Cause:** Model stops exploring, outputs the same thing every time.
**Fix:** Increase `temperature`: `0.9 → 1.0 → 1.2`, or decrease `beta`.

### Loss is NaN

**Cause:** Learning rate too high, or reward scaling issue.
**Fix:**
1. Reduce learning rate: `5e-6 → 1e-6`
2. Check reward values aren't extreme (should be 0–1 range)
3. Try `scale_rewards="group"` (normalizes within each group)

---

## Phase 5: Evaluation Issues

### Evaluation takes forever

**Cause:** Running eval during training with full test set and multiple generations.
**Fix:**
1. Use `eval_strategy="no"` during training
2. Run separate evaluation script with `num_generations_eval=1`
3. Limit eval samples: `--eval_samples 100`

### Test accuracy much lower than training reward

**Cause:** Normal — GRPO training reward is biased (model sees same prompts). Also temperature difference.
**Fix:** This is expected. 37.5% training → 18.5% test is typical. To narrow the gap:
1. Increase training steps
2. Use larger model
3. Use `temperature=0.6` for evaluation (lower than training)

### Can't load trained model

**Cause:** Saved as LoRA adapter, not full model.
**Fix:** Evaluation script handles this automatically — tries full model first, falls back to base+LoRA.

---

## Phase 6: Output Buffering

### Can't see training logs in real time

**Cause:** Piping to `tail` buffers output until process exits.
**Fix:**
```bash
# WRONG — buffered
python train_grpo.py ... 2>&1 | tail -80

# CORRECT — real-time
python train_grpo.py ... 2>&1 | tee training.log
```
