# grpo-finetune

End-to-end GRPO fine-tuning guidance for local CUDA environments.

## When to Use

Use this skill when you want an agent to help build a GRPO training workflow for
an LLM, including reward functions, TRL `GRPOTrainer` setup, LoRA configuration,
dry runs, full training, evaluation, and troubleshooting.

Do not use it for cloud-hosted MinT training. Use
[`mint-lora-training`](../mint-lora-training/README.md) for that workflow.

## Requirements

- Linux with an NVIDIA CUDA GPU
- Python 3.10+
- PyTorch with CUDA
- TRL, Transformers, PEFT, Datasets, and related Hugging Face tooling
- Access to the base model and dataset you plan to train on

macOS and Apple Silicon MPS are not supported for this local GRPO workflow.

## Quick Start

Install the skill into your agent's skills directory, then ask your agent to use
`grpo-finetune` for a specific model, dataset, and target output format.

Example request:

```text
Use grpo-finetune to train Qwen/Qwen2.5-0.5B-Instruct on GSM8K with XML
<think> and <answer> output.
```

The skill will guide the agent through:

1. collecting model, dataset, output format, and answer extraction requirements
2. checking OS, CUDA, and package versions
3. creating reward, training, and evaluation scripts from templates
4. running a two-step dry run before full training
5. evaluating final format compliance and task correctness

## Included Resources

- `templates/rewards.py`: format and correctness reward template
- `templates/train_grpo.py`: LoRA GRPO training template
- `templates/evaluate.py`: post-training evaluation template
- `references/troubleshooting.md`: common TRL, CUDA, vLLM, and OOM issues
- `references/training-curves.md`: expected training behavior reference
- `references/reward-patterns.md`: reward design patterns and pitfalls

## Safety Notes

Training jobs can consume substantial GPU time and storage. Start with the dry
run, keep checkpoints outside the repository, and do not commit models,
datasets, logs, or generated training outputs.
