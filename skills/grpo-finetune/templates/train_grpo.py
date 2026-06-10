#!/usr/bin/env python3
"""
GRPO Training Script — template for any model + dataset + format.

Customize:
  A. load_datasets() — change dataset name and format_example()
  B. SYSTEM_PROMPT — change to describe your task
  C. reward_funcs list — change to your reward functions

Usage:
    # Dry run (2 steps)
    python train_grpo.py --max_steps 2 --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 --num_generations 4 \
        --max_completion_length 256 --eval_strategy no --report_to none

    # Full run
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_grpo.py \
        --max_steps 500 --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 2 --num_generations 4 \
        --max_completion_length 384 --eval_strategy no --report_to none \
        2>&1 | tee training.log
"""

import os
import sys
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import GRPOTrainer, GRPOConfig

# Suppress vLLM (use HF generate instead)
os.environ["VLLM_AVAILABLE"] = "0"

sys.path.insert(0, os.path.dirname(__file__))
from rewards import format_reward, correctness_reward  # <-- your reward module

# ── A. Customize: System prompt ───────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the following math problem step by step. "
    "Put your reasoning inside <think_box>...</think_box> tags and your final answer "
    "inside <answer>...</answer> tags. "
    "Output ONLY the XML, with no other text before or after."
)


def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    return model, tokenizer


# ── B. Customize: Dataset loading ─────────────────────────────────────
def format_example(example, tokenizer):
    """Format a dataset example into prompt + answer columns.

    REQUIRED output columns:
      - 'prompt': formatted text input (using chat template)
      - 'answer': ground truth answer (passed to correctness_reward as kwarg)

    Customize the question extraction and answer parsing for your dataset.
    """
    question = example["question"]
    # GSM8K: answer ends with "#### <number>"
    # Change this line for other datasets:
    answer = example["answer"].split("####")[-1].strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem: {question}"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt, "answer": answer}


def load_datasets(tokenizer, dataset_name="gsm8k"):
    """Load and format dataset.

    Customize:
      - dataset_name: HF hub dataset ID
      - format_example: how to map raw data to prompt + answer
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, "main")

    train_dataset = dataset["train"].map(
        lambda x: format_example(x, tokenizer),
        remove_columns=dataset["train"].column_names,
    )
    test_dataset = dataset["test"].map(
        lambda x: format_example(x, tokenizer),
        remove_columns=dataset["test"].column_names,
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Test:  {len(test_dataset)} samples")

    # Show a sample
    print(f"\nSample prompt (first 300 chars):\n{train_dataset[0]['prompt'][:300]}...")
    print(f"Sample answer: {train_dataset[0]['answer']}")

    return train_dataset, test_dataset


def main():
    import argparse

    p = argparse.ArgumentParser(description="GRPO training")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset_name", default="gsm8k")
    p.add_argument("--output_dir", default="./outputs/run_1")
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--max_completion_length", type=int, default=384)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--report_to", default="none")
    p.add_argument("--loss_type", default="grpo",
                    choices=["grpo", "bnpo", "rloo", "reinforce", "dapo"])
    p.add_argument("--scale_rewards", default="group",
                    choices=["group", "batch", "none"])
    p.add_argument("--eval_samples", type=int, default=0,
                    help="Limit eval dataset to N samples (0=use all)")
    p.add_argument("--eval_strategy", default="no", choices=["no", "steps"])
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        # Standard LoRA targets for Qwen/Llama/Mistral:
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # Load datasets
    train_dataset, test_dataset = load_datasets(tokenizer, args.dataset_name)

    # Limit eval dataset if requested
    if args.eval_samples > 0 and args.eval_samples < len(test_dataset):
        test_dataset = test_dataset.select(range(args.eval_samples))
        print(f"Limited eval dataset to {args.eval_samples} samples")

    # Generation kwargs
    generation_kwargs = {
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": 0.95,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # GRPO configuration
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        run_name=f"grpo-{args.model_name.split('/')[-1]}-{args.dataset_name}",
        # Training
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=10,           # NOT warmup_ratio (deprecated in transformers 5.x)
        bf16=True,
        gradient_checkpointing=True,
        # GRPO specific
        num_generations=args.num_generations,
        num_generations_eval=1,     # 1 for fast evaluation
        generation_batch_size=args.num_generations,  # MUST == num_generations
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        temperature=args.temperature,
        use_vllm=False,            # Avoid vLLM version conflicts
        scale_rewards=args.scale_rewards,
        loss_type=args.loss_type,
        generation_kwargs=generation_kwargs,
        # Evaluation
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        # Logging
        logging_steps=args.logging_steps,
        report_to=args.report_to,   # STRING "none", NOT None
        # Checkpointing
        save_steps=args.save_steps,
        save_total_limit=3,
        save_strategy="steps",
        # Dataset
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    # Print config
    print(f"\n{'='*60}")
    print("GRPO Training Configuration")
    print(f"{'='*60}")
    print(f"  Model:              {args.model_name}")
    print(f"  Dataset:            {args.dataset_name}")
    print(f"  Output dir:         {args.output_dir}")
    print(f"  Generations/prompt: {args.num_generations}")
    print(f"  Max steps:          {args.max_steps}")
    print(f"  Learning rate:      {args.learning_rate}")
    print(f"  KL penalty (beta):  {args.beta}")
    print(f"  Temperature:        {args.temperature}")
    print(f"  Batch size:         {args.per_device_train_batch_size}")
    print(f"  Grad accum:         {args.gradient_accumulation_steps}")
    print(f"  LoRA rank:          {args.lora_r}")
    print(f"  Max completion len: {args.max_completion_length}")
    print(f"  Train samples:      {len(train_dataset)}")
    print(f"  Eval samples:       {len(test_dataset)}")

    # ── C. Customize: Reward functions ─────────────────────────────────
    # Change this list to your reward functions
    reward_functions = [format_reward, correctness_reward]

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,  # NOT 'tokenizer='
        peft_config=peft_config,
    )

    print(f"\nStarting GRPO training...")
    trainer.train()

    # Save final model
    final_output = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)
    print(f"\nTraining complete! Model saved to: {final_output}")

    # Save LoRA adapter separately
    lora_output = os.path.join(args.output_dir, "lora_adapter")
    trainer.model.save_pretrained(lora_output)
    print(f"LoRA adapter saved to: {lora_output}")


if __name__ == "__main__":
    main()
