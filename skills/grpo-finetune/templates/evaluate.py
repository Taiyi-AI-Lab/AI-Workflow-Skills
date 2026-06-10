#!/usr/bin/env python3
"""
Evaluate a GRPO-trained model on a test set.

Measures:
  - Accuracy (answer correctness)
  - Format score (XML structure compliance)
  - Perfect format percentage
  - Per-sample results saved to JSON

Usage:
    python evaluate.py --model_path ./outputs/run_1/final_model --num_samples 200
"""

import os
import sys
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from rewards import format_reward, correctness_reward, extract_answer

# Same system prompt as training
SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the following math problem step by step. "
    "Put your reasoning inside <think_box>...</think_box> tags and your final answer "
    "inside <answer>...</answer> tags. "
    "Output ONLY the XML, with no other text before or after."
)


def load_trained_model(model_path: str, base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Load trained model. Tries full model first, falls back to base+LoRA."""
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Loaded as full model")
    except Exception:
        print("Loading as base + LoRA adapter...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
        print("Merging LoRA weights...")
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def evaluate(
    model_path: str,
    dataset_name: str = "gsm8k",
    num_samples: int = 200,
    output_file: str = "./eval_results.json",
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    temperature: float = 0.6,      # Lower than training temperature
    max_new_tokens: int = 512,
):
    """Evaluate model on test set."""
    model, tokenizer = load_trained_model(model_path, base_model)

    # Load test dataset
    print(f"Loading {dataset_name} test set...")
    dataset = load_dataset(dataset_name, "main")["test"]

    if num_samples > 0:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    results = []
    correct = 0
    format_scores = []
    perfect_format = 0

    for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
        question = example["question"]
        # Customize answer extraction for your dataset:
        ground_truth = example["answer"].split("####")[-1].strip()

        # Format prompt with chat template
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem: {question}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # Compute rewards
        fmt_score = format_reward([completion])[0]
        corr_score = correctness_reward([completion], answer=[ground_truth])[0]

        format_scores.append(fmt_score)
        if corr_score > 0.5:
            correct += 1
        if fmt_score >= 0.99:
            perfect_format += 1

        extracted_answer = extract_answer(completion) or "N/A"

        results.append({
            "id": i,
            "question": question,
            "ground_truth": ground_truth,
            "completion": completion,
            "extracted_answer": extracted_answer,
            "format_score": fmt_score,
            "correct": bool(corr_score > 0.5),
        })

        # Progress report every 50 samples
        if (i + 1) % 50 == 0:
            acc = correct / (i + 1) * 100
            avg_fmt = sum(format_scores) / len(format_scores)
            print(
                f"  [{i+1}/{len(dataset)}] Acc: {acc:.1f}% | "
                f"Avg Format: {avg_fmt:.3f} | Perfect Format: {perfect_format}/{i+1}"
            )

    # Final metrics
    accuracy = correct / len(results) * 100
    avg_format = sum(format_scores) / len(format_scores)

    print(f"\n{'='*60}")
    print(f"Evaluation Results ({len(results)} samples)")
    print(f"{'='*60}")
    print(f"Accuracy:          {accuracy:.2f}%")
    print(f"Avg Format Score:  {avg_format:.4f}")
    print(f"Perfect Format:    {perfect_format}/{len(results)} ({perfect_format/len(results)*100:.1f}%)")
    print(f"Correct:           {correct}/{len(results)}")

    # Save detailed results
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "model_path": model_path,
            "dataset": dataset_name,
            "num_samples": len(results),
            "accuracy": accuracy,
            "avg_format_score": avg_format,
            "perfect_format": perfect_format,
            "perfect_format_pct": perfect_format / len(results) * 100,
            "correct": correct,
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {output_file}")

    # Show sample outputs
    correct_samples = [r for r in results if r["correct"]]
    incorrect_samples = [r for r in results if not r["correct"]]

    if correct_samples:
        print(f"\n{'='*60}")
        print("Sample Correct Generations")
        print(f"{'='*60}")
        for s in correct_samples[:3]:
            print(f"\nQ: {s['question'][:100]}...")
            print(f"GT: {s['ground_truth']} | Pred: {s['extracted_answer']} | Fmt: {s['format_score']:.2f}")
            print(f"Completion: {s['completion'][:250]}...")

    if incorrect_samples:
        print(f"\n{'='*60}")
        print("Sample Incorrect Generations")
        print(f"{'='*60}")
        for s in incorrect_samples[:3]:
            print(f"\nQ: {s['question'][:100]}...")
            print(f"GT: {s['ground_truth']} | Pred: {s['extracted_answer']} | Fmt: {s['format_score']:.2f}")
            print(f"Completion: {s['completion'][:250]}...")

    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Evaluate GRPO-trained model")
    p.add_argument("--model_path", default="./outputs/run_1/final_model")
    p.add_argument("--dataset_name", default="gsm8k")
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--output_file", default="./eval_results.json")
    p.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--temperature", type=float, default=0.6)
    args = p.parse_args()

    evaluate(
        args.model_path,
        args.dataset_name,
        args.num_samples,
        args.output_file,
        args.base_model,
        args.temperature,
    )
