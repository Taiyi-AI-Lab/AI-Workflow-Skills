#!/usr/bin/env python3
"""
Reward functions for GRPO training — format + correctness pattern.

CUSTOMIZE:
  A. Change XML tags (think/answer) to your desired format
  B. Change extract_answer() for your answer format
  C. Change normalize_answer() for your comparison logic

Self-test: python3 rewards.py
"""

import re


# ── A. Customize: Output format tags ──────────────────────────────────
THINK_OPEN = "OPSIS"
THINK_CLOSE = ""
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the following math problem step by step. "
    f"Put your reasoning inside {THINK_OPEN}...{THINK_CLOSE} tags and your final answer "
    f"inside {ANSWER_OPEN}...{ANSWER_CLOSE} tags. "
    "Output ONLY the XML, with no other text before or after."
)


# ── B. Customize: Answer extraction ───────────────────────────────────
def extract_answer(text: str) -> str | None:
    """Extract the content inside the last <answer>...</answer> tags."""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_think(text: str) -> str | None:
    """Extract the content inside the last <think_box>...</think_box> tags."""
    matches = re.findall(r"OPSIS(.*?)", text, re.DOTALL)
    return matches[-1].strip() if matches else None


# ── C. Customize: Answer comparison ───────────────────────────────────
def normalize_answer(s: str) -> str | None:
    """Normalize a numeric string for comparison.
    Handles commas, dollar signs, percentages, fractions, and whitespace.
    Returns None if the string cannot be parsed as a number.
    """
    if s is None:
        return None

    s = s.strip()

    # Remove commas from numbers: 1,000 -> 1000
    s = re.sub(r"(\d),(\d)", r"\1\2", s)

    # Handle dollar signs: $5.00 -> 5.00
    s = s.replace("$", "")

    # Handle percentage: 50% -> 50
    if s.endswith("%"):
        s = s[:-1]

    # Try to parse as float
    try:
        num = float(s)
        if num == int(num):
            return str(int(num))
        return f"{num:.6f}".rstrip("0").rstrip(".")
    except ValueError:
        pass

    # Try to evaluate simple fractions: "1/2" -> 0.5
    try:
        if "/" in s and all(c in "0123456789/." for c in s):
            parts = s.split("/")
            if len(parts) == 2:
                num = float(parts[0]) / float(parts[1])
                if num == int(num):
                    return str(int(num))
                return f"{num:.6f}".rstrip("0").rstrip(".")
    except (ValueError, ZeroDivisionError):
        pass

    return None


# ── Reward functions ──────────────────────────────────────────────────
def format_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward for XML format compliance (0.0–1.0).

    Sub-scores:
      0.15 — think tags present and closed
      0.15 — think content non-empty
      0.20 — answer tags present and closed
      0.20 — answer content non-empty
      0.15 — correct ordering (think before answer)
      0.15 — clean structure (no text outside tags)
    """
    rewards = []
    for completion in completions:
        score = 0.0

        # Check for think tags
        has_think_open = THINK_OPEN in completion
        has_think_close = THINK_CLOSE in completion
        think_content = extract_think(completion)
        has_nonempty_think = think_content is not None and len(think_content.strip()) > 0

        # Check for answer tags
        has_answer_open = ANSWER_OPEN in completion
        has_answer_close = ANSWER_CLOSE in completion
        answer_content = extract_answer(completion)
        has_nonempty_answer = answer_content is not None and len(answer_content.strip()) > 0

        # Sub-scores
        if has_think_open and has_think_close:
            score += 0.15
        if has_nonempty_think:
            score += 0.15
        if has_answer_open and has_answer_close:
            score += 0.20
        if has_nonempty_answer:
            score += 0.20

        # Bonus for correct ordering (think before answer)
        think_pos = completion.find(THINK_OPEN)
        answer_pos = completion.find(ANSWER_OPEN)
        if think_pos != -1 and answer_pos != -1 and think_pos < answer_pos:
            score += 0.15

        # Bonus for clean structure (no extra text outside tags)
        post_end = completion.rfind(ANSWER_CLOSE)
        has_any_tag = has_think_open or has_answer_open
        clean_structure = False
        if has_any_tag:
            clean_structure = True
            if think_pos != -1:
                pre_think = completion[:think_pos].strip()
                if pre_think:
                    clean_structure = False
            if post_end != -1:
                post_answer = completion[post_end + len(ANSWER_CLOSE):].strip()
                if post_answer:
                    clean_structure = False
        if clean_structure:
            score += 0.15

        rewards.append(score)

    return rewards


def correctness_reward(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    """Reward for answer correctness (0.0 or 1.0).

    IMPORTANT: Parameter name 'answer' MUST match dataset column name.
    TRL passes all non-'prompt' dataset columns as kwargs.
    """
    rewards = []
    for completion, ground_truth in zip(completions, answer):
        extracted = extract_answer(completion)
        if extracted is None:
            rewards.append(0.0)
            continue

        # Normalize both for comparison
        pred_norm = normalize_answer(extracted)
        gt_norm = normalize_answer(str(ground_truth))

        if pred_norm is None or gt_norm is None:
            rewards.append(0.0)
            continue

        # Exact string match after normalization
        rewards.append(1.0 if pred_norm == gt_norm else 0.0)

    return rewards


# ── Self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    good = "OPSISLet me solve this step by step.\n5 + 3 = 8\n<answer>8</answer>"
    bad = "The answer is 8"
    no_answer = "OPSISLet me think...\n"

    scores = format_reward([good, bad, no_answer])
    print(f"Good XML: {scores[0]:.2f} (expected ~0.85-1.00)")
    print(f"No XML:   {scores[1]:.2f} (expected ~0.00)")
    print(f"No answer: {scores[2]:.2f} (expected ~0.30)")

    results = correctness_reward(
        completions=["OPSISx\n<answer>8</answer>", "OPSISx\n<answer>9</answer>"],
        answer=["8", "8"],
    )
    print(f"Correct:   {results[0]} (expected 1.0)")
    print(f"Incorrect: {results[1]} (expected 0.0)")

    # Test normalization
    tests = [
        ("1,000", "1000", True),
        ("$5.00", "5", True),
        ("50%", "50", True),
        ("1/2", "0.5", True),
        ("3.14", "3.14", True),
    ]
    for a, b, expected in tests:
        result = normalize_answer(a) == normalize_answer(b)
        status = "OK" if result == expected else "FAIL"
        print(f"{status}: normalize('{a}') == normalize('{b}'): {result}")
