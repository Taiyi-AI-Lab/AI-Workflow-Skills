# Reward Function Design Patterns for GRPO

## Core Principles

1. **Smooth gradients**: Use multi-subscore rewards (0.0–1.0) instead of binary (0/1)
2. **Parameter names = dataset columns**: `def reward(completions, answer, **kwargs)` — `answer` must match column
3. **Return `list[float]`**: One score per completion, same length as input
4. **Self-test always**: Include `if __name__ == "__main__"` with edge cases

---

## Pattern 1: Format + Correctness (Most Common)

Best for teaching a model to output in a specific format while also solving a task.

```python
trainer = GRPOTrainer(
    reward_funcs=[format_reward, correctness_reward],
    ...
)
```

### Format Reward (smooth, 0.0–1.0)

Break into sub-scores that sum to 1.0:

```python
def format_reward(completions: list[str], **kwargs) -> list[float]:
    rewards = []
    for text in completions:
        score = 0.0
        if has_both_tags(text):      score += 0.15  # Tags exist
        if content_nonempty(text):    score += 0.15  # Has content
        if has_answer_tags(text):     score += 0.20  # Answer tags
        if answer_nonempty(text):     score += 0.20  # Has answer
        if correct_order(text):       score += 0.15  # Ordering
        if clean_structure(text):     score += 0.15  # No extra text
        rewards.append(score)
    return rewards
```

### Correctness Reward (sparse, 0.0 or 1.0)

```python
def correctness_reward(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    rewards = []
    for completion, gt in zip(completions, answer):
        extracted = extract_answer(completion)
        if extracted is None:
            rewards.append(0.0)
            continue
        pred = normalize(extracted)
        truth = normalize(gt)
        rewards.append(1.0 if pred == truth else 0.0)
    return rewards
```

---

## Pattern 2: Code Generation (Format + Tests)

For code generation tasks:

```python
def code_format_reward(completions, **kwargs):
    """Check for markdown code blocks with correct language tag."""
    scores = []
    for c in completions:
        score = 0.0
        if "```python" in c:     score += 0.3
        if "```" in c.split("```python")[-1]:  score += 0.3  # Closing
        if "def " in c:           score += 0.2  # Has function
        if "return" in c:         score += 0.2  # Has return
        scores.append(score)
    return scores

def code_correctness_reward(completions, test_cases, **kwargs):
    """Execute code against test cases."""
    scores = []
    for completion, tests in zip(completions, test_cases):
        code = extract_code(completion)
        if code is None:
            scores.append(0.0)
            continue
        passed = 0
        for inp, expected in eval(tests):
            try:
                result = exec_code(code, inp)
                if result == expected:
                    passed += 1
            except:
                pass
        scores.append(passed / len(eval(tests)))
    return scores
```

---

## Pattern 3: Single Reward (Simplest)

For when you only care about correctness:

```python
def accuracy_reward(completions, answer, **kwargs):
    scores = []
    for completion, gt in zip(completions, answer):
        # Simple string contains check
        if str(gt) in completion:
            scores.append(1.0)
        else:
            scores.append(0.0)
    return scores
```

---

## Edge Cases & Bugs

### The rfind(-1) Slicing Bug

```python
# BUG: rfind returns -1 when tag not found
# text[-1 + len("</answer>"):] = text[8:] — slices from middle!
post_end = text.rfind("</answer>")
tail = text[post_end + len("</answer>"):].strip()

# FIX: always check for -1
post_end = text.rfind("</answer>")
if post_end != -1:
    tail = text[post_end + len("</answer>"):].strip()
    if tail:
        clean = False
```

### Empty String Counts as "Content"

```python
# BUG: strip() of whitespace is empty, but len() > 0
if extract_answer(text) is not None:
    # This passes even for "  " content
    
# FIX: explicitly check stripped length
content = extract_answer(text)
if content is not None and len(content.strip()) > 0:
    score += 0.20
```

### Number Normalization Pitfalls

```python
# These should all match: 1000 == "1,000" == "$1,000" == "1000.0"
# Handle: commas, currency, percentages, fractions, decimals

def normalize_number(s):
    s = s.strip()
    s = re.sub(r"(\d),(\d)", r"\1\2", s)  # Remove commas
    s = s.replace("$", "")                  # Remove $
    if s.endswith("%"): s = s[:-1]          # Remove %
    try:
        num = float(s)
        return str(int(num)) if num == int(num) else f"{num:.6f}".rstrip("0")
    except:
        return None
```

---

## Design Checklist

- [ ] Format reward has sub-scores summing to 1.0
- [ ] Correctness reward parameter name matches dataset column
- [ ] All edge cases handled (missing tags, empty content, wrong order)
- [ ] Self-test covers: perfect output, no format, partial format, correct/incorrect
- [ ] No rfind(-1) slicing bugs
- [ ] normalize() handles all expected formats (commas, currency, etc.)
