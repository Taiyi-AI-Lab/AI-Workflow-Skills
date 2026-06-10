# Training Curve Reference — Qwen2.5-0.5B GRPO on GSM8K

Real training data from a successful run. Use this to judge whether your training is behaving normally.

## Config

- Model: Qwen2.5-0.5B-Instruct (0.5B params)
- Dataset: GSM8K (7473 train, 1319 test)
- Algorithm: GRPO with LoRA (r=16)
- Hardware: NVIDIA A100-SXM4-80GB
- Duration: 1h 43m (500 steps)

## Reward Progression

| Step | Format Reward | Correctness | KL Divergence | Entropy | Avg Completion Length |
|------|--------------|-------------|---------------|---------|----------------------|
| 10   | 0.42         | 5.0%        | 0.001         | 0.47    | 185                  |
| 50   | 0.70         | 2.5%        | 0.003         | 0.46    | 174                  |
| 90   | 0.80         | 17.5%       | 0.032         | 0.40    | 107                  |
| 150  | 0.88         | 17.5%       | 0.010         | 0.33    | 213                  |
| 180  | 0.93         | 25%         | —             | —       | —                    |
| 213  | 0.97         | 37.5%       | —             | —       | —                    |
| 250  | 0.96         | 27.5%       | —             | —       | —                    |
| 309  | 0.93         | 50%         | —             | —       | —                    |
| 360  | 0.83         | 22.5%       | 0.019         | 0.34    | 230                  |
| 370  | 0.98         | 20%         | 0.017         | 0.36    | 175                  |
| 420  | 0.95         | 52.5%       | 0.019         | 0.25    | 204                  |
| 430  | 0.996        | 22.5%       | 0.015         | 0.29    | 183                  |
| 450  | 1.00         | 27.5%       | 0.019         | 0.34    | 192                  |
| 460  | 0.97         | 47.5%       | 0.018         | 0.24    | 182                  |
| 480  | 0.95         | 15%         | 0.019         | 0.30    | 204                  |
| 490  | 0.91         | 27.5%       | 0.015         | 0.36    | 227                  |
| 500  | 0.989        | 37.5%       | 0.017         | 0.27    | 198                  |

## Evaluation Results (200 test samples, temperature=0.6)

| Metric | Value |
|--------|-------|
| Accuracy | 18.5% (37/200) |
| Avg Format Score | 0.9575 |
| Perfect XML Format | 88.5% (177/200) |

## Key Observations

1. **Format is easy to learn**: 0.42 → 0.99 in ~200 steps. XML structure converges fast.

2. **Correctness is noisy**: Fluctuates 2.5%–52.5% between logging steps. High variance is normal for GRPO — it's comparing within groups, not against an absolute standard.

3. **KL stays healthy**: Never exceeded 0.07, averaged ~0.017. The model doesn't drift far from base.

4. **Entropy decreases**: 0.47 → 0.27 means model becomes more confident/deterministic over time.

5. **Train-test gap**: 37.5% training reward vs 18.5% test accuracy is typical. Training reward is biased.

6. **Completion length varies**: Initially ~185 tokens, drops to ~65 (model learns conciseness), then rises to ~200 (model learns to reason more).

7. **Step time**: ~10-16 seconds, decreases as model learns shorter completions.

## What Healthy Training Looks Like

- Format reward: steady rise, stabilizes >0.9 by step 200
- Correctness: noisy but with upward trend
- KL: flat or slowly rising, never spiking
- Entropy: gradual decrease
- Loss: near zero (GRPO RL loss behaves differently from supervised)

## What Unhealthy Training Looks Like

- Format reward stuck <0.3 after 100 steps → reward function bug
- KL >0.1 and rising → increase beta
- Entropy → 0 in <50 steps → mode collapse, increase temperature
- Loss = NaN → reduce learning rate
- OOM at step 3-5 → reduce batch/gen/max_length
