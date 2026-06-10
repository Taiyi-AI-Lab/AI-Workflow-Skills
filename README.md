# AI Workflow Skills

Reusable agent skills for AI training and fine-tuning workflows.

[English](./README.md) · [中文文档](./README.zh-CN.md) · [日本語](./README.ja-JP.md)

## Install

Download a released skill zip from the links below, then extract it into your
agent's skill directory:

```bash
unzip grpo-finetune-2.0.0.zip -d ~/.codex/skills/
unzip mint-lora-training-1.0.0.zip -d ~/.codex/skills/
```

Claude Code users can also install this repository as a plugin marketplace
source when plugin marketplace support is available:

```text
/plugin marketplace add Taiyi-AI-Lab/AI-Workflow-Skills
```

## Skills

### [`grpo-finetune`](./skills/grpo-finetune)

**Category:** AI Training / Fine-Tuning

**Best for:** running an end-to-end local GRPO fine-tuning workflow for LLMs,
including environment checks, reward functions, training script templates, dry
runs, full training, evaluation, and troubleshooting.

Highlights:

- 7-step GRPO workflow from requirements gathering to final evaluation
- Ready-to-run templates for rewards, training, and evaluation
- Version-sensitive notes for TRL, vLLM, CUDA, OOM handling, and batching
- References for reward design, training-curve interpretation, and debugging

Links: [README](./skills/grpo-finetune/README.md) · [SKILL.md](./skills/grpo-finetune/SKILL.md) · <!-- DOWNLOAD:grpo-finetune:start -->_(no release yet — coming soon)_<!-- DOWNLOAD:grpo-finetune:end -->

### [`mint-lora-training`](./skills/mint-lora-training)

**Category:** AI Training / Fine-Tuning

**Best for:** running cloud-hosted MinT LoRA fine-tuning on macaron.im without
local GPU training, including SFT, DPO, GRPO, evaluation, adapter download, and
quota checks.

Highlights:

- Clear cloud-vs-local boundaries for MinT training
- Ready-to-run scripts for setup, SFT, DPO, GRPO, evaluation, download, and quota
- Workflow guides for first run, supervised fine-tuning, preference tuning, and RL
- Reference notes for supported models, endpoints, data formats, and known pitfalls

Links: [README](./skills/mint-lora-training/README.md) · [SKILL.md](./skills/mint-lora-training/SKILL.md) · <!-- DOWNLOAD:mint-lora-training:start -->_(no release yet — coming soon)_<!-- DOWNLOAD:mint-lora-training:end -->

## Maintainer Commands

```bash
npm run list
npm run pack -- --skill grpo-finetune
npm run pack -- --skill mint-lora-training
npm run validate
```

`npm run validate` checks manifests, builds release zips, verifies README
download markers, compiles bundled Python scripts, and scans for secrets or
private local paths.

## Release

Per-skill releases use this tag format:

```text
grpo-finetune-v2.0.0
mint-lora-training-v1.0.0
```

Use `npm run release:dry` before publishing. The tag-triggered GitHub Action
validates the manifest version, builds a zip and sha256 file, creates a GitHub
Release, and syncs the README download links.

## Layout

```text
skills/
├── grpo-finetune/
│   ├── README.md
│   ├── SKILL.md
│   ├── manifest.json
│   ├── references/
│   └── templates/
└── mint-lora-training/
    ├── README.md
    ├── SKILL.md
    ├── manifest.json
    ├── scripts/
    └── workflows/
```
