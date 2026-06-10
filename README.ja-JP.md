# AI Workflow Skills

AI training and fine-tuning workflows 向けの再利用可能な Agent Skills です。

[English](./README.md) · [中文文档](./README.zh-CN.md) · [日本語](./README.ja-JP.md)

## Install

下のリンクからリリース済みの skill zip をダウンロードし、agent の skill
ディレクトリへ展開します。

```bash
unzip grpo-finetune-2.0.0.zip -d ~/.codex/skills/
unzip mint-lora-training-1.0.0.zip -d ~/.codex/skills/
```

Claude Code の plugin marketplace が利用できる環境では、このリポジトリを
marketplace source として追加できます。

```text
/plugin marketplace add Taiyi-AI-Lab/AI-Workflow-Skills
```

## Skills

### [`grpo-finetune`](./skills/grpo-finetune)

**Category:** AI Training / Fine-Tuning

**Best for:** local CUDA 環境での end-to-end GRPO fine-tuning。環境確認、reward
functions、training templates、dry run、本番 training、evaluation、troubleshooting
を含みます。

Highlights:

- Requirements gathering から evaluation までの 7-step GRPO workflow
- Rewards、training、evaluation の ready-to-run templates
- TRL、vLLM、CUDA、OOM、batching など version-sensitive な注意点
- Reward design、training curve、debugging の reference

Links: [README](./skills/grpo-finetune/README.md) · [SKILL.md](./skills/grpo-finetune/SKILL.md) · <!-- DOWNLOAD:grpo-finetune:start -->[Download v2.0.0 .zip](https://github.com/Taiyi-AI-Lab/AI-Workflow-Skills/releases/download/grpo-finetune-v2.0.0/grpo-finetune-2.0.0.zip)<!-- DOWNLOAD:grpo-finetune:end -->

### [`mint-lora-training`](./skills/mint-lora-training)

**Category:** AI Training / Fine-Tuning

**Best for:** local GPU を使わず macaron.im の cloud-hosted MinT LoRA training を
実行する用途。SFT、DPO、GRPO、evaluation、adapter download、quota check を含みます。

Highlights:

- MinT training の cloud/local 責務を明確化
- Setup、SFT、DPO、GRPO、evaluation、download、quota 用 scripts
- First run、supervised fine-tuning、preference tuning、RL の workflow guides
- Supported models、endpoints、data formats、known pitfalls の reference

Links: [README](./skills/mint-lora-training/README.md) · [SKILL.md](./skills/mint-lora-training/SKILL.md) · <!-- DOWNLOAD:mint-lora-training:start -->_（まだリリースされていません）_<!-- DOWNLOAD:mint-lora-training:end -->

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

公開前に `npm run release:dry` を実行してください。tag-triggered GitHub Action
が manifest version、zip、sha256、GitHub Release、README download links を処理します。

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
