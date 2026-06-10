# AI Workflow Skills

面向 AI 训练与微调工作流的可复用 Agent Skills 仓库。

[English](./README.md) · [中文文档](./README.zh-CN.md) · [日本語](./README.ja-JP.md)

## 安装

从下方链接下载已发布的 skill zip，然后解压到你的 agent skill 目录：

```bash
unzip grpo-finetune-2.0.0.zip -d ~/.codex/skills/
unzip mint-lora-training-1.0.0.zip -d ~/.codex/skills/
```

如果你的 Claude Code 环境支持 plugin marketplace，也可以把本仓库作为 marketplace source 添加：

```text
/plugin marketplace add Taiyi-AI-Lab/AI-Workflow-Skills
```

## Skills

### [`grpo-finetune`](./skills/grpo-finetune)

**类别：** AI Training / Fine-Tuning

**适合：** 在本地 CUDA 环境执行端到端 GRPO 微调流程，包括环境检查、奖励函数、训练脚本模板、dry run、正式训练、评估和排障。

亮点：

- 7 步 GRPO 工作流，从需求确认到最终评估
- 内置 rewards、training、evaluation 的可运行模板
- 覆盖 TRL、vLLM、CUDA、OOM、batching 等版本敏感问题
- 包含奖励设计、训练曲线分析和故障排查参考文档

链接：[README](./skills/grpo-finetune/README.md) · [SKILL.md](./skills/grpo-finetune/SKILL.md) · <!-- DOWNLOAD:grpo-finetune:start -->[下载 v2.0.0 .zip](https://github.com/Taiyi-AI-Lab/AI-Workflow-Skills/releases/download/grpo-finetune-v2.0.0/grpo-finetune-2.0.0.zip)<!-- DOWNLOAD:grpo-finetune:end -->

### [`mint-lora-training`](./skills/mint-lora-training)

**类别：** AI Training / Fine-Tuning

**适合：** 在 macaron.im 上运行云端 MinT LoRA 微调，不依赖本地 GPU；覆盖 SFT、DPO、GRPO、评估、adapter 下载和 token 额度检查。

亮点：

- 明确区分 MinT 云端训练与本地脚本职责
- 内置 setup、SFT、DPO、GRPO、评估、下载和 quota 检查脚本
- 提供首次运行、监督微调、偏好微调和 RL 工作流
- 包含支持模型、endpoint、数据格式和常见坑参考

链接：[README](./skills/mint-lora-training/README.md) · [SKILL.md](./skills/mint-lora-training/SKILL.md) · <!-- DOWNLOAD:mint-lora-training:start -->_（暂未发布）_<!-- DOWNLOAD:mint-lora-training:end -->

## 维护命令

```bash
npm run list
npm run pack -- --skill grpo-finetune
npm run pack -- --skill mint-lora-training
npm run validate
```

`npm run validate` 会检查 manifest、构建 release zip、校验 README 下载标记、编译 Python 脚本，并扫描密钥或本机私有路径。

## 发布

单个 skill 的发布 tag 使用这个格式：

```text
grpo-finetune-v2.0.0
mint-lora-training-v1.0.0
```

发布前先运行 `npm run release:dry`。tag 触发的 GitHub Action 会校验 manifest 版本、构建 zip 和 sha256、创建 GitHub Release，并同步 README 下载链接。

## 目录结构

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
