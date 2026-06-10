# 贡献指南

感谢你改进 AI Workflow Skills。本仓库发布可复用的、兼容 SKILL.md 的 Agent Skills；
每个 skill 都独立维护版本并独立发布。

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

每个 skill 必须包含 `SKILL.md` 和 `manifest.json`。公开发布的 skill 还应包含简洁的 `README.md`，说明前置条件、安装方式和快速开始。

## 提交 PR 前

运行完整验证：

```bash
npm run validate
```

该命令会校验 manifest、构建 release zip、检查 README 下载标记、编译内置 Python 脚本，并扫描密钥或本机私有路径。

不要提交：

- API key 或 `.env` 文件
- 模型权重、adapter、checkpoint 或训练输出
- 本机绝对路径或私有工作区名称
- `dist/release/` 下的 release 产物

## 新增或更新 Skill

1. 创建或修改 `skills/<skill-name>/`。
2. 保持 `manifest.json#name`、`SKILL.md` frontmatter `name` 和目录名完全一致。
3. `manifest.json#version` 使用 SemVer。
4. 同步更新根 README 的英文、中文、日文条目。
5. 新 skill 需要添加 README 下载标记：
   `<!-- DOWNLOAD:<skill-name>:start --><!-- DOWNLOAD:<skill-name>:end -->`。
6. 运行 `npm run validate`。

## 发布 Tag

单个 skill 的发布 tag 使用这个格式：

```text
<skill-name>-v<semver>
```

示例：

```text
grpo-finetune-v2.0.0
mint-lora-training-v1.0.0
```

发布前先运行 `npm run release:dry`。发布 workflow 会校验对应的 `manifest.json` 版本，构建 `dist/release/` 下的 zip 和 `.sha256` 文件，创建 GitHub Release，并更新 README 下载标记。
