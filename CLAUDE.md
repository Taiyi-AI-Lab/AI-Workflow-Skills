# CLAUDE.md

This file provides guidance to Claude Code and other coding agents when working
in this repository.

## Project Overview

AI Workflow Skills is a public repository of reusable SKILL.md-compatible agent
skills. Each skill is a standalone package with its own `manifest.json` version
and release tag.

Current skills:

- `grpo-finetune`: local CUDA GRPO fine-tuning workflow for LLMs
- `mint-lora-training`: cloud-hosted MinT LoRA training workflow

## Repository Shape

```text
skills/                    # Skill source packages
scripts/release/           # Zero-dependency ESM maintainer tooling
.claude-plugin/            # Plugin marketplace metadata
.github/workflows/         # Validation and tag-driven release workflows
```

Each public skill should include:

```text
<skill-name>/
├── README.md              # Human-facing usage guide
├── SKILL.md               # Required agent instructions and frontmatter
├── manifest.json          # Required release metadata
├── references/            # Optional reference docs
├── scripts/               # Optional deterministic helper scripts
└── assets/                # Optional templates or static assets
```

## Common Commands

```bash
npm run list              # List skills and validate manifest structure
npm run pack:all          # Build release zips in dist/release/
npm run readme:check      # Verify README download marker sync
npm run check:python      # Compile bundled Python scripts
npm run check:sensitive   # Scan for secrets and private local paths
npm run validate          # Full CI-equivalent validation
```

## Release Flow

1. Keep the working tree clean and synced with `origin/main`.
2. Run `npm run release:dry`.
3. Run `npm run release` when the dry run is correct.
4. The release script bumps manifests when needed, syncs README markers, commits,
   creates tags, and pushes.
5. `.github/workflows/release-skill.yml` builds the zip, creates a GitHub
   Release, and syncs README download links back to the default branch.

Release tags use `<skill-name>-v<semver>`, for example:

```text
grpo-finetune-v2.0.0
mint-lora-training-v1.0.0
```

## Guardrails

- Do not commit API keys, `.env` files, model weights, adapters, checkpoints, or
  training outputs.
- Keep `manifest.json#name`, `SKILL.md` frontmatter `name`, and the folder name
  identical.
- Use SemVer for per-skill versions.
- Update English, Chinese, and Japanese README entries together.
- Run `npm run validate` before handing work back.

## References

- [agentskills.io](https://agentskills.io) — SKILL.md specification
- [Anthropic skill examples](https://github.com/anthropics/skills)
- [Contributing guide](./CONTRIBUTING.md)
