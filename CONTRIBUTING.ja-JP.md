# Contributing

AI Workflow Skills への貢献ありがとうございます。このリポジトリは
SKILL.md-compatible な reusable agent skills を公開します。各 skill は個別に
versioning され、個別に release されます。

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

Every skill must include `SKILL.md` and `manifest.json`. Public skills should
also include a concise `README.md` with prerequisites, install notes, and a
quick start.

## Before Opening a PR

Run the full validation command:

```bash
npm run validate
```

This validates manifests, builds release zips, checks README download markers,
compiles bundled Python scripts, and scans for secrets or private local paths.

Do not commit:

- API keys or `.env` files
- model weights, adapters, checkpoints, or training outputs
- local absolute paths or private workspace names
- release artifacts under `dist/release/`

## Adding or Updating a Skill

1. Create or modify `skills/<skill-name>/`.
2. Keep `skills/<skill-name>/manifest.json#name` and `SKILL.md` frontmatter
   `name` identical to the folder name.
3. Use SemVer in `manifest.json#version`.
4. Update root README entries in English, Chinese, and Japanese.
5. Add a README download marker for new skills:
   `<!-- DOWNLOAD:<skill-name>:start --><!-- DOWNLOAD:<skill-name>:end -->`.
6. Run `npm run validate`.

## Release Tags

Per-skill releases use this tag format:

```text
<skill-name>-v<semver>
```

Examples:

```text
grpo-finetune-v2.0.0
mint-lora-training-v1.0.0
```

Run `npm run release:dry` before publishing. The release workflow validates the
matching `manifest.json` version, builds a zip and `.sha256` file under
`dist/release/`, creates a GitHub Release, and updates README download markers.
