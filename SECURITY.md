# Security Policy

## Supported Versions

Security fixes apply to the default branch and the latest released zip for each
skill.

## Reporting a Vulnerability

Use GitHub Security Advisories when available. If advisories are not available,
open an issue with a minimal description and avoid posting secrets, tokens,
training data, or private model artifacts.

## Secret Handling

Do not commit:

- API keys or `.env` files
- model weights, LoRA adapters, checkpoints, or training outputs
- private datasets or evaluation samples
- local absolute paths that expose user names or private workspace locations

`npm run validate` includes a sensitive marker scan, but it is not a replacement
for reviewing diffs before publishing.

## Third-Party Services

Some skills call external services such as Hugging Face or MinT. Users are
responsible for checking provider terms, data handling rules, token costs, and
model license constraints before running training jobs.
