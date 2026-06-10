#!/usr/bin/env node
// Lightweight public-release guard for secrets and local machine paths.

import { readFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import path from "node:path";
import process from "node:process";
import { REPO_ROOT } from "./lib/skills.mjs";

const SECRET_KEY_RE = /\bsk-[A-Za-z0-9][A-Za-z0-9_-]{12,}\b/g;

const CHECKS = [
  {
    name: "private key block",
    re: /BEGIN [A-Z ]*PRIVATE KEY/g,
  },
  {
    name: "local user path",
    re: new RegExp("/" + "Users/" + "bi" + "ws" + "01", "g"),
  },
  {
    name: "private Codex document path",
    re: new RegExp("Documents/" + "Codex", "g"),
  },
  {
    name: "private workspace marker",
    re: new RegExp("\\bxm" + "26\\b", "g"),
  },
  {
    name: "API key",
    re: SECRET_KEY_RE,
  },
];

const SKIP_EXTENSIONS = new Set([
  ".gif",
  ".gz",
  ".ico",
  ".jpeg",
  ".jpg",
  ".pdf",
  ".png",
  ".safetensors",
  ".sha256",
  ".zip",
]);

function git(args) {
  return execFileSync("git", args, {
    cwd: REPO_ROOT,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
}

function listCandidateFiles() {
  const raw = git(["ls-files", "-z", "--cached", "--others", "--exclude-standard"]);
  return raw
    .split("\0")
    .filter(Boolean)
    .filter((file) => !file.startsWith("dist/"))
    .filter((file) => !SKIP_EXTENSIONS.has(path.extname(file).toLowerCase()));
}

function lineAndColumn(content, index) {
  const before = content.slice(0, index);
  const lines = before.split("\n");
  return { line: lines.length, column: lines[lines.length - 1].length + 1 };
}

async function scanFile(file) {
  const absolute = path.join(REPO_ROOT, file);
  const content = await readFile(absolute, "utf8").catch(() => null);
  if (content === null || content.includes("\0")) return [];

  const findings = [];
  for (const check of CHECKS) {
    check.re.lastIndex = 0;
    for (const match of content.matchAll(check.re)) {
      const location = lineAndColumn(content, match.index ?? 0);
      findings.push({
        file,
        ...location,
        name: check.name,
        text: match[0],
      });
    }
  }
  return findings;
}

function redact(text) {
  if (SECRET_KEY_RE.test(text)) return "sk-***";
  return text;
}

async function main() {
  const files = listCandidateFiles();
  const findings = [];
  for (const file of files) findings.push(...(await scanFile(file)));

  if (findings.length === 0) {
    console.log(`[sensitive] OK (${files.length} files scanned)`);
    return;
  }

  console.error(`[sensitive] Found ${findings.length} sensitive marker(s):`);
  for (const f of findings) {
    console.error(`  ${f.file}:${f.line}:${f.column} ${f.name}: ${redact(f.text)}`);
  }
  process.exit(1);
}

main().catch((err) => {
  console.error(`[sensitive] ERROR: ${err.message}`);
  process.exit(1);
});
