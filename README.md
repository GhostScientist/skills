# GhostScientist Skills

Agent Skills for research, deployment, Apple platforms, and technical writing.

These follow the [Agent Skills specification](https://agentskills.io/specification) — an open
standard supported by Claude Code, GitHub Copilot, Cursor, VS Code, Gemini CLI, and others. The
skills here are plain `SKILL.md` directories, so they work in any compliant client.

## Installation

### Claude Code (plugin marketplace)

```
/plugin marketplace add GhostScientist/skills
/plugin install research-skills
```

### Any other client

Skills are just directories. Clone the repo and copy or symlink the ones you want into your
client's skills directory:

```bash
git clone https://github.com/GhostScientist/skills.git
ln -s "$PWD/skills/paper-to-intuition" ~/.agents/skills/paper-to-intuition
```

| Client | Personal | Project |
|---|---|---|
| Cross-vendor | `~/.agents/skills/` | `.agents/skills/` |
| GitHub Copilot | `~/.copilot/skills/` | `.github/skills/` |
| Claude Code | `~/.claude/skills/` | `.claude/skills/` |
| Cursor | `~/.cursor/skills/` | `.cursor/skills/` |

`~/.agents/skills/` is the emerging vendor-neutral location and is read by both Copilot and
Cursor. Cursor and Copilot also read the Claude directories.

GitHub Copilot CLI can install a skill directly:

```bash
copilot skill add ./skills/paper-to-intuition
```

## Available Skills

### research-skills

Understanding papers, designing experiments, and developing research intuition.

| Skill | Description |
|-------|-------------|
| `paper-to-intuition` | Turns an academic paper into multi-layered understanding (ELI5 → researcher) with visual intuition diagrams and "what breaks if we remove X" analysis. |
| `implement-paper-from-scratch` | Walks through implementing a research paper step by step, with checkpoint questions to verify understanding. No copy-pasting. |
| `research-question-refiner` | Turns a vague research interest into a concrete, tractable research question with feasibility analysis and litmus tests. |
| `experiment-design-checklist` | Produces a rigorous experiment design: baselines, ablations, controls, statistical tests, compute budget, and confound mitigation. |
| `reviewer-2-simulator` | Critiques a paper draft as a skeptical but fair reviewer would — weak claims, missing baselines, and overclaims, before your actual reviewers find them. |
| `research-taste-developer` | Builds intuition for what separates good research from incremental research, by analyzing patterns in highly-cited work. |

### deployment-skills

Deploying ML models and applications to cloud platforms.

| Skill | Description |
|-------|-------------|
| `hugging-face-space-deployer` | Deploys a model to a Hugging Face Space. Auto-detects full models vs LoRA/PEFT adapters vs Inference API support, picks hardware, and ships Gradio/Streamlit templates. |

### design-skills

Visual design and Apple platform work.

| Skill | Description |
|-------|-------------|
| `ios-app-icon-generator` | Creates a complete iOS app icon set at every required size. Defines the visual identity first, then generates a self-contained HTML artifact with downloadable PNGs. |
| `create-watchos-version` | Analyzes an existing Apple platform project and produces a phased plan for a watchOS companion or standalone app, with API availability warnings before any code is written. |

### writing-skills

Technical writing and content creation.

| Skill | Description |
|-------|-------------|
| `turn-this-feature-into-a-blog-post` | Generates a technical blog post from a code implementation, structured What → Why → How in a friendly, authoritative voice. |

## Creating Your Own Skills

Start from [`template/SKILL.md`](template/SKILL.md), which documents the full frontmatter schema
inline.

Each skill needs:

1. A directory in `skills/` whose name matches the skill's `name` field exactly
2. A `SKILL.md` with `name` and `description` in YAML frontmatter
3. An entry in the appropriate plugin's `skills` array in `.claude-plugin/marketplace.json`

Key constraints from the spec:

- **`name`** — required, max 64 chars, lowercase alphanumerics and hyphens, must match the
  directory name.
- **`description`** — required, max 1024 chars. This is the only text the agent sees when
  deciding whether to load the skill, so enumerate trigger contexts explicitly and err on the
  side of being pushy.
- **Only six frontmatter fields are allowed**: `name`, `description`, `license`,
  `compatibility`, `metadata`, `allowed-tools`. Note that `version` is **not** top-level — it
  belongs under `metadata`. Claude Code accepts extra proprietary fields, but they hard-error
  when packaging for claude.ai or the Skills API.
- **Keep `SKILL.md` under 500 lines.** Move detail into `references/`, executables into
  `scripts/`, and templates into `assets/`. Always tell the agent *when* to load each file.

### Validation

```bash
python3 scripts/validate_skills.py
```

This checks spec compliance, body length, marketplace/disk sync, and orphaned bundled files
(files in `scripts/`, `references/`, or `assets/` that nothing references — the agent can never
load these). It runs on every push and pull request via
[`.github/workflows/validate-skills.yml`](.github/workflows/validate-skills.yml).

## License

Apache 2.0 — see [LICENSE](LICENSE).
