---
# REQUIRED. 1-64 chars, lowercase letters/numbers/hyphens only.
# Must match this skill's directory name exactly.
name: your-skill-name

# REQUIRED. Max 1024 characters. This is the only thing the agent sees when
# deciding whether to load the skill, so it does all the triggering work.
# Pattern: "<what it does>. Use when <enumerated trigger contexts>."
# Be imperative and slightly pushy -- agents under-trigger skills far more
# often than they over-trigger them. List the situations where it applies,
# including ones where the user won't name the domain directly.
description: Use this skill when <situation>. Covers <capabilities>. Use it whenever the user mentions <keywords>, asks to <phrasings>, or <adjacent situation> -- even if they don't explicitly say <the obvious term>.

# OPTIONAL. License name, or a reference to a bundled license file.
license: Apache-2.0

# OPTIONAL. Max 500 characters. Environment requirements: required tooling,
# system packages, network access. Most skills do not need this field.
# compatibility: Requires Python 3.10+ and network access to api.example.com

# OPTIONAL. Arbitrary string-to-string map. Note that `version` lives HERE --
# there is no top-level `version` field in the spec.
metadata:
  author: GhostScientist
  version: "1.0"

# OPTIONAL (experimental). Space-separated pre-approved tools. Support varies
# between agents, and the accepted values differ across clients -- Claude Code
# uses forms like `Bash(git:*) Read`, while Copilot uses `shell`. Omit unless
# you need it.
# allowed-tools: Bash(git:*) Read
---

# Your Skill Name

<!--
These six fields above are the COMPLETE set allowed by the Agent Skills spec
(https://agentskills.io/specification). Anything else -- `argument-hint`,
`model`, `context`, `paths`, `when_to_use`, etc. -- is Claude Code-specific and
will hard-error when packaging for claude.ai or the Skills API.

Run `python3 scripts/validate_skills.py` to check this file.
-->

One or two sentences on what this skill does and the failure mode it prevents.

## Workflow

1. Step one
2. Step two
3. Step three

## Step 1: ...

Write in the imperative, addressed to the agent. State what to do rather than
narrating why -- once a skill loads it stays in context, so every line is a
recurring token cost.

Only include what the agent would otherwise get wrong. If it already knows
something, cut it.

## Gotchas

- Concrete, non-obvious failure modes and their fixes.
- Avoid pinning versions, dates, or prices that will silently rot. Point at the
  authoritative URL instead.

<!--
Keep this body under 500 lines. Move detail into bundled files:

  your-skill-name/
  ├── SKILL.md       # this file
  ├── scripts/       # executable code the agent can run
  ├── references/    # docs loaded on demand
  └── assets/        # templates, images, data files

Reference them with relative paths, one level deep, and always say WHEN to
load each one:

  "Read `references/troubleshooting.md` if the build fails."   <- good
  "See references/ for more details."                          <- bad

Every bundled file must be referenced from SKILL.md (or from another reference
file). Unreferenced files can never be loaded by the agent; the validator
flags them as orphans.
-->
