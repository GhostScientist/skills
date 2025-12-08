# GhostScientist Skills

Custom skills for Claude Code.

## Installation

Add this repository as a marketplace in Claude Code:

```
/plugin marketplace add GhostScientist/skills
```

Then install the skills you want:

```
/plugin install writing-skills
```

## Available Skills

### writing-skills

Skills for technical writing and content creation.

| Skill | Description |
|-------|-------------|
| `turn-this-feature-into-a-blog-post` | Generates a technical blog post from code implementation. Structures content as What → Why → How with a friendly, authoritative style. |

### design-skills

Skills for visual design and creative asset generation.

| Skill | Description |
|-------|-------------|
| `ios-app-icon-generator` | Creates a complete iOS app icon set with all required sizes. Follows a philosophy-first approach, then generates a self-contained HTML artifact with downloadable PNGs. |

### research-skills

Skills for AI/ML research: understanding papers, designing experiments, and developing research intuition.

| Skill | Description |
|-------|-------------|
| `paper-to-intuition` | Transforms an academic paper into multi-layered understanding (ELI5 → researcher) with visual intuition diagrams and "what breaks if we remove X" analysis. |
| `implement-paper-from-scratch` | Guides you through implementing a research paper step-by-step with checkpoint questions to verify understanding. No copy-pasting, just learning. |
| `research-question-refiner` | Transforms a vague research interest into a concrete, tractable research question with feasibility analysis and litmus tests. |
| `experiment-design-checklist` | Generates a rigorous experiment design: baselines, ablations, controls, statistical tests, compute budget, and confound mitigation. |
| `reviewer-2-simulator` | Critiques your paper draft as a skeptical (but fair) reviewer would. Finds weak claims, missing baselines, and overclaims before your actual reviewers do. |
| `research-taste-developer` | Develops intuition for what makes research "good" vs "incremental." Analyzes patterns in highly-cited work and what top researchers do differently. |

## Creating Your Own Skills

Use the `template/SKILL.md` as a starting point. Each skill needs:

1. A folder in `skills/` with your skill name
2. A `SKILL.md` file with YAML frontmatter (`name` and `description`) and markdown instructions
3. An entry in `.claude-plugin/marketplace.json`

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
