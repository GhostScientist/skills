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

## Creating Your Own Skills

Use the `template/SKILL.md` as a starting point. Each skill needs:

1. A folder in `skills/` with your skill name
2. A `SKILL.md` file with YAML frontmatter (`name` and `description`) and markdown instructions
3. An entry in `.claude-plugin/marketplace.json`

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
