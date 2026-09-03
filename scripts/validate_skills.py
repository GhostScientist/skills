#!/usr/bin/env python3
"""
Validate every skill in this repository against the Agent Skills specification.

Spec: https://agentskills.io/specification

Checks performed:
  * Frontmatter parses and contains only the six spec-allowed fields
  * name    -- 1-64 chars, lowercase alnum + hyphens, matches parent directory
  * description -- non-empty, <= 1024 characters
  * compatibility -- <= 500 characters
  * metadata -- flat map of string keys to string values
  * SKILL.md body <= 500 lines
  * No orphaned bundled files (files in scripts/, references/, assets/ that
    nothing references, and which the agent therefore can never load)
  * marketplace.json lists every skill, and every listed path exists

Usage:
    python3 scripts/validate_skills.py
    python3 scripts/validate_skills.py --skills-dir skills
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Per the spec's closed set of allowed frontmatter fields. Anything else is a
# hard error in Anthropic's packaging path, so treat it as one here too.
ALLOWED_FIELDS = {
    "name",
    "description",
    "license",
    "allowed-tools",
    "metadata",
    "compatibility",
}

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_COMPATIBILITY_LENGTH = 500
MAX_BODY_LINES = 500

BUNDLE_DIRS = ("scripts", "references", "assets")


def parse_frontmatter(text: str):
    """Return (frontmatter_dict, body, error).

    Uses PyYAML when available; otherwise falls back to a minimal parser that
    handles the flat-scalars-plus-one-nested-map subset the spec allows. The
    fallback errors out loudly on anything it cannot parse rather than
    silently passing it.
    """
    match = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
    if not match:
        return None, None, "missing or malformed YAML frontmatter"
    raw, body = match.group(1), match.group(2)

    try:
        import yaml

        return yaml.safe_load(raw) or {}, body, None
    except ImportError:
        pass

    data: dict = {}
    current_key = None
    for lineno, line in enumerate(raw.split("\n"), start=2):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith((" ", "\t")):
            if current_key is None:
                return None, None, f"line {lineno}: unexpected indentation"
            sub = re.match(r"^\s+([A-Za-z0-9_-]+):\s*(.*)$", line)
            if not sub:
                return None, None, f"line {lineno}: cannot parse nested key"
            data.setdefault(current_key, {})
            if not isinstance(data[current_key], dict):
                return None, None, f"line {lineno}: '{current_key}' has both a value and children"
            data[current_key][sub.group(1)] = _scalar(sub.group(2))
            continue
        top = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if not top:
            return None, None, f"line {lineno}: cannot parse '{line[:40]}'"
        key, value = top.group(1), top.group(2).strip()
        current_key = key
        data[key] = {} if value == "" else _scalar(value)
    return data, body, None


def _scalar(value: str):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def validate_name(name, skill_dir: Path, errors: list):
    if not isinstance(name, str) or not name:
        errors.append("'name' must be a non-empty string")
        return
    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"'name' is {len(name)} chars, exceeds {MAX_NAME_LENGTH}")
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name):
        errors.append(
            f"'name' ({name!r}) must be lowercase alphanumerics separated by single "
            "hyphens, with no leading, trailing, or consecutive hyphens"
        )
    if name != skill_dir.name:
        errors.append(f"'name' ({name!r}) must match directory name ({skill_dir.name!r})")


def validate_metadata(metadata, errors: list):
    if not isinstance(metadata, dict):
        errors.append("'metadata' must be a map of string keys to string values")
        return
    for key, value in metadata.items():
        if not isinstance(value, (str, int, float)):
            errors.append(f"'metadata.{key}' must be a string value")
        elif not isinstance(value, str):
            errors.append(
                f"'metadata.{key}' is {type(value).__name__}; quote it so it stays a string"
            )


def find_orphans(skill_dir: Path, skill_md: str) -> list:
    """Bundled files that nothing references are unreachable to the agent."""
    corpus = [skill_md]
    for bundle in BUNDLE_DIRS:
        for path in sorted((skill_dir / bundle).rglob("*")):
            if path.is_file() and path.suffix in {".md", ".txt"}:
                corpus.append(path.read_text(encoding="utf-8", errors="replace"))
    haystack = "\n".join(corpus)

    orphans = []
    for bundle in BUNDLE_DIRS:
        bundle_dir = skill_dir / bundle
        if not bundle_dir.is_dir():
            continue
        for path in sorted(bundle_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(skill_dir).as_posix()
            if rel not in haystack and path.name not in haystack:
                orphans.append(rel)
    return orphans


def validate_skill(skill_dir: Path) -> list:
    errors: list = []
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return ["missing SKILL.md"]

    text = skill_md.read_text(encoding="utf-8")
    frontmatter, body, parse_error = parse_frontmatter(text)
    if parse_error:
        return [parse_error]
    if not isinstance(frontmatter, dict):
        return ["frontmatter must be a mapping"]

    unknown = sorted(set(frontmatter) - ALLOWED_FIELDS)
    if unknown:
        errors.append(
            f"unexpected frontmatter key(s): {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(ALLOWED_FIELDS))}"
        )

    if "name" not in frontmatter:
        errors.append("'name' is required")
    else:
        validate_name(frontmatter["name"], skill_dir, errors)

    description = frontmatter.get("description")
    if not description:
        errors.append("'description' is required and must be non-empty")
    elif not isinstance(description, str):
        errors.append("'description' must be a string")
    elif len(description) > MAX_DESCRIPTION_LENGTH:
        errors.append(
            f"'description' is {len(description)} chars, exceeds {MAX_DESCRIPTION_LENGTH}"
        )

    compatibility = frontmatter.get("compatibility")
    if isinstance(compatibility, str) and len(compatibility) > MAX_COMPATIBILITY_LENGTH:
        errors.append(
            f"'compatibility' is {len(compatibility)} chars, exceeds {MAX_COMPATIBILITY_LENGTH}"
        )

    if "version" in frontmatter:
        errors.append("'version' is not a top-level spec field; nest it under 'metadata'")

    if "metadata" in frontmatter:
        validate_metadata(frontmatter["metadata"], errors)

    body_lines = len(body.split("\n"))
    if body_lines > MAX_BODY_LINES:
        errors.append(
            f"SKILL.md body is {body_lines} lines, exceeds {MAX_BODY_LINES}; "
            "move detail into references/"
        )

    for orphan in find_orphans(skill_dir, text):
        errors.append(f"orphaned bundled file (never referenced): {orphan}")

    return errors


def validate_marketplace(repo_root: Path, skill_dirs: list) -> list:
    errors: list = []
    manifest = repo_root / ".claude-plugin" / "marketplace.json"
    if not manifest.is_file():
        return ["missing .claude-plugin/marketplace.json"]

    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"marketplace.json is not valid JSON: {exc}"]

    for field in ("name", "owner", "plugins"):
        if field not in data:
            errors.append(f"marketplace.json missing required field '{field}'")

    listed: set = set()
    for plugin in data.get("plugins", []):
        for field in ("name", "source"):
            if field not in plugin:
                errors.append(f"plugin {plugin.get('name', '?')!r} missing required '{field}'")
        for entry in plugin.get("skills", []):
            path = repo_root / entry.lstrip("./")
            listed.add(path.name)
            if not (path / "SKILL.md").is_file():
                errors.append(f"marketplace.json references missing skill: {entry}")

    for name in sorted({d.name for d in skill_dirs} - listed):
        errors.append(f"skill '{name}' exists on disk but is not listed in marketplace.json")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skills-dir", default="skills")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    skills_dir = repo_root / args.skills_dir
    if not skills_dir.is_dir():
        print(f"error: no such directory: {skills_dir}", file=sys.stderr)
        return 2

    skill_dirs = sorted(d for d in skills_dir.iterdir() if d.is_dir())
    failures = 0

    for skill_dir in skill_dirs:
        errors = validate_skill(skill_dir)
        if errors:
            failures += 1
            print(f"FAIL  {skill_dir.name}")
            for error in errors:
                print(f"      - {error}")
        else:
            print(f"ok    {skill_dir.name}")

    marketplace_errors = validate_marketplace(repo_root, skill_dirs)
    if marketplace_errors:
        failures += 1
        print("FAIL  marketplace.json")
        for error in marketplace_errors:
            print(f"      - {error}")
    else:
        print("ok    marketplace.json")

    print()
    if failures:
        print(f"{failures} check group(s) failed across {len(skill_dirs)} skills")
        return 1
    print(f"All {len(skill_dirs)} skills valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
