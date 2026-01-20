#!/usr/bin/env python3
"""
Version management helper for the lance-context project.

It wraps bump-my-version so we can keep the various crates and the Python package
on the same semantic version.

Examples:
  # Bump to a specific version
  python ci/bump_version.py --version 0.2.0
  python ci/bump_version.py --version 0.2.0-beta.1

  # Bump relative to the current version
  python ci/bump_version.py --bump patch
  python ci/bump_version.py --bump minor
  python ci/bump_version.py --bump major
  python ci/bump_version.py --bump pre_n
  python ci/bump_version.py --bump pre_label

  # Preview changes without touching files
  python ci/bump_version.py --version 0.2.0 --dry-run
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a command and return the completed process, aborting on failure."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=capture_output, text=True, check=False)
    if result.returncode != 0:
        print(f"Error running command: {' '.join(cmd)}")
        if capture_output:
            print(f"stderr: {result.stderr}")
        sys.exit(result.returncode)
    return result


def get_current_version() -> str:
    """Extract the current version from .bumpversion.toml."""
    config_path = Path(".bumpversion.toml")
    if not config_path.exists():
        raise FileNotFoundError(".bumpversion.toml not found in current directory")

    with config_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip().startswith('current_version = "'):
                return line.split('"')[1]

    raise ValueError("Could not find current_version in .bumpversion.toml")


def parse_version(raw: str) -> dict[str, str | None]:
    """Break a semantic version (optionally with prerelease) into components."""
    pattern = r"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(-(?P<pre_label>alpha|beta|rc)\.(?P<pre_n>\d+))?"
    match = re.match(pattern, raw)
    if not match:
        raise ValueError(f"Invalid version format: {raw}")
    return match.groupdict()


def is_prerelease(raw: str) -> bool:
    """Return True when the version string represents a pre-release tag."""
    parts = parse_version(raw)
    return parts.get("pre_label") is not None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bump version using bump-my-version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--version", help="New version to set (e.g. 0.2.0 or 0.2.0-beta.1)")
    group.add_argument(
        "--bump",
        choices=["major", "minor", "patch", "pre_label", "pre_n"],
        help="Bump a specific component",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show changes without modifying files")

    args = parser.parse_args()

    current = get_current_version()
    print(f"Current version: {current}")

    base_cmd = ["bump-my-version", "bump"]

    if args.dry_run:
        print("\nDry run mode - no changes will be made")
        base_cmd.extend(["--dry-run", "--verbose"])
    else:
        base_cmd.extend(["--no-commit", "--no-tag"])

    base_cmd.append("--allow-dirty")

    if args.version:
        target = args.version
        print(f"Target version: {target}")
        try:
            parse_version(target)
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)

        cmd = base_cmd + ["--current-version", current, "--new-version", target]
    else:
        bump_part = args.bump
        if bump_part in {"pre_n", "pre_label"} and not is_prerelease(current):
            print(f"Error: Cannot bump '{bump_part}' on stable version {current}.")
            print("Use --version to move to a pre-release first.")
            sys.exit(1)
        cmd = base_cmd + [bump_part]

    run_command(cmd, capture_output=False)

    if not args.dry_run:
        updated = get_current_version()
        print(f"\nSuccessfully updated version from {current} to {updated}")


if __name__ == "__main__":
    main()
