#!/usr/bin/env python3
"""
Script to calculate the next version for lance-context based on release type.
"""

import argparse
import os
import sys
from packaging import version


def calculate_next_version(current_version: str, release_type: str, channel: str) -> str:
    """Calculate the next version based on release type and channel."""

    parsed = version.parse(current_version)

    if hasattr(parsed, "release"):
        major, minor, patch = (
            parsed.release[:3] if len(parsed.release) >= 3 else (*parsed.release, 0, 0)[:3]
        )
    else:
        parts = current_version.split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0

    if release_type == "major":
        new_version = f"{major + 1}.0.0"
    elif release_type == "minor":
        new_version = f"{major}.{minor + 1}.0"
    elif release_type == "patch":
        new_version = f"{major}.{minor}.{patch + 1}"
    elif release_type == "current":
        new_version = current_version
    else:
        raise ValueError(f"Unknown release type: {release_type}")

    return new_version


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate next version")
    parser.add_argument("--current", required=True, help="Current version")
    parser.add_argument(
        "--type",
        required=True,
        choices=["major", "minor", "patch", "current"],
        help="Release type",
    )
    parser.add_argument(
        "--channel",
        required=True,
        choices=["stable", "preview"],
        help="Release channel",
    )

    args = parser.parse_args()

    try:
        new_version = calculate_next_version(args.current, args.type, args.channel)
        print(f"version={new_version}")

        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a", encoding="utf-8") as fh:
                fh.write(f"version={new_version}\n")

    except Exception as exc:  # pragma: no cover - used in CI contexts
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
