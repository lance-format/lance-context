from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from lance_context import Context


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    artifacts_dir = project_root / ".artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    dataset_path = artifacts_dir / f"travel_context_{uuid4().hex}.lance"
    ctx = Context.create(dataset_path.as_posix())
    print(f"Created context store at {dataset_path}")

    ctx.add("system", "You are a friendly travel agent who compares destinations.")
    ctx.add("user", "Where should I travel in spring if I love street food?")
    ctx.add(
        "assistant",
        "Consider Osaka, Japan for cherry blossoms and the Kuromon market.",
    )

    first_version = ctx.version()

    ctx.add("user", "Any budget-friendly option in Europe?")
    ctx.add(
        "assistant",
        "Porto, Portugal offers great food, coastal views, and reasonable prices.",
    )

    print(f"Entries stored: {ctx.entries()}")
    print(f"Current version: {ctx.version()}")

    ctx.checkout(first_version)
    print(f"Rolled back to version {first_version}; entries now {ctx.entries()}")

    print(
        "Re-run this script to create a fresh dataset, or reuse the printed path to "
        "append more entries from another process."
    )


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
