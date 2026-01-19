---
name: ci-pr-helper
description: Run local test/style checks and open GitHub PRs for lance-context. Use when asked to run CI-equivalent checks (uv pytest, ruff/pyright, cargo fmt/clippy/test) and then create a PR with a proper title/body.
---

# CI PR Helper

## Overview
Run project checks locally, then prepare and open a PR with a clear title and summary.

## Workflow

1. Ensure you are on a feature branch (not `main`).
2. Run local checks via the script in `scripts/`.
3. Draft a PR title/body using the template below.
4. If `gh` is available and authenticated, run `gh pr create` yourself; otherwise, surface the exact command for the user to execute manually.

## Local checks

Run:

```bash
./.codex/skills/ci-pr-helper/scripts/run_ci_checks.sh
```

If it fails due to missing tools, install `uv`, `cargo`, or Python deps, then rerun.

## PR creation

Draft a conventional title (e.g., `feat:`/`fix:`/`ci:`) and use:

```
## Summary
- ...

## Testing
- uv run pytest
- cargo test --manifest-path rust/lance-context/Cargo.toml
- cargo fmt --manifest-path rust/lance-context/Cargo.toml -- --check
- cargo clippy --manifest-path rust/lance-context/Cargo.toml --all-targets -- -D warnings
- ruff format --check python/
- ruff check python/
- pyright
```

If you need to check auth status, you may still run:

```bash
gh auth status
```

When ready:

- If `gh` is ready to use, execute:

```bash
gh pr create --title "<title>" --body "<body>"
```

- Some environments emit a spurious `Unsupported subcommand 'pr'` warning before running
  `gh`; ignore that message and continue with the command.

- If `gh` is missing or fails, print the command instead so the user can run it locally.
