#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; install uv to run Python checks."
  exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found; install Rust to run Rust checks."
  exit 1
fi

cd "$root"

cargo fmt --manifest-path rust/lance-context/Cargo.toml -- --check
cargo clippy --manifest-path rust/lance-context/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path rust/lance-context/Cargo.toml

cd "$root/python"
uv run pytest
uv run ruff format --check python/
uv run ruff check python/
uv run pyright
