# Development

## Python (uv)

```bash
cd python
uv venv
uv pip install -e ".[tests]"
uv run pytest
```

## Python release dependency locking

The Python crate is a member of the root Cargo workspace. Keep only the root
`Cargo.lock`: an obsolete `python/Cargo.lock` makes Maturin package that nested
lock instead, leaving source-distribution builds without a workspace lock.

From the repository root, build release artifacts with
`maturin build --manifest-path python/Cargo.toml --release --sdist`.
The publish workflow verifies that the sdist contains the exact root lockfile
and no nested lock. Cargo can then retain the pinned dependency versions when
building the source archive.

Do not add `--locked` to the combined `--sdist` build: Maturin removes workspace
members unused by the Python package, so Cargo must prune their lockfile
entries in the extracted tree. This is different from resolving versions
without a lockfile. Dependency upgrades belong in the checked-in root lock.
