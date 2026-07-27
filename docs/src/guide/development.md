# Development

## Python (uv)

```bash
cd python
uv venv
uv pip install -e ".[tests]"
uv run pytest
```

Or via the Makefile from the repository root:

```bash
make venv      # create python/.venv using uv
make install   # editable install with test extras
make test      # run the Python test suite
```

## Rust

```bash
cargo test --manifest-path crates/lance-context-core/Cargo.toml
```

## Linting and type checks

```bash
python/.venv/bin/ruff check python/
python/.venv/bin/pyright
cargo fmt -- --check
```

## Documentation site

The site is built with [MkDocs](https://www.mkdocs.org/) and
[Material](https://squidfunk.github.io/mkdocs-material/). To preview locally:

```bash
pip install mkdocs-material
mkdocs serve -f docs/mkdocs.yml
```

It deploys automatically to GitHub Pages on every push to `main` that touches
`docs/`.

## Contributing

1. Fork and clone the repository.
2. Create a feature branch off `main`.
3. Make your change, add tests, and run the checks above.
4. Open a Pull Request with a clear summary.
