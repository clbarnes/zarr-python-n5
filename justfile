# List available targets.
default:
    just --list

# Generate documentation under `./docs/`.
docs:
    rm -rf docs/zarr_n5
    uv run pdoc \
        --output-directory docs \
        --no-include-undocumented \
        --docformat markdown \
        --search \
        zarr_n5

# Run linters and type checkers.
lint:
    uv run ruff check src tests
    uv run mypy src tests
    uv run ruff format --check src tests

# Format python code.
format:
    uv run ruff format src tests

# Run unit tests.
test:
    uv run pytest -v
