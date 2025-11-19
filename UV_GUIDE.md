# Using uv with HyperNodes

This project is optimized for [uv](https://github.com/astral-sh/uv), the modern Python package manager.

## What is uv?

`uv` is an extremely fast Python package and project manager written in Rust by Astral (the creators of Ruff). It's a drop-in replacement for pip, pip-tools, virtualenv, and more.

### Key Benefits

- ⚡ **10-100x faster** than pip
- 🔧 **Unified toolchain** - one tool for everything
- 📦 **Built-in publishing** - no need for build/twine
- 🔒 **Lock files** - reproducible environments
- 🎯 **Better defaults** - automatic venv management

## Installation

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Or with homebrew
brew install uv
```

## Common Tasks

### Development

```bash
# Install dependencies (creates venv automatically)
uv sync

# Install with optional dependencies
uv sync --extra viz --extra telemetry

# Install all optional dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Add a new dependency
uv add numpy

# Add a dev dependency
uv add --dev ruff
```

### Building and Publishing

```bash
# Build the package
uv build

# Publish to PyPI
export UV_PUBLISH_TOKEN="pypi-YOUR_TOKEN"
uv publish

# Publish to TestPyPI
uv publish \
  --publish-url https://test.pypi.org/legacy/ \
  --token pypi-YOUR_TESTPYPI_TOKEN

# Dry run (test without uploading)
uv publish --dry-run
```

### Running Scripts

```bash
# Run a Python file in the project environment
uv run python script.py

# Run a specific example
uv run python examples/fluent_api_example.py

# Run with specific Python version
uv run --python 3.12 python script.py
```

### Managing Environments

```bash
# Create a virtual environment
uv venv

# Create with specific Python version
uv venv --python 3.12

# Activate (standard venv activation)
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Or run commands directly without activation
uv run python your_script.py
```

## Project Structure

This project uses:
- `pyproject.toml` - Project metadata and dependencies
- `uv.lock` - Locked dependency versions (committed to git)
- `src/hypernodes/` - Source code
- `tests/` - Test suite

## For Contributors

### First-time Setup

```bash
# Clone the repo
git clone https://github.com/gilad-rubin/hypernodes.git
cd hypernodes

# Install all dependencies (including dev)
uv sync --all-extras

# Install pre-commit hooks (if configured)
uv run pre-commit install

# Run tests
uv run pytest
```

### Making Changes

```bash
# Make your changes...

# Run tests
uv run pytest

# Format code (if using ruff)
uv run ruff format .

# Lint code
uv run ruff check .
```

### Adding Dependencies

```bash
# Add runtime dependency
uv add package-name

# Add optional dependency (to a group)
# Edit pyproject.toml manually to add to [project.optional-dependencies]

# Add dev dependency
uv add --dev package-name

# Lock dependencies
uv lock
```

## Comparison to Traditional Tools

### Old workflow:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install build twine
python -m build
python -m twine upload dist/*
```

### With uv:
```bash
uv sync --all-extras
uv build
uv publish
```

**Result:** Faster, simpler, more reliable! ⚡

## Common Commands Cheat Sheet

| Task | Command |
|------|---------|
| Install dependencies | `uv sync` |
| Install with extras | `uv sync --extra viz` |
| Add dependency | `uv add package` |
| Run tests | `uv run pytest` |
| Run script | `uv run python script.py` |
| Build package | `uv build` |
| Publish to PyPI | `uv publish --token TOKEN` |
| Create venv | `uv venv` |
| Install package | `uv pip install package` |

## Why uv for This Project?

1. **Speed**: Development iterations are much faster
2. **Simplicity**: One tool instead of many
3. **Reliability**: Lock files ensure reproducible builds
4. **Modern**: Built for Python 3.12+ workflows
5. **Publishing**: Integrated build and publish workflow

## Learn More

- [uv Documentation](https://docs.astral.sh/uv/)
- [uv GitHub](https://github.com/astral-sh/uv)
- [Astral Blog](https://astral.sh/blog)

## Fallback to pip

If you prefer using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

But we highly recommend trying uv! 🚀
