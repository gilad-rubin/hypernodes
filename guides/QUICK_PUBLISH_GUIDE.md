# Quick Start: Publishing HyperNodes to PyPI with uv

## One-Time Setup

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Get your PyPI API token from:
#    https://pypi.org/manage/account/token/

# 3. Set token as environment variable
export UV_PUBLISH_TOKEN="pypi-YOUR_TOKEN_HERE"

# Add to your shell profile to make it permanent:
echo 'export UV_PUBLISH_TOKEN="pypi-YOUR_TOKEN_HERE"' >> ~/.bashrc  # or ~/.zshrc
```

## Publishing Workflow

### The Simple Version (3 commands!)

```bash
# 1. Build the package
uv build

# 2. Test on TestPyPI (optional but recommended)
uv publish --publish-url https://test.pypi.org/legacy/ --token pypi-TESTPYPI_TOKEN

# 3. Publish to PyPI
uv publish
```

Done! 🎉

---

## Detailed Workflow

### For New Releases

```bash
# Step 1: Update version
# Edit pyproject.toml and change: version = "0.2.0"

# Step 2: Update changelog
# Add changes to CHANGELOG.md

# Step 3: Commit and tag
git add pyproject.toml CHANGELOG.md
git commit -m "Release v0.2.0"
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin main --tags

# Step 4: Build
cd /workspace
uv build --clear

# Step 5: Publish
uv publish
```

---

## Testing Locally

### Quick Test

```bash
# Build
uv build

# Create test environment and install
uv venv test-env
source test-env/bin/activate
uv pip install dist/hypernodes-*.whl

# Test it works
python -c "from hypernodes import Pipeline, node; print('✓ Works!')"

# Cleanup
deactivate
rm -rf test-env
```

### Test on TestPyPI

```bash
# Set TestPyPI token
export UV_PUBLISH_TOKEN="pypi-YOUR_TESTPYPI_TOKEN"
export UV_PUBLISH_URL="https://test.pypi.org/legacy/"

# Publish to test
uv publish

# Install from test
uv pip install --index-url https://test.pypi.org/simple/ hypernodes
```

---

## Command Reference

### Building

```bash
uv build                # Build wheel + sdist
uv build --wheel        # Build only wheel
uv build --sdist        # Build only sdist
uv build --clear        # Clear dist/ first
```

### Publishing

```bash
# To PyPI (production)
uv publish --token pypi-TOKEN

# To TestPyPI
uv publish \
  --publish-url https://test.pypi.org/legacy/ \
  --token pypi-TESTPYPI_TOKEN

# Dry run (no upload)
uv publish --dry-run

# With environment variable
export UV_PUBLISH_TOKEN="pypi-TOKEN"
uv publish
```

### Installation (for users)

```bash
# Basic install
uv pip install hypernodes

# With optional dependencies
uv pip install "hypernodes[all]"          # Everything
uv pip install "hypernodes[viz,telemetry]"  # Specific extras
```

---

## Environment Variables

```bash
# Publishing
export UV_PUBLISH_TOKEN="pypi-YOUR_TOKEN"           # Auth token
export UV_PUBLISH_URL="https://pypi.org/legacy/"    # Upload URL (optional, default is PyPI)
export UV_PUBLISH_USERNAME="__token__"              # Username (optional, defaults to __token__)

# For TestPyPI
export UV_PUBLISH_TOKEN="pypi-YOUR_TESTPYPI_TOKEN"
export UV_PUBLISH_URL="https://test.pypi.org/legacy/"
```

---

## Troubleshooting

### Build Issues

```bash
# Clean everything and rebuild
rm -rf dist/ build/ *.egg-info/
uv build --clear
```

### Token Issues

```bash
# Make sure token starts with "pypi-"
echo $UV_PUBLISH_TOKEN

# Test with dry run
uv publish --dry-run --token pypi-YOUR_TOKEN
```

### Version Already Exists

You can't overwrite. Increment version in `pyproject.toml`:

```toml
[project]
version = "0.1.1"  # Bump this
```

---

## Comparing to Old Workflow

### ❌ Old way (multiple tools):
```bash
pip install build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

### ✅ New way (uv):
```bash
uv build
uv publish
```

**That's it!** Simpler, faster, better. 🚀

---

## Package Information

- **Name:** hypernodes
- **Current Version:** 0.1.0
- **Repository:** https://github.com/gilad-rubin/hypernodes
- **PyPI:** https://pypi.org/project/hypernodes/

## Quick Links

- **Get PyPI token:** https://pypi.org/manage/account/token/
- **Get TestPyPI token:** https://test.pypi.org/manage/account/token/
- **uv docs:** https://docs.astral.sh/uv/
- **Issues:** https://github.com/gilad-rubin/hypernodes/issues

---

## The Absolute Minimum

Already have `uv` and a token?

```bash
uv build && uv publish --token pypi-YOUR_TOKEN
```

Done! 🎯
