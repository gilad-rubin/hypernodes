# Publishing to PyPI with uv

This document describes how to publish the HyperNodes package to PyPI using `uv`, the modern Python package manager.

## Why uv?

`uv` is an extremely fast Python package and project manager written in Rust. It provides:
- **10-100x faster** than pip/pip-tools
- **Built-in publishing** - no need for separate build/twine tools
- **Unified toolchain** - one tool for dependencies, builds, and publishing
- **Better defaults** - automatic virtual environments, lock files, etc.

## Prerequisites

### 1. Install uv

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Or with homebrew (macOS)
brew install uv
```

### 2. Set up PyPI credentials

You'll need API tokens from:
- [PyPI](https://pypi.org/manage/account/token/) (production)
- [TestPyPI](https://test.pypi.org/manage/account/token/) (testing)

#### Option A: Environment Variables (Recommended)

```bash
# For PyPI
export UV_PUBLISH_TOKEN="pypi-YOUR_PYPI_TOKEN_HERE"

# For TestPyPI
export UV_PUBLISH_TOKEN="pypi-YOUR_TESTPYPI_TOKEN_HERE"
export UV_PUBLISH_URL="https://test.pypi.org/legacy/"
```

Add these to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) for persistence.

#### Option B: Pass token directly in command

```bash
uv publish --token pypi-YOUR_TOKEN_HERE
```

#### Option C: Use uv configuration file

Create `~/.config/uv/uv.toml`:

```toml
[publish]
# Default to PyPI (no URL needed for PyPI)

# For TestPyPI, add:
# url = "https://test.pypi.org/legacy/"
```

Then set token via environment variable per session.

## Building the Package

With `uv`, building is simple:

```bash
# Build both wheel and sdist
uv build

# Build only wheel
uv build --wheel

# Build only source distribution
uv build --sdist

# Clear output directory first
uv build --clear
```

This creates:
- `dist/hypernodes-X.Y.Z-py3-none-any.whl` (wheel)
- `dist/hypernodes-X.Y.Z.tar.gz` (source distribution)

## Testing Before Publishing

### 1. Test the build locally

Install from the local wheel:

```bash
# Create a test environment
uv venv test-env
source test-env/bin/activate  # or `test-env\Scripts\activate` on Windows

# Install the wheel
uv pip install dist/hypernodes-*.whl

# Test it works
python -c "
from hypernodes import Pipeline, node

@node(output_name='result')
def test(x: int) -> int:
    return x * 2

pipeline = Pipeline(nodes=[test])
print(pipeline.run(inputs={'x': 5}))
"

# Clean up
deactivate
rm -rf test-env
```

### 2. Publish to TestPyPI (Recommended)

Always test on TestPyPI first:

```bash
# Publish to TestPyPI
uv publish \
  --token pypi-YOUR_TESTPYPI_TOKEN \
  --publish-url https://test.pypi.org/legacy/

# Or with environment variable
export UV_PUBLISH_TOKEN="pypi-YOUR_TESTPYPI_TOKEN"
export UV_PUBLISH_URL="https://test.pypi.org/legacy/"
uv publish
```

Then test installing from TestPyPI:

```bash
# Install from TestPyPI
uv pip install --index-url https://test.pypi.org/simple/ hypernodes

# Verify it works
python -c "from hypernodes import Pipeline; print('Success!')"
```

## Publishing to PyPI

Once you've verified everything works on TestPyPI:

```bash
# Publish to PyPI (production)
uv publish --token pypi-YOUR_PYPI_TOKEN

# Or with environment variable
export UV_PUBLISH_TOKEN="pypi-YOUR_PYPI_TOKEN"
uv publish
```

That's it! `uv publish` automatically:
- ✅ Validates the package
- ✅ Checks for existing versions (no duplicates)
- ✅ Uploads to PyPI
- ✅ Verifies the upload

## Versioning Workflow

For each new release:

```bash
# 1. Update version in pyproject.toml
# Edit pyproject.toml: version = "0.2.0"

# 2. Update CHANGELOG.md
# Add new version section with changes

# 3. Commit changes
git add pyproject.toml CHANGELOG.md
git commit -m "Release v0.2.0"

# 4. Tag the release
git tag -a v0.2.0 -m "Release v0.2.0"

# 5. Push to GitHub
git push origin main
git push origin v0.2.0

# 6. Build with uv
uv build --clear

# 7. Test on TestPyPI (optional but recommended)
uv publish --publish-url https://test.pypi.org/legacy/ --token pypi-TESTPYPI_TOKEN

# 8. Publish to PyPI
uv publish --token pypi-YOUR_PYPI_TOKEN
```

## Advanced Options

### Dry Run

Test publishing without actually uploading:

```bash
uv publish --dry-run
```

### Check for duplicates

`uv publish` automatically checks PyPI for existing versions and skips them.

### Custom index

Publish to a private PyPI index:

```bash
uv publish \
  --publish-url https://your-private-pypi.com/simple/ \
  --token your-private-token
```

### Publish specific files

```bash
# Publish only the wheel
uv publish dist/*.whl

# Publish specific version
uv publish dist/hypernodes-0.1.0*
```

## Troubleshooting

### "File already exists"

You can't overwrite existing versions on PyPI. You must increment the version in `pyproject.toml`.

### "Invalid token"

- Ensure your token starts with `pypi-`
- Check token hasn't expired
- Verify you're using the right token (PyPI vs TestPyPI)
- Make sure token has upload permissions

### Import fails after install

Check optional dependencies:

```bash
# Install with all optional dependencies
uv pip install "hypernodes[all]"

# Or specific extras
uv pip install "hypernodes[daft,viz,telemetry]"
```

### Build fails

```bash
# Clean and rebuild
rm -rf dist/ build/ *.egg-info/
uv build --clear
```

## Comparing to Traditional Tools

### Old way (pip + build + twine):
```bash
pip install build twine
python -m build
python -m twine upload dist/*
```

### New way (uv):
```bash
uv build
uv publish --token pypi-TOKEN
```

**Result:** Simpler, faster, unified workflow! 🚀

## Post-Publishing Checklist

After publishing:

- [ ] Verify on PyPI: https://pypi.org/project/hypernodes/
- [ ] Test install: `uv pip install hypernodes`
- [ ] Check README renders correctly on PyPI
- [ ] Create GitHub release matching the tag
- [ ] Update documentation if needed
- [ ] Announce the release

## Security Best Practices

1. **Never commit tokens** to git
2. **Use environment variables** for tokens
3. **Rotate tokens** regularly
4. **Use scoped tokens** (per-project when possible)
5. **Test on TestPyPI** first, always

## Additional Resources

- [uv Documentation](https://docs.astral.sh/uv/)
- [uv GitHub Repository](https://github.com/astral-sh/uv)
- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)

---

**Ready to publish?** 

```bash
uv build
uv publish --token pypi-YOUR_TOKEN
```

That's it! 🎉
