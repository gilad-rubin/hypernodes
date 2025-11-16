# Quick Start: Publishing HyperNodes to PyPI

## Prerequisites Setup
```bash
# Install publishing tools
pip install build twine

# Get PyPI API token from https://pypi.org/manage/account/token/
# Get TestPyPI token from https://test.pypi.org/manage/account/token/
```

## Publishing Workflow

### Step 1: Build the Package
```bash
cd /workspace
python3 -m build
```

### Step 2: Verify Build
```bash
# Check package with twine
python3 -m twine check dist/*

# Should output:
# Checking dist/hypernodes-0.1.0-py3-none-any.whl: PASSED
# Checking dist/hypernodes-0.1.0.tar.gz: PASSED
```

### Step 3: Test on TestPyPI (RECOMMENDED)
```bash
# Upload to TestPyPI
python3 -m twine upload --repository testpypi dist/*

# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ hypernodes

# Verify it works
python3 -c "from hypernodes import Pipeline, node; print('Success!')"
```

### Step 4: Publish to PyPI
```bash
# Upload to production PyPI
python3 -m twine upload dist/*

# Verify on PyPI
# Visit: https://pypi.org/project/hypernodes/

# Test install
pip install hypernodes
```

## Configuration File (~/.pypirc)

Create `~/.pypirc` with your API tokens:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_PYPI_TOKEN_HERE

[testpypi]
username = __token__
password = pypi-YOUR_TESTPYPI_TOKEN_HERE
```

**Important:** Set proper permissions:
```bash
chmod 600 ~/.pypirc
```

## Future Releases

For each new version:

```bash
# 1. Update version in pyproject.toml
# 2. Update CHANGELOG.md
# 3. Commit changes
git add pyproject.toml CHANGELOG.md
git commit -m "Release v0.2.0"

# 4. Tag the release
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0

# 5. Clean and rebuild
rm -rf dist/
python3 -m build

# 6. Upload to PyPI
python3 -m twine upload dist/*
```

## Troubleshooting

**Problem:** "File already exists on PyPI"
**Solution:** You can't overwrite. Increment version in pyproject.toml

**Problem:** Import fails after install
**Solution:** Check optional dependencies. Install with extras:
```bash
pip install hypernodes[all]  # or [daft], [viz], [telemetry], etc.
```

**Problem:** Build fails
**Solution:** 
```bash
# Clean build artifacts
rm -rf build/ dist/ *.egg-info/
# Rebuild
python3 -m build
```

## Package Information

- **Name:** hypernodes
- **Current Version:** 0.1.0
- **License:** MIT
- **Python:** >=3.12
- **Repository:** https://github.com/gilad-rubin/hypernodes

## Getting Help

- **Issues:** https://github.com/gilad-rubin/hypernodes/issues
- **PyPI Help:** https://pypi.org/help/
- **Packaging Guide:** https://packaging.python.org/

---

**Ready to publish?** Run: `python3 -m twine upload dist/*`
