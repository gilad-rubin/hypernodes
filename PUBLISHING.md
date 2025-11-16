# Publishing to PyPI

This document describes how to publish the HyperNodes package to PyPI.

## Prerequisites

1. Install build tools:
```bash
pip install build twine
```

2. Create accounts on:
   - [PyPI](https://pypi.org/) (production)
   - [TestPyPI](https://test.pypi.org/) (testing)

3. Set up API tokens:
   - Create API tokens in your PyPI account settings
   - Store them securely (e.g., in `~/.pypirc`)

## Building the Package

Build the distribution packages:

```bash
python -m build
```

This creates:
- `dist/hypernodes-X.Y.Z-py3-none-any.whl` (wheel)
- `dist/hypernodes-X.Y.Z.tar.gz` (source distribution)

## Testing Before Publishing

### 1. Test the build locally

Install the package locally from the wheel:

```bash
pip install dist/hypernodes-*.whl
```

Test importing and basic functionality:

```python
from hypernodes import Pipeline, node

@node(output_name="result")
def test_node(x: int) -> int:
    return x * 2

pipeline = Pipeline(nodes=[test_node])
result = pipeline.run(inputs={"x": 5})
print(result)  # Should print: {'result': 10}
```

### 2. Upload to TestPyPI (recommended)

First, test on TestPyPI:

```bash
python -m twine upload --repository testpypi dist/*
```

Then test installing from TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ hypernodes
```

## Publishing to PyPI

Once you've verified everything works:

```bash
python -m twine upload dist/*
```

You'll be prompted for your PyPI credentials or API token.

## Using API Tokens

Create a `~/.pypirc` file:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-...  # Your PyPI API token

[testpypi]
username = __token__
password = pypi-...  # Your TestPyPI API token
```

## Versioning

Before publishing a new version:

1. Update the version in `pyproject.toml`
2. Update `CHANGELOG.md` with the new version and changes
3. Commit the changes
4. Tag the release: `git tag -a v0.1.0 -m "Release v0.1.0"`
5. Push the tag: `git push origin v0.1.0`

## Post-Publishing Checklist

- [ ] Verify the package appears on PyPI: https://pypi.org/project/hypernodes/
- [ ] Test installing with `pip install hypernodes`
- [ ] Check the package description renders correctly on PyPI
- [ ] Create a GitHub release matching the version tag
- [ ] Announce the release

## Troubleshooting

### Build fails

- Ensure all files are properly formatted
- Check that `src/hypernodes/__init__.py` exists and is valid
- Verify `pyproject.toml` syntax is correct

### Upload fails

- Check your API token is valid
- Ensure you're not trying to upload the same version twice
- For TestPyPI, ensure you're using the correct repository URL

### Package doesn't import correctly

- Verify the package structure with `python -m zipfile -l dist/*.whl`
- Check that all `__init__.py` files exist
- Ensure optional dependencies are properly marked

## Additional Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [Hatchling Documentation](https://hatch.pypa.io/latest/)
- [PyPI Help](https://pypi.org/help/)
