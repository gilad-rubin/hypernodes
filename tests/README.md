# Hypernodes Test Suite

## Overview

This folder contains the refined test suite for Hypernodes with the new `SequentialEngine` architecture.

All tests use the default `SequentialEngine` (no need to specify engine parameter).

---

## Test Files

### **Core Functionality**

#### `test_construction.py`
- Basic node and pipeline construction
- PipelineNode creation
- Parameter and output name handling

#### `test_execution.py`
- Single node execution
- Sequential node chains
- Diamond dependencies
- Multiple outputs
- Selective output filtering

#### `test_map.py`
- Basic map operations
- Map with fixed parameters
- Zip mode (parallel iteration)
- Product mode (all combinations)
- Map over sequential nodes
- Selective output in maps

#### `test_nested_pipelines.py`
- Simple nested pipelines
- Multi-level nesting
- Input/output mapping
- Internal mapping with `map_over`
- Namespace collision avoidance
- Config inheritance

#### `test_caching.py`
- Basic caching behavior
- Cache invalidation on input change
- Selective caching (`cache=False`)
- Nested pipeline cache inheritance
- Caching with map operations

#### `test_callbacks.py`
- Basic callback events
- Multiple node callbacks
- Map operation callbacks
- Nested pipeline callback inheritance
- Multiple callback registration

### **Visualization**

#### `test_visualization_depth.py`
- Visualization depth control
- Nested pipeline expansion

#### `test_visualization_graphviz_io.py`
- Graphviz rendering
- SVG/PNG export

---

## Running Tests

### Run All Tests
```bash
uv run pytest tests/
```

### Run Specific Test File
```bash
uv run pytest tests/test_execution.py -v
```

### Run Specific Test
```bash
uv run pytest tests/test_execution.py::test_single_node_pipeline -v
```

### Run with Verbose Output
```bash
uv run pytest tests/test_nested_pipelines.py -xvs
```

### Run Tests Matching Pattern
```bash
uv run pytest tests/ -k "cache" -v
```

---

## Test Organization

### **Naming Convention**
- `test_<category>.py` - Test file for a specific category
- `test_<feature>_<detail>` - Individual test function

### **Test Structure**
Each test follows this pattern:
1. **Setup**: Create nodes and pipelines
2. **Execute**: Run or map the pipeline
3. **Assert**: Verify expected outputs

### **Key Principles**
- ✅ **No engine specified** - uses `SequentialEngine` by default
- ✅ **Self-contained** - each test is independent
- ✅ **Clear intent** - test names describe what they test
- ✅ **Minimal** - tests focus on one thing

---

## Configuration Inheritance

Tests demonstrate that nested pipelines inherit parent configuration:

```python
from hypernodes import DiskCache

inner = Pipeline(nodes=[...])  # No config

outer = Pipeline(
    nodes=[inner.as_node()],
    cache=DiskCache(path=".cache"),  # Inner inherits this
    callbacks=[callback],             # Inner inherits this
)
```

---

## Old Tests

The `tests/old/` folder contains the original test suite. These tests are preserved for reference but may use outdated patterns (e.g., bare `Pipeline` objects, explicit executor specifications).

**Do not use these as examples.** Use the tests in the main `tests/` folder instead.

---

## Adding New Tests

When adding new tests:

1. **Choose the right file** based on category
2. **Use descriptive names** - `test_<what>_<scenario>`
3. **Keep it simple** - one concept per test
4. **Use default engine** - don't specify unless testing specific engine
5. **Document complex scenarios** - add docstring explaining what's tested

Example:
```python
def test_map_with_selective_output():
    """Test that map respects output_name filtering.
    
    This verifies that when output_name is specified in map(),
    only those outputs are included in the result dictionaries.
    """
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    pipeline = Pipeline(nodes=[double])
    results = pipeline.map(
        inputs={"x": [1, 2, 3]},
        map_over="x",
        output_name="doubled"
    )
    
    assert all("doubled" in r for r in results)
```

---

## Quick Reference

### Run Fast Tests
```bash
uv run pytest tests/test_execution.py tests/test_construction.py -v
```

### Run All Except Slow Tests
```bash
uv run pytest tests/ -v --ignore=tests/old/
```

### Run Only Nested Pipeline Tests
```bash
uv run pytest tests/test_nested_pipelines.py -v
```

### Run Caching Tests
```bash
uv run pytest tests/test_caching.py -v
```

---

## Test Coverage

Current test coverage by feature:

- ✅ **Basic Execution**: 6 tests
- ✅ **Map Operations**: 7 tests
- ✅ **Nested Pipelines**: 9 tests
- ✅ **Caching**: 6 tests
- ✅ **Callbacks**: 5 tests
- ✅ **Visualization**: 2 tests

**Total: 35+ tests**

---

## CI/CD Integration

To integrate with CI/CD, add to your workflow:

```yaml
- name: Run tests
  run: |
    uv run pytest tests/ -v --ignore=tests/old/
```

---

## Debugging Failed Tests

If a test fails:

1. **Run with `-xvs`** to see detailed output and stop on first failure
2. **Check the error message** - usually very clear about what's wrong
3. **Verify your changes** didn't break config inheritance or mapping
4. **Check for typos** in node names, parameter names, or mappings

```bash
uv run pytest tests/test_nested_pipelines.py::test_simple_nested_pipeline -xvs
```

---

**Happy Testing! 🎉**

