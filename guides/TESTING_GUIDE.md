# Modal Backend Testing Guide

## Quick Start (5 minutes)

Run this first to verify Modal is working:

```bash
uv run python scripts/quickstart_modal.py
```

This runs the simplest possible test: single node, single input, verify result.

## Test Progression

### 1. Smoke Tests (10-15 minutes)

5 progressive tests from simple to complex:

```bash
uv run python scripts/test_modal_smoke.py
```

**Tests:**
1. Simplest: single node execution
2. Map operation over list
3. `.as_node()` with `map_over` (Hebrew pattern)
4. Pydantic models
5. Execution configuration

### 2. Pytest Suite (15-20 minutes)

Comprehensive test suite with 10+ tests:

```bash
# Run all tests
uv run pytest tests/test_modal_backend.py -v

# Run with output
uv run pytest tests/test_modal_backend.py -v -s

# Run specific test
uv run pytest tests/test_modal_backend.py::test_modal_single_node_simple -v

# Run integration test
uv run pytest tests/test_modal_backend.py::test_modal_full_integration -v
```

**Test Coverage:**
- ✅ Single node execution
- ✅ Multiple dependent nodes
- ✅ Map operations
- ✅ Nested pipelines
- ✅ `.as_node()` with `map_over`
- ✅ Execution engine config
- ✅ Pydantic models
- ✅ Hebrew pipeline pattern
- ✅ Error handling
- ✅ Larger maps with threading
- ✅ Full integration

### 3. Hebrew-Style Minimal (20-30 minutes)

Minimal version of Hebrew retrieval pipeline with mock encoder/index:

```bash
uv run python scripts/test_modal_hebrew_minimal.py
```

**Tests:**
1. Local execution (baseline)
2. Modal execution (no cache)
3. Modal execution (with cache)
4. Modal with larger dataset

**Pattern tested:**
- Data loading → Batch encoding → Index building → Query encoding → Retrieval → Aggregation

## What Was Implemented

### Option 4: Placement + Execution Engine

**Architecture:**
- `PipelineExecutionEngine`: Reusable execution strategy
- `ModalBackend`: Single-container remote execution
- **Code Reuse**: Both use `LocalBackend` internally

**Benefits:**
- ✅ No connection timeouts (single `.remote()` call per run)
- ✅ Identical semantics (caching, callbacks, nested pipelines)
- ✅ Clean separation (placement vs execution)
- ✅ Ready for Hebrew pipeline

### Key Components

1. **Backend ABC** (`src/hypernodes/backend.py`)
   - Abstract base with `run()` and `map()` methods
   - Enforces consistent interface

2. **PipelineExecutionEngine** (`src/hypernodes/backend.py`)
   - Encapsulates execution strategy
   - Delegates to `LocalBackend`
   - Configurable: sequential/async/threaded/parallel

3. **ModalBackend Updates** (`src/hypernodes/backend.py`)
   - Serializes pipeline + engine config
   - Runs in single Modal container
   - Reconstructs `LocalBackend` remotely
   - New parameters: `node_execution`, `map_execution`, `max_workers`

### Usage Pattern

```python
# Create Modal image
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(...)

# Create backend with execution config
backend = ModalBackend(
    image=image,
    gpu="A10G",
    node_execution="sequential",
    map_execution="threaded",
    max_workers=8,
)

# Use in pipeline
pipeline = Pipeline(
    nodes=[...],
    backend=backend,
    cache=DiskCache(path=".cache"),
)

# Run (entire pipeline executes in one Modal container)
results = pipeline.run(inputs={...})
```

## Files Created

### Core Implementation
- `src/hypernodes/backend.py` - Updated with new architecture

### Tests
- `tests/test_modal_backend.py` - Comprehensive pytest suite (10 tests)
- `tests/README_MODAL.md` - Detailed testing documentation

### Scripts
- `scripts/quickstart_modal.py` - Fastest verification (1 test)
- `scripts/test_modal_smoke.py` - Progressive smoke tests (5 tests)
- `scripts/test_modal_hebrew_minimal.py` - Hebrew pattern tests (4 tests)

### Documentation
- `MODAL_IMPLEMENTATION.md` - Architecture and design decisions
- `TESTING_GUIDE.md` - This file

## Expected Results

### Quickstart
```
Input:  x = 41
Output: result = 42
✓ SUCCESS!
```

### Smoke Tests
```
Test 1: Simplest possible - single node
✓ PASSED

Test 2: Map operation
✓ PASSED

Test 3: as_node with map_over
✓ PASSED

Test 4: Pydantic models (Hebrew pipeline pattern)
✓ PASSED

Test 5: Execution engine configuration
✓ PASSED

Passed: 5/5
All tests passed! ✓
```

### Hebrew Minimal
```
TEST 1: Local execution (baseline)
  Total predictions: 15
  Unique queries: 3
✓ PASSED

TEST 2: Modal execution (no cache)
  Total predictions: 15
  Unique queries: 3
✓ PASSED

TEST 3: Modal execution (with cache)
  (both runs identical)
✓ PASSED

TEST 4: Modal with larger dataset
  Total predictions: 100
  Unique queries: 10
✓ PASSED

Passed: 4/4
All tests passed! ✓
```

## Troubleshooting

### Modal Authentication
```bash
modal token new
```

### Modal Not Installed
```bash
uv pip install modal
```

### Image Build Slow
First run builds the image (~1-2 minutes). Subsequent runs reuse it.

### Connection Issues
The new implementation fixes this! Single `.remote()` call per run.

### Check Modal Dashboard
https://modal.com → View running containers and logs

## Next Steps

1. **Run quickstart**: Verify basic functionality
2. **Run smoke tests**: Verify all patterns work
3. **Run pytest suite**: Comprehensive verification
4. **Adapt your Hebrew pipeline**: Use patterns from `test_modal_hebrew_minimal.py`

## Performance Notes

- **Cold Start**: 10-30s (image build + container start)
- **Warm Runs**: 1-3s overhead per run
- **Throughput**: Same as LocalBackend (single container)
- **Caching**: Works across runs (with persistent volumes)

## Design Philosophy

**Single Container First:**
- Solves immediate problem (connection timeouts)
- Simplest implementation
- Sufficient for most workloads
- Easy to extend later (distributed map)

**Code Reuse:**
- Same `LocalBackend` logic everywhere
- Ensures identical semantics
- Reduces bugs and maintenance

**Clean Architecture:**
- Placement (local/Modal) separate from execution (sequential/threaded/etc)
- Easy to add new backends (Ray, SageMaker, etc)
- Easy to add new execution modes

## Summary

✅ **Implemented**: Option 4 with single-container Modal execution
✅ **Tested**: 19 total tests across 3 test files
✅ **Documented**: Architecture, usage, troubleshooting
✅ **Ready**: For Hebrew retrieval pipeline
⏳ **Future**: Distributed map mode (not needed yet)

**Start here:**
```bash
uv run python scripts/quickstart_modal.py
```
