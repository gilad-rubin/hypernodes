# Backend → Engine Architecture Migration Plan

## Overview

This document outlines the phased migration from the current `Backend` architecture to the new `Engine` architecture as defined in [EXECUTOR_ARCHITECTURE_FINAL.md](EXECUTOR_ARCHITECTURE_FINAL.md).

### Key Changes
- `Backend` → `Engine` (terminology change for orchestrators)
- `LocalExecutor` → `HyperNodesEngine` (clearer naming)
- `DaftExecutor` → `DaftEngine` (Daft is an engine, not just an executor)
- `Pipeline(backend=...)` → `Pipeline(engine=...)`
- Add `AsyncExecutor` for I/O-bound work
- String aliases for common patterns: `"sequential"`, `"async"`, `"threaded"`, `"parallel"`
- Engines accept `node_executor` and `map_executor` (not `node_execution` and `map_execution`)

### No Backwards Compatibility
This is a breaking change. All code will be updated to use the new API.

---

## Current Codebase Structure

### Modules Under `src/hypernodes/`
```
backend.py                    # Contains Backend ABC and LocalBackend
daft_backend.py              # Contains DaftBackend
executors/
  ├── base.py                # Contains Executor ABC
  ├── local.py               # Contains LocalExecutor and PipelineExecutionEngine
  └── daft.py                # Contains DaftExecutor
pipeline.py                  # Pipeline class (uses backend parameter)
node.py                      # Node decorator and class
cache.py                     # Caching system
callbacks.py                 # Callback system
visualization.py             # Pipeline visualization
telemetry/
  ├── tracing.py            # Distributed tracing
  ├── progress.py           # Progress bars
  └── waterfall.py          # Waterfall visualizations
```

### Test Files Under `tests/`
```
test_phase1_core_execution.py          # Basic pipeline execution
test_phase2_map_operations.py          # Map operations
test_phase3_caching.py                 # Caching behavior
test_phase3_class_caching.py           # Class-based caching
test_phase4_callbacks.py               # Callback system
test_phase5_nested_pipelines.py        # Nested pipeline composition
test_selective_output.py               # Selective execution
test_telemetry_basic.py                # Telemetry integration
test_daft_backend.py                   # Daft backend
test_daft_backend_map_over.py          # Daft map operations
test_daft_backend_complex_types.py     # Daft complex types
test_modal_backend.py                  # Modal backend (REMOVE)
test_modal_map.py                      # Modal map (REMOVE)
test_modal_torch_cpu.py                # Modal torch (REMOVE)
test_executor_performance.py           # NEW: Executor performance tests
```

### Scripts Under `scripts/`
```
benchmark_hypernodes_vs_daft.py       # Performance benchmarks
debug_daft_pipeline.py                # Daft debugging
retrieval_daft_fixed.py               # Daft retrieval example
retrieval_daft_working_example.py     # Daft working example
retrieval_elegant_pydantic.py         # Pydantic example
test_as_node_progress.py              # Progress testing
test_cache_encoder.py                 # Cache encoder testing
test_cache_issue.py                   # Cache debugging
diagnose_modal_issue.py               # Modal debugging (REMOVE)
fix_modal_hebrew_example.py           # Modal example (REMOVE)
quickstart_modal.py                   # Modal quickstart (REMOVE)
+ More scripts...
```

---

## Migration Phases

## Phase 1: Core Refactoring ✅ (ALREADY DONE)

**Status**: ✅ Complete

**What Was Done**:
- Created `executors/base.py` with `Executor` ABC
- Created `executors/local.py` with `LocalExecutor`
- Created `executors/daft.py` with `DaftExecutor`
- Removed Modal executor (guidance-only via `guides/modal_functions.md`)

**Files Modified**:
- ✅ `src/hypernodes/executors/base.py` - Created
- ✅ `src/hypernodes/executors/local.py` - Created
- ✅ `src/hypernodes/executors/daft.py` - Created
- ✅ `guides/modal_functions.md` - Created (guidance only)

---

## Phase 2: Rename to Engine Architecture

**Goal**: Rename `Backend` → `Engine` and update terminology throughout

**Changes**:

### 2.1. Rename Base Classes

**File**: `src/hypernodes/executors/base.py`
- [ ] Rename `Executor` → `Engine`
- [ ] Update docstrings to reflect "Engine" terminology
- [ ] Update `_ctx` parameter to indicate it's internal-only
- [ ] Update method signatures:
  - `run(pipeline, inputs, output_name, _ctx)` - prefix `_ctx` with underscore
  - `map(pipeline, items, inputs, output_name, _ctx)` - prefix `_ctx` with underscore

**File**: `src/hypernodes/executors/local.py`
- [ ] Rename `LocalExecutor` → `HyperNodesEngine`
- [ ] Remove `PipelineExecutionEngine` class (no longer needed)
- [ ] Update `__init__` parameters:
  - `node_execution` → `node_executor` (accepts executor instances or strings)
  - `map_execution` → `map_executor` (accepts executor instances or strings)
  - Remove `max_workers` (built into executor instances)
  - Remove `executor` parameter (replaced by node_executor/map_executor)
- [ ] Update docstrings to reflect new parameter names
- [ ] Inherit from `Engine` (not `BaseExecutor`)

**File**: `src/hypernodes/executors/daft.py`
- [ ] Rename `DaftExecutor` → `DaftEngine`
- [ ] Inherit from `Engine` (not `BaseExecutor`)
- [ ] Update docstrings

**Test**: Create `tests/test_phase2_engine_renaming.py`
```python
def test_engine_base_class():
    """Verify Engine ABC works correctly."""

def test_hypernodes_engine_creation():
    """Verify HyperNodesEngine can be created."""

def test_daft_engine_creation():
    """Verify DaftEngine can be created."""
```

---

## Phase 3: AsyncExecutor Implementation

**Goal**: Create AsyncExecutor for I/O-bound concurrent work

**Changes**:

**File**: `src/hypernodes/executors/async_executor.py` (NEW)
- [ ] Create `AsyncExecutor` class
- [ ] Implement `concurrent.futures.Executor` protocol
- [ ] Methods:
  - `__init__(max_concurrent: int = 10)`
  - `submit(fn: Callable, *args, **kwargs) -> Future`
  - `shutdown(wait: bool = True)`
- [ ] Handle Jupyter event loop compatibility:
  - Detect running event loop
  - Reuse existing loop
  - Use `nest_asyncio` if needed
- [ ] Support both sync and async callables
- [ ] Return `concurrent.futures.Future` for compatibility

**File**: `src/hypernodes/executors/__init__.py`
- [ ] Export `AsyncExecutor`
- [ ] Export `HyperNodesEngine` (not `LocalExecutor`)
- [ ] Export `DaftEngine` (not `DaftExecutor`)
- [ ] Export `Engine` base class

**Test**: Update `tests/test_executor_performance.py`
- [x] Test async map operations (ALREADY EXISTS)
- [x] Test async node operations (ALREADY EXISTS)
- [ ] Test Jupyter compatibility (needs manual verification)
- [ ] Test mixed sync/async callables

---

## Phase 4: String Alias Support in HyperNodesEngine

**Goal**: Support string aliases for common executor patterns

**Changes**:

**File**: `src/hypernodes/executors/local.py`
- [ ] Update `HyperNodesEngine.__init__` to accept:
  - `node_executor: Union[Executor, Literal["sequential", "async", "threaded", "parallel"], None]`
  - `map_executor: Union[Executor, Literal["sequential", "async", "threaded", "parallel"], None]`
- [ ] Implement `_resolve_executor()` method to convert strings to executors:
  - `"sequential"` → `None` (sync execution, no executor)
  - `"async"` → `AsyncExecutor(max_concurrent=10)`
  - `"threaded"` → `ThreadPoolExecutor(max_workers=os.cpu_count() or 4)`
  - `"parallel"` → `ProcessPoolExecutor(max_workers=os.cpu_count() or 4)`
- [ ] Store resolved executors as `self._node_executor` and `self._map_executor`
- [ ] Update `run()` and `map()` to use resolved executors

**Test**: Update `tests/test_executor_performance.py`
- [x] Test string alias "sequential" (ALREADY EXISTS)
- [x] Test string alias "async" (ALREADY EXISTS)
- [x] Test string alias "threaded" (ALREADY EXISTS)
- [x] Test string alias "parallel" (ALREADY EXISTS)
- [ ] Test executor instance passing still works

---

## Phase 5: Pipeline Integration

**Goal**: Update Pipeline class to use `engine` parameter

**Changes**:

**File**: `src/hypernodes/pipeline.py`
- [ ] Update `__init__` signature:
  - `backend: Optional[Backend]` → `engine: Optional[Engine]`
  - Update default to `HyperNodesEngine()` (not `LocalBackend()`)
- [ ] Update `effective_backend` property → `effective_engine` property
- [ ] Update `run()` method to call `engine.run()` with `_ctx` parameter
- [ ] Update `map()` method:
  - Keep existing signature: `map(inputs, map_over, map_mode, output_name, _ctx)`
  - Convert `inputs` + `map_over` + `map_mode` → `items` for engine
  - Call `engine.map()` with converted items
  - Convert engine output back to expected format
- [ ] Update nested pipeline inheritance to use `engine`
- [ ] Update `PipelineNode` to work with engine

**File**: `src/hypernodes/__init__.py`
- [ ] Export `Engine` instead of `Backend`
- [ ] Export `HyperNodesEngine` instead of `LocalBackend`
- [ ] Export `DaftEngine` instead of `DaftBackend`
- [ ] Export `AsyncExecutor`
- [ ] Remove `LocalBackend` and `Backend` exports

**Test**: Create `tests/test_phase5_pipeline_integration.py`
```python
def test_pipeline_with_engine_parameter():
    """Verify Pipeline accepts engine parameter."""

def test_pipeline_default_engine():
    """Verify Pipeline defaults to HyperNodesEngine."""

def test_pipeline_effective_engine():
    """Verify effective_engine property works."""

def test_nested_pipeline_engine_inheritance():
    """Verify nested pipelines inherit engine correctly."""
```

---

## Phase 6: Deprecate Old Backend Classes

**Goal**: Remove old Backend classes and migrate all references

**Changes**:

**File**: `src/hypernodes/backend.py` (DELETE or DEPRECATE)
- [ ] Option A: Delete entire file
- [ ] Option B: Add deprecation warnings that redirect to new classes:
  ```python
  class Backend(Engine):
      def __init__(self, *args, **kwargs):
          warnings.warn("Backend is deprecated, use Engine instead", DeprecationWarning)
          super().__init__(*args, **kwargs)
  ```

**File**: `src/hypernodes/daft_backend.py` (DELETE)
- [ ] Delete file (functionality moved to `executors/daft.py`)

**Test**: No new tests needed (old functionality removed)

---

## Phase 7: Update All Existing Tests

**Goal**: Update all existing tests to use new API

**Changes**:

### 7.1. Core Execution Tests

**File**: `tests/test_phase1_core_execution.py`
- [ ] Replace `LocalBackend` → `HyperNodesEngine`
- [ ] Replace `backend=` → `engine=`
- [ ] Update imports
- [ ] Verify all tests pass

**Test Count**: ~15-20 tests to update

### 7.2. Map Operation Tests

**File**: `tests/test_phase2_map_operations.py`
- [ ] Replace `LocalBackend` → `HyperNodesEngine`
- [ ] Replace `backend=` → `engine=`
- [ ] Verify map operations work with new engine
- [ ] Verify `map_mode="zip"` and `map_mode="product"` work

**Test Count**: ~10-15 tests to update

### 7.3. Caching Tests

**File**: `tests/test_phase3_caching.py`
- [ ] Replace `LocalBackend` → `HyperNodesEngine`
- [ ] Replace `backend=` → `engine=`
- [ ] Verify caching behavior unchanged

**File**: `tests/test_phase3_class_caching.py`
- [ ] Replace `LocalBackend` → `HyperNodesEngine`
- [ ] Replace `backend=` → `engine=`
- [ ] Verify class-based caching unchanged

**Test Count**: ~20-25 tests to update

### 7.4. Callback Tests

**File**: `tests/test_phase4_callbacks.py`
- [ ] Replace `LocalBackend` → `HyperNodesEngine`
- [ ] Replace `backend=` → `engine=`
- [ ] Verify callbacks fire correctly
- [ ] Verify callback context is correct

**Test Count**: ~10-15 tests to update

### 7.5. Nested Pipeline Tests

**File**: `tests/test_phase5_nested_pipelines.py`
- [ ] Replace `LocalBackend` → `HyperNodesEngine`
- [ ] Replace `backend=` → `engine=`
- [ ] Verify nested pipeline inheritance works
- [ ] Verify `as_node()` works correctly

**Test Count**: ~10-12 tests to update

### 7.6. Selective Output Tests

**File**: `tests/test_selective_output.py`
- [ ] Replace `LocalBackend` → `HyperNodesEngine`
- [ ] Replace `backend=` → `engine=`
- [ ] Verify selective execution still works

**Test Count**: ~5-8 tests to update

### 7.7. Telemetry Tests

**File**: `tests/test_telemetry_basic.py`
- [ ] Replace `LocalBackend` → `HyperNodesEngine`
- [ ] Replace `backend=` → `engine=`
- [ ] Verify telemetry integration works

**Test Count**: ~5-10 tests to update

### 7.8. Daft Tests

**File**: `tests/test_daft_backend.py`
- [ ] Replace `DaftBackend` → `DaftEngine`
- [ ] Replace `backend=` → `engine=`
- [ ] Update imports

**File**: `tests/test_daft_backend_map_over.py`
- [ ] Replace `DaftBackend` → `DaftEngine`
- [ ] Replace `backend=` → `engine=`

**File**: `tests/test_daft_backend_complex_types.py`
- [ ] Replace `DaftBackend` → `DaftEngine`
- [ ] Replace `backend=` → `engine=`

**Test Count**: ~15-20 tests to update

### 7.9. Modal Tests (REMOVE)

**File**: `tests/test_modal_backend.py` (DELETE)
- [ ] Delete file
- [ ] Modal is now guidance-only

**File**: `tests/test_modal_map.py` (DELETE)
- [ ] Delete file

**File**: `tests/test_modal_torch_cpu.py` (DELETE)
- [ ] Delete file

---

## Phase 8: Update Scripts

**Goal**: Update all scripts to use new API

**Changes**:

### 8.1. Daft Scripts

**File**: `scripts/benchmark_hypernodes_vs_daft.py`
- [ ] Replace `DaftBackend` → `DaftEngine`
- [ ] Replace `LocalBackend` → `HyperNodesEngine`
- [ ] Replace `backend=` → `engine=`

**File**: `scripts/debug_daft_pipeline.py`
- [ ] Replace `DaftBackend` → `DaftEngine`
- [ ] Replace `backend=` → `engine=`

**File**: `scripts/retrieval_daft_fixed.py`
- [ ] Replace `DaftBackend` → `DaftEngine`
- [ ] Replace `backend=` → `engine=`

**File**: `scripts/retrieval_daft_working_example.py`
- [ ] Replace `DaftBackend` → `DaftEngine`
- [ ] Replace `backend=` → `engine=`

**File**: `scripts/retrieval_elegant_pydantic.py`
- [ ] Update if it uses backend parameter

### 8.2. Testing Scripts

**File**: `scripts/test_as_node_progress.py`
- [ ] Replace `LocalBackend` → `HyperNodesEngine`
- [ ] Replace `backend=` → `engine=`

**File**: `scripts/test_cache_encoder.py`
- [ ] Replace `LocalBackend` → `HyperNodesEngine`
- [ ] Replace `backend=` → `engine=`

**File**: `scripts/test_cache_issue.py`
- [ ] Replace `LocalBackend` → `HyperNodesEngine`
- [ ] Replace `backend=` → `engine=`

### 8.3. Modal Scripts (REMOVE)

**File**: `scripts/diagnose_modal_issue.py` (DELETE)
- [ ] Delete file

**File**: `scripts/fix_modal_hebrew_example.py` (DELETE)
- [ ] Delete file

**File**: `scripts/quickstart_modal.py` (DELETE)
- [ ] Delete file

**Estimated Scripts to Update**: ~8-10 scripts

---

## Phase 9: Documentation Updates

**Goal**: Update all documentation to reflect new architecture

**Changes**:

### 9.1. Main Documentation

**File**: `README.md`
- [ ] Update quick start examples to use `engine=`
- [ ] Update API examples
- [ ] Update executor examples
- [ ] Add AsyncExecutor examples
- [ ] Update terminology (Backend → Engine)

**File**: `docs/` (if exists)
- [ ] Update all documentation files
- [ ] Update API reference
- [ ] Update tutorials
- [ ] Add migration guide

### 9.2. Code Examples

**File**: `examples/` (if exists)
- [ ] Update all example files
- [ ] Replace `backend=` → `engine=`
- [ ] Add async executor examples
- [ ] Add string alias examples

### 9.3. Inline Documentation

**Files**: All Python files
- [ ] Update docstrings to use Engine terminology
- [ ] Update code comments
- [ ] Update type hints if needed

---

## Phase 10: Final Testing and Validation

**Goal**: Ensure everything works together

**Test Plan**:

### 10.1. Run Full Test Suite
```bash
uv run pytest tests/ -v
```
- [ ] All tests pass
- [ ] No deprecated warnings
- [ ] Performance tests show expected speedups

### 10.2. Run All Scripts
```bash
# Daft scripts
uv run python scripts/benchmark_hypernodes_vs_daft.py
uv run python scripts/debug_daft_pipeline.py
uv run python scripts/retrieval_daft_fixed.py
uv run python scripts/retrieval_daft_working_example.py

# Testing scripts
uv run python scripts/test_as_node_progress.py
uv run python scripts/test_cache_encoder.py
uv run python scripts/test_cache_issue.py
```
- [ ] All scripts run without errors
- [ ] Results are as expected

### 10.3. Performance Validation
```bash
uv run pytest tests/test_executor_performance.py -v -s
```
- [ ] Async executor shows speedup for I/O-bound work
- [ ] Threaded executor shows speedup for blocking I/O
- [ ] Parallel executor shows speedup for CPU-bound work
- [ ] Sequential executor baseline works correctly

### 10.4. Integration Testing

Create `tests/test_final_integration.py`:
```python
def test_complete_pipeline_workflow():
    """End-to-end test with all features."""

def test_nested_pipelines_with_different_engines():
    """Verify nested pipelines work with different engines."""

def test_mixed_executors():
    """Verify mixing different executor types works."""

def test_caching_with_new_engines():
    """Verify caching works correctly with new engines."""

def test_callbacks_with_new_engines():
    """Verify callbacks work correctly with new engines."""
```

---

## Summary

### Files to Create
1. ✅ `tests/test_executor_performance.py` - Performance validation tests
2. `src/hypernodes/executors/async_executor.py` - AsyncExecutor implementation
3. `tests/test_phase2_engine_renaming.py` - Engine renaming tests
4. `tests/test_phase5_pipeline_integration.py` - Pipeline integration tests
5. `tests/test_final_integration.py` - Final integration tests

### Files to Modify
1. `src/hypernodes/executors/base.py` - Rename Executor → Engine
2. `src/hypernodes/executors/local.py` - LocalExecutor → HyperNodesEngine + string aliases
3. `src/hypernodes/executors/daft.py` - DaftExecutor → DaftEngine
4. `src/hypernodes/executors/__init__.py` - Update exports
5. `src/hypernodes/pipeline.py` - backend → engine parameter
6. `src/hypernodes/__init__.py` - Update exports
7. All test files (~10-15 files)
8. All scripts (~8-10 files)
9. `README.md` and documentation

### Files to Delete
1. `src/hypernodes/backend.py` - Old Backend class
2. `src/hypernodes/daft_backend.py` - Old DaftBackend class
3. `tests/test_modal_backend.py` - Modal tests
4. `tests/test_modal_map.py` - Modal tests
5. `tests/test_modal_torch_cpu.py` - Modal tests
6. `scripts/diagnose_modal_issue.py` - Modal scripts
7. `scripts/fix_modal_hebrew_example.py` - Modal scripts
8. `scripts/quickstart_modal.py` - Modal scripts

### Estimated Effort

| Phase | Effort | Priority |
|-------|--------|----------|
| Phase 1 | ✅ Done | - |
| Phase 2 | 2-3 hours | HIGH |
| Phase 3 | 3-4 hours | HIGH |
| Phase 4 | 2-3 hours | HIGH |
| Phase 5 | 3-4 hours | HIGH |
| Phase 6 | 1 hour | MEDIUM |
| Phase 7 | 4-6 hours | HIGH |
| Phase 8 | 2-3 hours | MEDIUM |
| Phase 9 | 3-4 hours | MEDIUM |
| Phase 10 | 2-3 hours | HIGH |
| **Total** | **22-33 hours** | - |

### Success Criteria

✅ All tests pass
✅ All scripts run successfully
✅ Performance tests show expected speedups
✅ No deprecated code remains
✅ Documentation is complete and accurate
✅ Caching behavior unchanged
✅ Callback behavior unchanged
✅ Nested pipelines work correctly
✅ String aliases work as expected
✅ AsyncExecutor works in Jupyter notebooks

---

## Next Steps

1. **Review this plan** - Ensure all stakeholders agree
2. **Start Phase 2** - Begin with core renaming
3. **Test incrementally** - Run tests after each phase
4. **Update documentation** - Keep docs in sync with code
5. **Celebrate!** - New architecture is cleaner and more powerful
