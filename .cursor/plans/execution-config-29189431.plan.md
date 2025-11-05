<!-- 29189431-5786-4139-b710-e4e5efbaab56 5e7f2026-7784-4d6c-bd52-9e0cf12a8cb2 -->
# Execution Configuration Architecture Implementation

## Overview

This refactor introduces a clean three-layer architecture for execution configuration in HyperNodes, separating infrastructure (WHERE) from execution strategy (HOW) with a unified configuration bundle (RuntimeConfig).

## Phase 1: Core Abstractions

### Create `src/hypernodes/config.py`

- `RuntimeConfig` class with fields: backend, executor, cache, callbacks, retries, timeout
- Immutable dataclass-style configuration object
- Support for lazy initialization of cache (CacheSpec or factory pattern)
- Support for lazy initialization of callbacks
- Context manager for temporary config overrides: `RuntimeConfig.use(config)`

### Create `src/hypernodes/executor.py`

- `Executor` abstract base class with methods:
- `run_pipeline(pipeline, inputs, ctx, output_name)` - execute DAG
- `map_items(pipeline, items, inputs, ctx, output_name)` - execute map operation
- Basic executors:
- `SequentialExecutor` - one node/item at a time
- `AsyncExecutor` - concurrent with asyncio
- `ThreadPoolExecutor` - wrapper around concurrent.futures.ThreadPoolExecutor
- `ProcessPoolExecutor` - wrapper around concurrent.futures.ProcessPoolExecutor

### Create global configuration management

- `_global_default_config` module variable
- `set_default_config(config: RuntimeConfig)` function
- `get_default_config() -> RuntimeConfig` function
- Default: `RuntimeConfig(backend=LocalBackend(), executor=ThreadPoolExecutor(max_workers=4))`

## Phase 2: Backend Refactoring

### Simplify `Backend` abstract base class

- Remove execution strategy logic (no more node_execution/map_execution)
- Keep only: `run(pipeline, inputs, ctx, output_name)` and `map(pipeline, items, inputs, ctx, output_name)`
- Backend now represents only WHERE code runs

### Refactor `LocalBackend`

- Remove: `node_execution`, `map_execution`, `max_workers`, `executor` parameters
- LocalBackend becomes trivial: just delegates to the executor from RuntimeConfig
- Implementation extracts executor from pipeline's effective_config
- Remove all `_run_sequential`, `_run_async`, `_run_threaded`, `_run_parallel` methods
- Remove all `_map_*` methods

### Keep `ModalBackend` mostly as-is

- Already represents WHERE (Modal cloud)
- Update to work with RuntimeConfig and Executor
- May need to serialize RuntimeConfig for remote execution
- Update internal execution to use PipelineExecutionEngine with executor

### Create `RemoteBackend` (optional for Phase 2)

- Generic remote execution backend
- Parameters: `kind` ("dask", "ray", "spark"), `address`, `security`
- Delegates to appropriate remote cluster execution

## Phase 3: DaftExecutor Implementation

### Rename `src/hypernodes/daft_backend.py` → `src/hypernodes/daft_executor.py`

- Change class name: `DaftBackend` → `DaftExecutor`
- Inherit from `Executor` base class
- Keep all existing Daft-specific logic (UDF conversion, DataFrame ops, etc.)
- Update methods to match Executor interface:
- `run_pipeline()` instead of `run()`
- `map_items()` instead of `map()`
- Keep parameters: `collect`, `show_plan`, `debug`, `runner`

### Add `DaskExecutor` (stub for future)

- Basic structure inheriting from Executor
- Parameters: `scheduler_address`, `scheduler` (threads/processes)
- Implementation delegates to Dask's delayed/futures API

### Add `RayExecutor` (stub for future)

- Basic structure inheriting from Executor
- Parameters: `ray_address`, `num_cpus`, `num_gpus`

## Phase 4: Advanced Executors

### Implement `CompositeExecutor`

- Parameters: `map_executor`, `node_executor`
- Delegates map operations to `map_executor`
- Delegates node execution to `node_executor`
- Enables patterns like: thread pool for map, async for nodes

### Extract `PipelineExecutionEngine` logic

- Move reusable execution logic from LocalBackend to standalone module
- Used by both LocalBackend and remote backends (Modal, etc.)
- Handles: topological sort, dependency resolution, caching, callbacks

## Phase 5: Pipeline API Changes

### Update `Pipeline` class

- **Constructor changes:**
- Remove: `backend`, `cache`, `callbacks` parameters
- Add: `config: Optional[RuntimeConfig] = None`
- Keep: `nodes`, `name`, `parent`

- **Property changes:**
- Replace: `effective_backend`, `effective_cache`, `effective_callbacks`
- Add: `effective_config` property (inherits from parent or uses global)

- **Method changes:**
- Update `run()` signature: add `config: Optional[RuntimeConfig] = None` parameter
- Update `map()` signature: add `config: Optional[RuntimeConfig] = None` parameter
- Add `with_config(config: RuntimeConfig) -> Pipeline` fluent method
- Remove: `with_backend()`, `with_cache()`, `with_callbacks()` (use `with_config` instead)

- **Configuration hierarchy logic:**
- Call-site config > Node config > Pipeline config > Global config > Library defaults
- Each level fully overrides (no partial merging for simplicity)

### Update `PipelineNode` class

- No changes to interface (config inherited from inner pipeline)
- May need internal updates for config passing

### Update `Node` class

- Add `with_config(config: RuntimeConfig) -> Node` method (for node-level overrides)
- Store config as optional attribute

## Phase 6: Cache & Callback Lazy Initialization

### Create `CacheSpec` class

- Factory pattern for cache creation
- Fields: `kind` ("disk", "memory"), `path`, `options`
- Method: `create() -> Cache` - instantiate the actual cache
- Serializable (no live file handles)

### Update `RuntimeConfig` to support lazy init

- Cache field accepts: `Cache` instance, `CacheSpec`, or callable `() -> Cache`
- Callbacks field accepts: list of callback instances or factories
- Materialize at execution time (in the target environment)

### Update cache usage in executors

- Check if cache is spec/factory and instantiate lazily
- Cache instances per pipeline/executor (not global)

## Phase 7: Update Examples & Documentation

### Update all examples

- `examples/daft_backend_example.py` → use DaftExecutor
- `examples/modal_backend_test.py` → use RuntimeConfig
- `examples/fluent_api_example.py` → update to new API
- Create new examples:
- `examples/runtime_config_example.py` - show RuntimeConfig usage
- `examples/composite_executor_example.py` - show CompositeExecutor
- `examples/config_hierarchy_example.py` - show configuration precedence

### Update documentation

- `docs/advanced/execution-engines.md` → complete rewrite matching the spec
- `docs/getting-started/quick-start.md` → update for new API
- `docs/in-depth/core-concepts.md` → add RuntimeConfig section
- Create new docs:
- `docs/advanced/executors.md` - detailed executor guide
- `docs/advanced/runtime-config.md` - configuration guide

### Update README

- Update quick start example
- Update feature list to mention new architecture
- Add migration guide section (for external users)

## Phase 8: Update Tests

### Update existing tests

- `tests/test_phase1_core_execution.py` - update for RuntimeConfig API
- `tests/test_phase2_map_operations.py` - update for RuntimeConfig API
- `tests/test_phase3_caching.py` - update for lazy cache init
- `tests/test_phase4_callbacks.py` - update for RuntimeConfig API
- `tests/test_phase5_nested_pipelines.py` - update for config inheritance
- `tests/test_daft_backend.py` → rename to `test_daft_executor.py`, update
- `tests/test_modal_backend.py` - update for RuntimeConfig

### Create new tests

- `tests/test_runtime_config.py` - config hierarchy, lazy init, context managers
- `tests/test_executors.py` - test all executor implementations
- `tests/test_composite_executor.py` - test composite executor patterns
- `tests/test_config_inheritance.py` - test nested pipeline config inheritance

## Phase 9: Update Package Exports

### Update `src/hypernodes/__init__.py`

- Remove: `Backend`, `LocalBackend`, `ModalBackend` from main exports (move to submodule)
- Add: `RuntimeConfig`, `set_default_config`, `get_default_config`
- Add: All executor classes
- Update: `DaftBackend` → `DaftExecutor`
- Reorganize **all** by category (Config, Executors, Backends, etc.)

### Create `src/hypernodes/backends/__init__.py` (optional)

- Organize backends in submodule: `LocalBackend`, `ModalBackend`, `RemoteBackend`

### Create `src/hypernodes/executors/__init__.py` (optional)

- Organize executors in submodule for cleaner imports

## Key Files Summary

### New Files

- `src/hypernodes/config.py` - RuntimeConfig, global config management
- `src/hypernodes/executor.py` - Executor base + basic implementations
- `src/hypernodes/daft_executor.py` - Renamed from daft_backend.py
- `examples/runtime_config_example.py`
- `examples/composite_executor_example.py`
- `examples/config_hierarchy_example.py`
- `tests/test_runtime_config.py`
- `tests/test_executors.py`
- `tests/test_composite_executor.py`

### Modified Files

- `src/hypernodes/backend.py` - Simplified Backend, refactored LocalBackend, updated ModalBackend
- `src/hypernodes/pipeline.py` - New config parameter, removed old fluent methods
- `src/hypernodes/node.py` - Add with_config method
- `src/hypernodes/cache.py` - Add CacheSpec for lazy init
- `src/hypernodes/__init__.py` - Update exports
- All example files
- All test files
- All documentation files

### Deleted Files

- `src/hypernodes/daft_backend.py` (renamed to daft_executor.py)

## Implementation Notes

- **Breaking Changes**: This is a complete API refactor - no backward compatibility
- **Testing Strategy**: Update tests incrementally as each phase completes
- **Migration Path**: Since no backward compat, focus on clear documentation
- **Execution Order**: Follow phases sequentially - later phases depend on earlier ones

### To-dos

- [ ] Create config.py with RuntimeConfig, CacheSpec, and global config management
- [ ] Create executor.py with Executor abstract base class and basic executors (Sequential, Async, ThreadPool, ProcessPool)
- [ ] Refactor backend.py: simplify Backend ABC, refactor LocalBackend to delegate to executors, update ModalBackend
- [ ] Rename daft_backend.py to daft_executor.py and refactor DaftBackend to DaftExecutor implementing Executor interface
- [ ] Implement CompositeExecutor and extract PipelineExecutionEngine for reusable execution logic
- [ ] Update Pipeline class: replace backend/cache/callbacks with config parameter, update run/map methods, add with_config, implement effective_config
- [ ] Update Node class with with_config method for node-level configuration overrides
- [ ] Implement lazy initialization for cache and callbacks using CacheSpec and factory patterns
- [ ] Update all examples and create new ones demonstrating RuntimeConfig, CompositeExecutor, and config hierarchy
- [ ] Rewrite execution-engines.md and create new documentation for executors and runtime configuration
- [ ] Update all existing tests for new API and create new test files for RuntimeConfig, executors, and config inheritance
- [ ] Update __init__.py with new exports (RuntimeConfig, executors) and reorganize package structure