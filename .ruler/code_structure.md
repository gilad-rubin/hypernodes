# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

HyperNodes is a hierarchical, modular pipeline system with intelligent caching for ML/AI workflows. The system treats caching as a first-class citizen and enables building complex pipelines from simple, reusable nodes.

**Core Philosophy**: Build once, cache intelligently, run anywhere.

## Commands

### Running Scripts
```bash
uv run <script.py>
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_phase1_core_execution.py

# Run specific test
uv run pytest tests/test_phase1_core_execution.py::test_basic_pipeline
```

### File Operations
Use `trash` instead of `rm` to allow file recovery:
```bash
trash <file>
```

## Architecture

### Core Components

**Node → Pipeline → Node (Hierarchical Composition)**

The system has a recursive structure where:
1. Functions become Nodes (via `@node` decorator)
2. Nodes compose into Pipelines (DAG of nodes)
3. Pipelines can be wrapped to act as Nodes in other pipelines (via `PipelineNode` or `.as_node()`)

### Key Classes

#### Node (`src/hypernodes/node.py`)
- Wraps Python functions with pipeline metadata
- Stores `parameters` (function signature), `output_name`, and `cache` flag
- Parameters define dependencies implicitly (parameter names match output names of other nodes)

#### Pipeline (`src/hypernodes/pipeline.py`)
- Manages DAG execution using NetworkX
- Computes topological order and resolves dependencies
- Has two execution methods:
  - `.run(inputs)`: Execute with single input set
  - `.map(inputs, map_over)`: Execute over multiple items with per-item caching
- Can be nested: pipelines contain other pipelines as nodes
- Provides `.as_node()` method to adapt interface with input/output mapping

#### PipelineNode (`src/hypernodes/pipeline.py`)
- Adapts a Pipeline to behave as a Node
- Handles `input_mapping` (outer → inner parameter names)
- Handles `output_mapping` (inner → outer output names)
- Supports `map_over` parameter to vectorize operations

#### Engine (`src/hypernodes/engine.py`, `src/hypernodes/engines.py`)
- Abstract base class defining execution interface
- **HypernodesEngine**: Main implementation for node-by-node execution
  - Orchestrates pipeline execution (dependency resolution, topological ordering)
  - Manages map operations (item preparation, result transposition)
  - Supports node-level parallelism via executors
  - Computes dependency levels for parallel execution within pipeline
- **DaftEngine** (`integrations/daft/engine.py`): Distributed DataFrame-based execution (optional: `pip install 'hypernodes[viz]'` + daft)

#### Executors (`src/hypernodes/executors.py`)
- Uniform concurrent.futures-like interface for different execution strategies
- **SequentialExecutor**: Immediate synchronous execution (debugging, simple pipelines)
- **AsyncExecutor**: Asyncio-based concurrent execution with auto-wrapping for sync functions
  - Runs event loop in background thread (Jupyter-compatible)
  - High concurrency for I/O-bound workloads (default: 100 workers)
- **ThreadPoolExecutor**: Standard library thread-based parallelism
- **ProcessPoolExecutor**: Standard library process-based parallelism
- **loky**: Optional robust parallel execution with cloudpickle support

#### Node Execution (`src/hypernodes/node_execution.py`)
- Single-responsibility module for executing individual nodes
- Handles signature computation, cache get/put, callbacks
- Supports both sync and async node execution
- Error handling and node identification

#### Async Utilities (`src/hypernodes/async_utils.py`)
- Utilities for async/sync interoperability
- Event loop management for different contexts (Jupyter, standalone)

#### Cache (`src/hypernodes/cache.py`)
- Content-addressed caching via computation signatures
- Signature: `hash(code_hash + env_hash + inputs_hash + deps_hash)`
- **DiskCache**: Persistent filesystem-based cache (uses pickle)
- Cache invalidation is automatic when code, inputs, or dependencies change

#### Callbacks (`src/hypernodes/callbacks.py`)
- Hooks into pipeline execution lifecycle
- **ProgressCallback** (`telemetry/progress.py`): Live progress bars with tqdm/rich
- **TelemetryCallback** (`telemetry/tracing.py`): Distributed tracing with Logfire
- **WaterfallCallback** (`telemetry/waterfall.py`): Waterfall visualization of execution timing

### Dependency Resolution

Dependencies are implicit through parameter matching:
```python
@node(output_name="cleaned_text")
def clean(passage: str) -> str: ...

@node(output_name="word_count")
def count(cleaned_text: str) -> int: ...  # Depends on clean() via parameter name
```

Pipeline builds a DAG by matching parameter names to output names.

### Map Operations

Two modes for `.map()`:
1. **With input_mapping**: Parameter names are mapped, items passed directly
2. **Without input_mapping + dict items**: List of dicts is transposed (dict of lists) automatically

Per-item caching: Each item in `.map()` is cached independently by signature.

### Nested Pipelines

Pipelines inherit configuration (backend, cache, callbacks) from parent unless explicitly overridden. Use `.as_node()` to adapt nested pipeline interfaces:

```python
inner = Pipeline(nodes=[...])
outer_node = inner.as_node(
    input_mapping={"outer_param": "inner_param"},
    output_mapping={"inner_output": "outer_output"},
    map_over="outer_param"  # Vectorize over this parameter
)
outer = Pipeline(nodes=[load, outer_node, save])
```

## Code Conventions

### Module Organization
```
src/hypernodes/
├── node.py                 # Node class and @node decorator
├── pipeline.py             # Pipeline and PipelineNode classes
├── engine.py               # Engine base class and HypernodesEngine
├── engines.py              # Unified engine imports
├── executors.py            # SequentialExecutor, AsyncExecutor
├── node_execution.py       # Single node execution logic
├── async_utils.py          # Async/sync interop utilities
├── cache.py                # DiskCache and signature computation
├── callbacks.py            # PipelineCallback base class
├── visualization.py        # DAG visualization
├── exceptions.py           # Custom exceptions
├── telemetry/              # Observability modules
│   ├── progress.py         # ProgressCallback
│   ├── tracing.py          # TelemetryCallback (OpenTelemetry)
│   ├── waterfall.py        # WaterfallCallback
│   └── environment.py      # Environment detection
└── integrations/
    └── daft/
        └── engine.py       # DaftEngine (optional)

tests/
├── test_phase1_core_execution.py        # Basic pipeline execution
├── test_phase2_map_operations.py        # .map() functionality
├── test_phase2_engine_renaming.py       # Engine API
├── test_phase3_caching.py               # Cache behavior
├── test_phase3_class_caching.py         # Class-based caching
├── test_phase4_callbacks.py             # Callback system
├── test_phase5_nested_pipelines.py      # Nested composition
├── test_engine.py                       # Engine functionality
├── test_engine_execution.py             # Engine execution modes
├── test_executor_adapters.py            # Executor interfaces
├── test_executor_performance.py         # Performance benchmarks
├── test_async_and_executors.py          # Async execution
├── test_async_autowrap.py               # Async auto-wrapping
├── test_node_execution.py               # Node-level execution
├── test_selective_output.py             # Selective node execution
├── test_cache_encoder_like_objects.py   # Cache key generation
├── test_cache_mapped_pipeline_items.py  # Per-item caching
├── test_daft_backend.py                 # DaftEngine integration
├── test_daft_backend_complex_types.py   # Complex type handling
├── test_daft_backend_map_over.py        # Daft map operations
├── test_daft_preserve_original_column.py # Daft column preservation
├── test_parallel_reuse.py               # Executor reuse
├── test_pipeline_naming_and_nesting.py  # Pipeline naming
├── test_visualization_depth.py          # Visualization depth control
└── test_telemetry_basic.py              # Telemetry integration
```

### Testing Philosophy
- Test with single inputs first, then scale to multiple
- Tests should verify both functionality and caching behavior
- Use `DiskCache` in tests to verify cache hits/misses
- ~112 test functions across 24 test files
- Tests organized by phases (1-5) and features

### When Making Changes
1. Run relevant tests after changes: `uv run pytest tests/<file>.py`
2. For caching changes, run: `tests/test_phase3_caching.py` and `tests/test_phase3_class_caching.py`
3. For map operations, run: `tests/test_phase2_map_operations.py`
4. For nested pipelines, run: `tests/test_phase5_nested_pipelines.py`
5. For engine/executor changes, run: `tests/test_engine*.py` and `tests/test_executor*.py`
6. For async changes, run: `tests/test_async*.py`

## Development Workflow

### API Keys and Secrets
- Check `.env` for existing keys before requesting new ones
- Use `dotenv` to load environment variables
- Never hardcode secrets in code

### Jupyter Notebooks
- Notebooks in `notebooks/` directory
- Notebook working directory = project root (no need for `sys.path` manipulation)
- When editing modules, restart kernel or reload module to see changes
- Avoid special Unicode emojis (can crash notebooks)
- Run cells after creation to verify behavior

### Optional Dependencies
The project uses optional dependency groups:
- `[viz]`: Visualization (graphviz, rich, plotly, ipywidgets)
- `[modal]`: Remote execution on Modal
- `[telemetry]`: Tracing with Logfire
- `[examples]`: Dependencies for example scripts (daft, numpy, pydantic)

Install with: `pip install 'hypernodes[viz,modal]'`

## Architecture Deep Dive

### Execution Flow (HypernodesEngine)

1. **Dependency Graph Construction**: Pipeline builds NetworkX DAG from nodes
2. **Topological Sort**: Compute execution order respecting dependencies
3. **Selective Execution**: If `output_name` specified, only execute required nodes
4. **Executor Selection**: Resolve executor specification (string or instance)
   - `"sequential"` → SequentialExecutor
   - `"async"` → AsyncExecutor (auto-wraps sync functions)
   - `"threaded"` → ThreadPoolExecutor
   - `"parallel"` → ProcessPoolExecutor or loky
5. **Parallel Level Computation**: For async/threaded executors, compute dependency levels
   - Uses NetworkX topological_generations
   - Nodes at same level have no dependencies on each other → can run in parallel
6. **Node Execution Loop**:
   - Check cache (if enabled)
   - Collect dependencies from available values
   - Execute node function via executor
   - Store output in cache (if enabled)
   - Fire callbacks (before/after node, before/after pipeline)

### Map Execution Strategy

For `.map()` operations:
1. Items are transformed into list of input dicts
2. Each item gets its own cache signature
3. Engine uses map_executor (defaults to node_executor if not specified)
4. Executor processes items:
   - `sequential`: One at a time
   - `async`: Up to 100 concurrent items (I/O-bound)
   - `threaded`: CPU-count concurrent items (mixed workload)
   - `parallel`: CPU-count concurrent processes (CPU-bound)
5. Cached items are skipped automatically during execution
6. Results aggregated and returned

### Signature Computation

Cache key is deterministic hash of:
- **code_hash**: Function source code + closure variables
- **inputs_hash**: All input values (recursive hashing for complex objects)
- **deps_hash**: Signatures of upstream dependencies (recursive)
- **env_hash**: Environment configuration (version, salt)

Custom objects can implement `__cache_key__()` for control over hashing.

## Visualization

Use `pipeline.visualize()` to generate DAG visualization:
- Requires `[viz]` optional dependencies
- Supports depth control for nested pipelines
- Can export to file: `pipeline.visualize(filename="dag.svg")`
- Multiple design styles available via `DESIGN_STYLES`
