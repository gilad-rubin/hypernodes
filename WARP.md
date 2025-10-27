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

#### Backend (`src/hypernodes/backend.py`)
- Abstract execution strategy
- **LocalBackend**: Sequential, async, threaded, or parallel execution
- **ModalBackend**: Remote execution on Modal infrastructure (optional: `pip install 'hypernodes[modal]'`)
- **DaftBackend**: Distributed DataFrame-based execution (optional: `pip install 'hypernodes[viz]'` + daft)

#### Cache (`src/hypernodes/cache.py`)
- Content-addressed caching via computation signatures
- Signature: `hash(code_hash + env_hash + inputs_hash + deps_hash)`
- **DiskCache**: Persistent filesystem-based cache (uses pickle)
- Cache invalidation is automatic when code, inputs, or dependencies change

#### Callbacks (`src/hypernodes/callbacks.py`)
- Hooks into pipeline execution lifecycle
- **ProgressCallback** (`telemetry/progress.py`): Live progress bars with tqdm/rich
- **TelemetryCallback** (`telemetry/tracing.py`): Distributed tracing with Logfire

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
- Core logic: `src/hypernodes/`
- Tests: `tests/` (organized by feature phases: core, map, caching, callbacks, nested)
- Scripts: `scripts/` (for testing and debugging)
- Examples: `examples/`
- Documentation: `docs/`

### Testing Philosophy
- Test with single inputs first, then scale to multiple
- Tests should verify both functionality and caching behavior
- Use `DiskCache` in tests to verify cache hits/misses

### When Making Changes
1. Run relevant tests after changes: `uv run pytest tests/<file>.py`
2. For caching changes, run: `tests/test_phase3_caching.py` and `tests/test_phase3_class_caching.py`
3. For map operations, run: `tests/test_phase2_map_operations.py`
4. For nested pipelines, run: `tests/test_phase5_nested_pipelines.py`

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

### Execution Flow (LocalBackend)

1. **Dependency Graph Construction**: Pipeline builds NetworkX DAG from nodes
2. **Topological Sort**: Compute execution order respecting dependencies
3. **Selective Execution**: If `output_name` specified, only execute required nodes
4. **Node Execution Loop**:
   - Check cache (if enabled)
   - Collect dependencies from available values
   - Execute node function
   - Store output in cache (if enabled)
   - Fire callbacks (before/after node, before/after pipeline)

### Map Execution Strategy

For `.map()` operations:
1. Items are transformed into list of input dicts
2. Each item gets its own cache signature
3. Backend executes items (sequential/async/threaded/parallel based on `map_execution`)
4. Cached items are skipped automatically
5. Results aggregated and returned

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
