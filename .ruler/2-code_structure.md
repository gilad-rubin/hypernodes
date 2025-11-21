# HyperNodes Codebase Structure

## Overview
HyperNodes is a cache-first pipeline framework for ML/AI workflows. It enables building DAGs from decorated functions with automatic dependency resolution, intelligent caching, and pluggable execution engines.

## Core Architecture

### 1. Node (`node.py`)
Wraps a Python function with pipeline metadata. The `@node` decorator creates Node instances.

```python
@node(output_name="result", cache=True)
def process(x: int) -> int:
    return x * 2
```

**Key properties:**
- `func`: The wrapped function
- `output_name`: Name(s) of outputs (str or tuple for multiple)
- `root_args`: Tuple of input parameter names (from function signature)
- `code_hash`: Cached SHA256 hash of function source (computed once at creation)
- `cache`: Whether to cache this node's output

### 2. Pipeline (`pipeline.py`)
Manages a DAG of nodes. **Pure definition** - no execution state.

**NEW ARCHITECTURE:** Pipeline no longer holds `cache` or `callbacks`. These are now configured at the engine level.

```python
from hypernodes import Pipeline, SeqEngine, DiskCache
from hypernodes.telemetry import ProgressCallback

# Execution config is at the engine level
engine = SeqEngine(
    cache=DiskCache(path=".cache"),
    callbacks=[ProgressCallback()]
)
pipeline = Pipeline(nodes=[node1, node2], engine=engine)

# Execute once
result = pipeline.run(inputs={"x": 5})  # Returns: {"result": 10}

# Execute multiple times (map over inputs)
results = pipeline.map(inputs={"x": [1,2,3]}, map_over="x")
# Returns: [{"result": 2}, {"result": 4}, {"result": 6}]
```

**Key methods:**
- `run()`: Execute pipeline once with given inputs
- `map()`: Execute pipeline multiple times over collections (zip/product modes)
- `as_node()`: Wrap pipeline as a node for nesting
- `visualize()`: Generate Graphviz visualization

### 3. HyperNode Protocol (`hypernode.py`)
Structural protocol (duck typing) for executable units. Both `Node` and `PipelineNode` implement this.

**Required attributes:**
- `name`, `cache`, `root_args`, `output_name`, `code_hash`

### 4. Engines & Orchestration

#### Engines (`engines.py`, `sequential_engine.py`)
Pluggable execution strategies. All implement the `Engine` protocol.

**NEW ARCHITECTURE:** Engines now own the runtime environment:
- `cache`: Cache backend instance
- `callbacks`: List of callback instances  
- Execution strategy (sequential, parallel, distributed)

```python
from hypernodes import SeqEngine, DiskCache
from hypernodes.telemetry import ProgressCallback

# SeqEngine - simple topological execution
engine = SeqEngine(
    cache=DiskCache(path=".cache"),
    callbacks=[ProgressCallback()]
)

# DaftEngine - distributed execution
from hypernodes.engines import DaftEngine
engine = DaftEngine(
    use_batch_udf=True,
    cache=DiskCache(path=".cache"),
    callbacks=[ProgressCallback()]
)

pipeline = Pipeline(nodes=[...], engine=engine)
```

**Available engines:**
- `SeqEngine`: Default - simple topological execution, no parallelism
- `DaftEngine`: Distributed DataFrame execution (optional, install getdaft)
- `DaskEngine`: Parallel map operations using Dask Bag (optional, install dask)

#### ExecutionOrchestrator (`orchestrator.py`)
**NEW:** Shared lifecycle management for all engines.

Centralizes the "outer loop" of execution:
- CallbackDispatcher setup
- Pipeline metadata tracking
- Start/End event notifications
- Callback/engine compatibility validation

```python
# Used internally by engines
with ExecutionOrchestrator(pipeline, callbacks, context) as orchestrator:
    orchestrator.validate_callbacks("SeqEngine")
    orchestrator.notify_start(inputs)
    # ... execute nodes ...
    orchestrator.notify_end(outputs)
```

**Benefits:**
- Consistent behavior across all engines
- No code duplication for lifecycle management
- Easy to add new engines without re-implementing orchestration

#### DaftEngine - High-Performance Distributed Execution

The `DaftEngine` (`src/hypernodes/integrations/daft/engine.py`) is a facade that delegates execution to specialized operations (`src/hypernodes/integrations/daft/operations.py`). It transforms pipelines into lazy DataFrame operations using [Daft](https://www.getdaft.io/).

```python
from hypernodes.engines import DaftEngine

# Auto-optimized for best performance (recommended)
engine = DaftEngine(use_batch_udf=True, cache=DiskCache(path=".cache"))
pipeline = Pipeline(nodes=[...], engine=engine)
results = pipeline.map(inputs={"x": [1,2,3]}, map_over="x")
```

**Architecture:**
- **Engine (`engine.py`)**: Orchestrates execution, manages caching, and creates operations.
- **Operations (`operations.py`)**: Modular execution strategies:
  - `FunctionNodeOperation`: Standard scalar nodes (sync/async).
  - `BatchNodeOperation`: Optimized batch UDFs.
  - `PipelineNodeOperation`: Nested pipelines (handles `explode`/`implode` for map_over).
  - `DualNodeOperation`: Switches between singular and batch implementations.
- **CodeGen (`codegen.py`)**: Tracks imports and UDF definitions to generate standalone Daft scripts.

**Key features:**
- **Lazy execution**: Builds computation graph, executes on `.collect()`
- **Batch UDFs**: Auto-calculates optimal batch sizes (64-1024) based on data size
- **Auto-tuned parallelism**: Optimizes `max_workers` (8-16x CPU cores) for I/O-bound tasks
- **Stateful parameters**: Auto-detects or explicitly handles expensive resources (models, DB connections)
- **Per-item caching**: In map mode, caches individual items (not batches) for efficient incremental computation

**Configuration options:**
```python
engine = DaftEngine(
    use_batch_udf=True,  # Use batch UDFs for performance
    cache=DiskCache(path=".cache"),  # Enable per-item caching
    default_daft_config={
        "batch_size": 1024,      # Override auto-calculation
        "max_workers": 128,      # ThreadPoolExecutor workers
        "use_process": False,    # Process isolation (avoids GIL)
        "gpus": 1                # GPU resources
    }
)
```

**Caching behavior:**
- **run() mode**: Standard signature-based caching (same as SeqEngine)
- **map() mode**: Per-item caching - each item in the batch is cached individually
  - Pre-checks cache for all items before execution
  - Only executes uncached items through Daft
  - Merges cached and newly computed results
  - Example: `[1, 2, 3]` cached → `[1, 2, 4]` only executes item `4`

**Stateful parameters** (expensive to initialize):
```python
# Auto-detected via __daft_stateful__ attribute
class Model:
    __daft_stateful__ = True
    def __init__(self):
        self.model = load_expensive_model()
    
    def predict(self, text: str) -> str:
        return self.model(text)

# Or explicit hint
@node(output_name="prediction", stateful_params=["model"])
def predict(text: str, model: Model) -> str:
    return model.predict(text)
```

### 6. Cache (`cache.py`)
Content-addressed caching using computation signatures.

**NEW ARCHITECTURE:** Cache implementations are configured at the engine level.

**Signature formula:**
```
sig(node) = hash(code_hash + env_hash + inputs_hash + deps_hash)
```

- `DiskCache`: Pickle-based filesystem cache
- Cache keys are deterministic - same inputs + code = same signature

```python
from hypernodes import SeqEngine, DiskCache

# Configure cache at engine level
engine = SeqEngine(cache=DiskCache(path=".cache"))
pipeline = Pipeline(nodes=[...], engine=engine)
```

**Cache Hierarchy:**
1. **Engine Level**: `engine.cache` - the backend instance
2. **Node Level**: `node.cache` (True/False) - whether this node should be cached
3. **Effective**: Caching happens if `engine.cache is not None and node.cache is True`

### 7. Callbacks (`callbacks.py`, `orchestrator.py`)
Lifecycle hooks for observability. All callbacks inherit from `PipelineCallback`.

**NEW ARCHITECTURE:** Callbacks are configured at the engine level.

```python
from hypernodes import SeqEngine
from hypernodes.telemetry import ProgressCallback

# Configure callbacks at engine level
engine = SeqEngine(callbacks=[ProgressCallback()])
pipeline = Pipeline(nodes=[...], engine=engine)
```

**Lifecycle events:**
- `on_pipeline_start/end`
- `on_node_start/end`
- `on_node_cached` (cache hit)
- `on_map_start/end`
- `on_map_item_start/end`
- `on_nested_pipeline_start/end`

**Available callbacks (telemetry/):**
- `ProgressCallback`: Live tqdm progress bars (auto-detects Jupyter vs CLI)
- `TelemetryCallback`: Distributed tracing with Logfire

**Engine Compatibility:**
Callbacks can declare which engines they support:
```python
class DaftOnlyCallback(PipelineCallback):
    @property
    def supported_engines(self):
        return ["DaftEngine"]  # Fails early if used with SeqEngine

# Usage
engine = SeqEngine(callbacks=[DaftOnlyCallback()])  # ❌ ValueError!
```

Validation happens at execution time via `ExecutionOrchestrator.validate_callbacks()`.

## Advanced Features

### Nested Pipelines
Pipelines can be composed hierarchically using `.as_node()`:

```python
inner = Pipeline(nodes=[clean_text])
outer = Pipeline(nodes=[inner.as_node(), analyze])
result = outer.run(inputs={"text": "hello"})
```

**Input/Output mapping:**
```python
# Inner expects "passage", outer provides "document"
adapted = inner.as_node(
    input_mapping={"document": "passage"},  # outer → inner
    output_mapping={"cleaned": "processed"}  # inner → outer
)
```

#### Nested Pipelines with `map_over` - Internal Batch Processing

The `map_over` parameter enables **internal mapping**: turn a single-item pipeline into a batch processor.

**Pattern: Single → Batch transformation**
```python
# Inner pipeline processes ONE item
@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

single_item_pipeline = Pipeline(nodes=[double])

# Transform it to process LISTS internally
batch_pipeline = single_item_pipeline.as_node(
    map_over="items",                    # Outer parameter name (a list)
    input_mapping={"items": "x"},        # Map list items → inner param
    output_mapping={"doubled": "results"} # Collect results
)

# Use in outer pipeline
outer = Pipeline(nodes=[batch_pipeline])
result = outer.run(inputs={"items": [1, 2, 3]})
# Returns: {"results": [2, 4, 6]}
```

**How it works:**
1. Outer pipeline receives `items=[1,2,3]` (a list)
2. `map_over="items"` triggers internal `.map()` on inner pipeline
3. Inner pipeline processes each: `x=1`, `x=2`, `x=3`
4. Results collected as list: `[2, 4, 6]`

**Real-world example - Document processing:**
```python
# Inner: processes ONE document
@node(output_name="cleaned")
def clean_text(passage: str) -> str:
    return passage.strip().lower()

@node(output_name="summary")
def summarize(cleaned: str) -> str:
    return cleaned[:100]

doc_processor = Pipeline(nodes=[clean_text, summarize])

# Adapt to process BATCHES of documents
batch_processor = doc_processor.as_node(
    map_over="documents",
    input_mapping={"documents": "passage"},
    output_mapping={"summary": "summaries"}
)

# Now processes lists!
outer = Pipeline(nodes=[batch_processor])
result = outer.run(inputs={"documents": ["  Doc 1  ", "  Doc 2  "]})
# Returns: {"summaries": ["doc 1", "doc 2"]}
```

**Combining with zip/product modes:**
```python
# Multiple map_over parameters
adapted = inner.as_node(
    map_over=["docs", "templates"],
    map_mode="zip",  # or "product"
    input_mapping={"docs": "text", "templates": "template"}
)
```

### Map Operations
Execute pipeline over collections with two modes:

**Zip mode** (parallel iteration):
```python
results = pipeline.map(
    inputs={"x": [1,2,3], "y": [10,20,30]},
    map_over=["x", "y"],
    map_mode="zip"
)
# Pairs: (1,10), (2,20), (3,30)
```

**Product mode** (all combinations):
```python
results = pipeline.map(
    inputs={"x": [2,3], "y": [10,100]},
    map_over=["x", "y"],
    map_mode="product"
)
# Pairs: (2,10), (2,100), (3,10), (3,100)
```

### Selective Outputs
Request only specific outputs to avoid unnecessary computation:

```python
result = pipeline.run(inputs={"x": 5}, output_name=["result1", "result2"])
# Only returns requested outputs
```

## Key Files Reference

### Core (`src/hypernodes/`)
- `node.py`: Node class and `@node` decorator
- `pipeline.py`: Pipeline class with run/map/as_node methods (pure definition, no execution state)
- `sequential_engine.py`: Default execution engine (owns cache, callbacks)
- `orchestrator.py`: **NEW** - Shared execution orchestration for all engines
- `node_execution.py`: Single node execution logic (decoupled, accepts explicit cache/callbacks)
- `graph_builder.py`: DAG construction from node list (SimpleGraphBuilder implementation)
- `map_planner.py`: Map operation planning (zip vs product)
- `cache.py`: Caching system with signature computation
- `callbacks.py`: Callback protocol + context + dispatcher

### Visualization (`src/hypernodes/viz/`)
- `visualization.py`: Graphviz rendering and legacy visualization functions
- `graph_serializer.py`: Frontend-agnostic graph serialization
- `visualization_engines.py`: Pluggable rendering engines (Graphviz, IPyWidget)
- `visualization_widget.py`: Interactive IPyWidget visualization components

### Integrations (`src/hypernodes/integrations/`)
- `daft/engine.py`: DaftEngine facade
- `daft/operations.py`: Modular Daft operations (Node, Batch, Pipeline)
- `daft/codegen.py`: Code generation context
- `dask/engine.py`: DaskEngine for parallel maps

### Telemetry (`src/hypernodes/telemetry/`)
- `progress.py`: ProgressCallback (tqdm)
- `tracing.py`: TelemetryCallback (Logfire)

### Tests (`tests/`)
- `test_execution.py`: Basic run() tests
- `test_map.py`: Map operation tests
- `test_caching.py`: Cache behavior tests
- `test_nested_pipelines.py`: Nested pipeline tests
- `test_callbacks.py`: Callback system tests

## Common Patterns

### Basic Pipeline
```python
from hypernodes import Pipeline, node

@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

pipeline = Pipeline(nodes=[double])
result = pipeline.run(inputs={"x": 5})  # {"doubled": 10}
```

### With Caching + Progress
```python
from hypernodes import Pipeline, SeqEngine, DiskCache
from hypernodes.telemetry import ProgressCallback

# Configure engine with cache and callbacks
engine = SeqEngine(
    cache=DiskCache(path=".cache"),
    callbacks=[ProgressCallback()]
)
pipeline = Pipeline(nodes=[slow_node1, slow_node2], engine=engine)
```

### Multiple Outputs
```python
@node(output_name=("mean", "std"))
def stats(data: list) -> tuple:
    return sum(data)/len(data), calculate_std(data)
```

### Binding Default Inputs
Use `.bind()` to set default input values that can be overridden at runtime:

```python
from hypernodes import Pipeline, node

@node(output_name="scaled")
def scale(value: int, factor: int) -> int:
    return value * factor

# Bind default inputs
pipeline = Pipeline(nodes=[scale]).bind(factor=10)

# Check what's bound and what's still needed
print(pipeline)  # Pipeline(nodes=1, bound=[factor=10], needs=[value])
print(pipeline.bound_inputs)      # {'factor': 10}
print(pipeline.root_args)         # ('value', 'factor') - full contract
print(pipeline.unfulfilled_args)  # ('value',) - what's still needed

# Run with bound defaults
result = pipeline.run(inputs={"value": 5})  # {"scaled": 50}

# Override bound inputs
result = pipeline.run(inputs={"value": 5, "factor": 100})  # {"scaled": 500}

# Works with map operations
results = pipeline.map(inputs={"value": [1, 2, 3]}, map_over="value")
# [{"scaled": 10}, {"scaled": 20}, {"scaled": 30}]

# Remove specific bindings
pipeline.unbind("factor")

# Or clear all bindings
pipeline.unbind()
```

**Tracking Input Fulfillment:**
- `.bound_inputs` - Dict of values set via `.bind()`
- `.unfulfilled_args` - Tuple of parameter names NOT yet bound
- `.root_args` - Full input contract (all parameters)
- Validation only requires unfulfilled args at runtime
- `__repr__()` shows bound inputs and unfulfilled args for debugging
- Visualizations show bound inputs with lighter/transparent color

**Common use cases:**
- Bind expensive resources (models, DB connections) once, use across multiple runs
- Set default hyperparameters that can be overridden per experiment
- Simplify API by pre-configuring common inputs

**For nested pipelines:**
```python
# Inner pipeline fully bound
inner = Pipeline(nodes=[add]).bind(x=5, y=10)
print(inner.unfulfilled_args)  # () - fully satisfied!

inner_node = inner.as_node()
print(inner_node.unfulfilled_args)  # () - exposes inner's status

outer = Pipeline(nodes=[inner_node])
outer.run()  # ✅ Works! No inputs needed
```

---

**Note:** After recent refactor, focus on newer files. Ignore `src/hypernodes/old/` and `tests/old/` directories.

- **Debug notebook:** `notebooks/verify_ipywidget.ipynb` contains a minimal ipywidget rendering sanity check (runs a tiny pipeline, decodes the iframe HTML, and surfaces any client-side errors for troubleshooting).

---

## Visualization System Details

The visualization system is organized in `src/hypernodes/viz/` with a clean separation between:
- **Graph Serialization** (`graph_serializer.py`): Frontend-agnostic semantic graph data
- **Rendering Engines** (`visualization_engines.py`): Pluggable backends (Graphviz, IPyWidget, etc.)
- **Legacy Visualization** (`visualization.py`): Backwards-compatible Graphviz rendering
- **Interactive Widgets** (`visualization_widget.py`): IPyWidget-based React Flow visualizations
  - **Fix (Nov 2025)**: Changed from base64 data URI to `srcdoc` attribute for better VSCode notebook compatibility

### GraphSerializer (`viz/graph_serializer.py`)

**Key Fix (Nov 2025):** Proper handling of cross-level connections and input/output mappings.

**Critical features:**
- **Cross-level edge resolution**: When a nested pipeline is expanded, edges connect directly to inner nodes that produce outputs, not to the PipelineNode wrapper
- **Input/output mapping labels**: Adds edge labels like `"outer_param → inner_param"` when parameters are renamed across boundaries
- **Global output tracking**: `_output_to_node_id` maps output names to actual producer node IDs across all nesting levels
- **Expanded pipeline detection**: Detects when dependencies are on expanded PipelineNodes and replaces them with inner producer nodes

**Edge creation logic:**
1. For node dependencies: Check if dependency is an expanded PipelineNode
   - If yes: Find actual inner nodes that produce needed outputs, create edges to those
   - If no: Create regular node→node edge
2. For parameter edges: Check if parameter is produced by node in outer scope
   - If yes: Create node→node edge with mapping label if names differ
   - If no: Create parameter→node edge using outer parameter name

**Example scenario:**
```python
# Inner pipeline outputs "evaluation_result"
# Outer expects "evaluation_results" via output_mapping
# GraphSerializer creates edge: evaluate_answer → compute_metrics
# With label: "evaluation_result → evaluation_results"
```

**Test coverage:** `tests/test_visualization_mappings.py` - comprehensive tests for all mapping scenarios
