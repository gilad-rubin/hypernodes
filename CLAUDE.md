---
Source: .ruler/0-instructions.md
---
## General
- use uv run X to run scripts
- use trash X instead of rm X. This allows me to rescue deleted files if I need.
- when making changes in the codebase - run them to verify everything works
- if an API key is needed, first check in the .env to make sure it exists. use dotenv to load if needed.
- prefer to search online and in documentations before acting
- tests go to tests folder. scripts go to scripts folder
- when you're finished with a task - you can write a summary, but just one is enough. no need for multiple summaries and markdowns.
- update .ruler/2-code-structure.md if you've changed something in the code structure :)

## Coding Principles
- When designing and implementing features - always prefer using SOLID principles.
- Use simple, human readable functions rather than massive long indented functions.
- Split classes functions into helper functions if needed

## Tools
- use tavily web search and context7 MCP servers whenever you're stuck or want to understand how a library works

## Jupyter
- Use concise, human readable cells
- avoid redundancy in notebooks. keep the cells and notebook as a whole concise.
- avoid using "special" emojis in jupyter, it can crash the notebook. you can use basic ones, like X, V etc...
- remember that jupyter has its own async handling. remember to use the correct syntax.
- If you're editing a module while reviewing the output in jupyter, remember to either restart the kernel or reload the module to see changes

- jupyter notebook's working directory is the project's working directory, so no need to do sys.path.insert(0, '/Users/...')
- run cells after you create them to verify things work as expected. read the output and decide what to do next
- when trying to figure out something about an object - iterate over it by running the cell and examining the output and refining

---
Source: .ruler/2-code_structure.md
---
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
Manages and executes a DAG of nodes. Auto-builds dependency graph from node signatures.

```python
pipeline = Pipeline(
    nodes=[node1, node2],
    engine=SequentialEngine(),  # Default
    cache=DiskCache(path=".cache"),
    callbacks=[ProgressCallback()]
)

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

### 4. Engines (`engines.py`, `sequential_engine.py`)
Pluggable execution strategies. All implement the `Engine` protocol.

**Available engines:**
- `SequentialEngine`: Default - simple topological execution, no parallelism
- `DaskEngine`: Parallel map operations using Dask Bag (optional, install dask)
- `DaftEngine`: Distributed DataFrame execution (optional, install getdaft)

#### DaftEngine - High-Performance Distributed Execution

DaftEngine transforms pipelines into lazy DataFrame operations using `@daft.func` UDFs:

```python
from hypernodes.engines import DaftEngine

# Auto-optimized for best performance (recommended)
engine = DaftEngine(use_batch_udf=True)
pipeline = Pipeline(nodes=[...], engine=engine)
results = pipeline.map(inputs={"x": [1,2,3]}, map_over="x")
```

**Key features:**
- **Lazy execution**: Builds computation graph, executes on `.collect()`
- **Batch UDFs**: Auto-calculates optimal batch sizes (64-1024) based on data size
- **Auto-tuned parallelism**: Optimizes `max_workers` (8-16x CPU cores) for I/O-bound tasks
- **Stateful parameters**: Auto-detects or explicitly handles expensive resources (models, DB connections)

**Configuration options:**
```python
engine = DaftEngine(
    use_batch_udf=True,  # Use batch UDFs for performance
    default_daft_config={
        "batch_size": 1024,      # Override auto-calculation
        "max_workers": 128,      # ThreadPoolExecutor workers
        "use_process": False,    # Process isolation (avoids GIL)
        "gpus": 1                # GPU resources
    }
)
```

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

### 5. Cache (`cache.py`)
Content-addressed caching using computation signatures.

**Signature formula:**
```
sig(node) = hash(code_hash + env_hash + inputs_hash + deps_hash)
```

- `DiskCache`: Pickle-based filesystem cache
- Cache keys are deterministic - same inputs + code = same signature

```python
cache = DiskCache(path=".cache")
pipeline = Pipeline(nodes=[...], cache=cache)
```

### 6. Callbacks (`callbacks.py`)
Lifecycle hooks for observability. All callbacks inherit from `PipelineCallback`.

**Lifecycle events:**
- `on_pipeline_start/end`
- `on_node_start/end`
- `on_cache_hit/miss`

**Available callbacks (telemetry/):**
- `ProgressCallback`: Live tqdm progress bars (auto-detects Jupyter vs CLI)
- `TelemetryCallback`: Distributed tracing with Logfire

```python
from hypernodes.telemetry import ProgressCallback
pipeline = Pipeline(nodes=[...], callbacks=[ProgressCallback()])
```

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
- `pipeline.py`: Pipeline class with run/map/as_node methods
- `typed_interface.py`: TypedDict generation for IDE autocomplete
- `sequential_engine.py`: Default execution engine
- `node_execution.py`: Single node execution logic (caching + callbacks)
- `graph_builder.py`: DAG construction from node list
- `map_planner.py`: Map operation planning (zip vs product)
- `cache.py`: Caching system with signature computation
- `callbacks.py`: Callback protocol + context
- `visualization.py`: Graphviz rendering

### Integrations (`src/hypernodes/integrations/`)
- `daft/engine.py`: DaftEngine for distributed execution
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
from hypernodes import Pipeline, node, DiskCache
from hypernodes.telemetry import ProgressCallback

pipeline = Pipeline(
    nodes=[slow_node1, slow_node2],
    cache=DiskCache(path=".cache"),
    callbacks=[ProgressCallback()]
)
```

### Multiple Outputs
```python
@node(output_name=("mean", "std"))
def stats(data: list) -> tuple:
    return sum(data)/len(data), calculate_std(data)
```

### Typed Interfaces (IDE Autocomplete)
```python
from hypernodes import Pipeline, node

@node(output_name="result")
def process(x: int, y: str) -> float:
    return float(x) * len(y)

pipeline = Pipeline(nodes=[process], name="MyPipeline")

# Generate TypedDict interfaces
InputType = pipeline.get_input_type()
OutputType = pipeline.get_output_type()

# ✅ IDE autocomplete works!
inputs: InputType = {"x": 5, "y": "hello"}
result: OutputType = pipeline.run(inputs=inputs)
print(result["result"])  # IDE autocompletes key names
```

---

**Note:** After recent refactor, focus on newer files. Ignore `src/hypernodes/old/` and `tests/old/` directories.
