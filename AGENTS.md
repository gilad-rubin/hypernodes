---
alwaysApply: true
---
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

## Architecture (IMPORTANT)
- **Pipeline = Pure Definition**: Pipeline class only defines DAG structure. NO cache or callbacks.
- **Engine = Execution Runtime**: Engines own cache, callbacks, and execution strategy.
- **Current API**: `engine = SeqEngine(cache=..., callbacks=...); Pipeline(nodes=[...], engine=engine)`
- See guides/API_MIGRATION.md for complete migration guide.

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
- `ui_handler.py`: Backend state manager + serialization for all frontends
- `graphviz_ui.py`: Graphviz rendering engine
- `js_ui.py`: IPyWidget/React Flow rendering helpers
- `visualization_engine.py`: Pluggable engine registry/protocol

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
- **GraphWalker (`graph_walker.py`)**: Core graph traversal that generates node/edge structure from pipeline. **CRITICAL**: Uses `traverse_collapsed` parameter to control whether collapsed pipelines expose internal structure.
- **UIHandler (`ui_handler.py`)**: Backend state manager and serializer powering all frontends (Graphviz + React Flow). Handles depth, grouping, expansion/collapse, and emits semantic graph data (nodes/edges/levels) with grouped inputs and mapping labels.
- **JSRenderer (`js/renderer.py`)**: Transforms VisualizationGraph → React Flow node/edge format.
- **HTML Generator (`js/html_generator.py`)**: Generates complete HTML with React/ELK/Tailwind embedded.
- **State Utils (`assets/viz/state_utils.js`)**: Client-side state transformations (applyState, applyVisibility, compressEdges, groupInputs) and debug API.
- **Rendering Engines** (`visualization_engine.py` + implementations): Graphviz (`graphviz/renderer.py`) and IPyWidget/React Flow.
- **Legacy Visualization**: Older helpers live under `viz/graphviz_ui.py`; ignore `src/hypernodes/old/`.

### Key Parameters

**GraphWalker `traverse_collapsed`:**
- **`False`** (for static Graphviz): Collapsed pipelines remain truly collapsed
- **`True`** (for interactive viz): Pre-fetches internal structure for expand/collapse

### JS Visualization Data Flow

```
Python:  GraphWalker → UIHandler → JSRenderer → html_generator
                                         ↓
Browser: JSON → applyState → applyVisibility → compressEdges → groupInputs → ELK → ReactFlow
```

### Debug API (Browser Console)

The visualization exposes `HyperNodesVizState.debug`:

```javascript
// Enable verbose logging for edge compression and layout
HyperNodesVizState.debug.enableDebug()

// Analyze current graph state
HyperNodesVizState.debug.analyzeState()

// Get current pipeline expansion state
HyperNodesVizState.debug.getExpansionState()

// Simulate edge compression with specific expansion state
HyperNodesVizState.debug.simulateCompression({ 'rag_pipeline': false })
```

### Key Client-Side Functions (state_utils.js)

| Function | Purpose |
|----------|---------|
| `applyState` | Applies theme, separateOutputs mode, combines outputs |
| `applyVisibility` | Hides children of collapsed pipelines |
| `compressEdges` | Remaps edges to visible ancestors when pipelines collapse |
| `groupInputs` | Groups inputs targeting the same node |

### Common Issues & Debugging

| Issue | Check | Location |
|-------|-------|----------|
| Missing edges after collapse | `compressEdges` output | `state_utils.js` |
| Hanging arrows | Handle positions | `html_generator.py` |
| Nodes not grouping | `groupInputs` | `state_utils.js` |
| Types missing on inputs | `_extract_input_type` | `graph_walker.py` |

See `.ruler/visualization_best_practices.md` for complete debugging guide.

---
Source: .ruler/3-theme_detection.md
---
# Trial 5: Parent CSS Variables
# Checks if we can read CSS variables from the parent document.
js = """
const style = getComputedStyle(window.parent.document.documentElement);
const bg = style.getPropertyValue('--vscode-editor-background');
document.getElementById('result').innerText = 'VS Code Bg Var (Parent): ' + (bg || 'Not found');
"""
create_test_widget(js, "Trial 5: Parent CSS Variables")

this works for bg color!

# Trial 6: Body Attribute
# Checks for data-vscode-theme-kind attribute on parent body.
js = """
const kind = window.parent.document.body.getAttribute('data-vscode-theme-kind');
document.getElementById('result').innerText = 'Theme Kind Attr: ' + (kind || 'Not found');
"""
create_test_widget(js, "Trial 6: Body Attribute")

this works for light/dark theme. you just need to search for "dark" or "light" in the text

---
Source: .ruler/visualization_best_practices.md
---
# Visualization System - Architecture & Debugging Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  React Flow + ELK Layout                                ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ ││
│  │  │ CustomNode  │  │ CustomEdge  │  │ PipelineGroup   │ ││
│  │  └─────────────┘  └─────────────┘  └─────────────────┘ ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │  state_utils.js                                         ││
│  │  applyState → applyVisibility → compressEdges →        ││
│  │  groupInputs                                            ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ HTML with embedded JSON
┌─────────────────────────────────────────────────────────────┐
│                    Python Backend                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ UIHandler   │→ │ JSRenderer  │→ │ html_generator.py   │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│         ▲                                                    │
│  ┌─────────────┐                                            │
│  │ GraphWalker │                                            │
│  └─────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
src/hypernodes/viz/
├── graph_walker.py      # Traverses pipeline DAG, generates flat node/edge structure
├── ui_handler.py        # Manages depth, expansion, serialization
├── structures.py        # Data classes: FunctionNode, PipelineNode, DataNode, VizEdge
├── js/
│   ├── renderer.py      # Transforms VisualizationGraph → React Flow format
│   └── html_generator.py # Generates complete HTML with React/ELK/Tailwind
├── graphviz/
│   └── renderer.py      # Static Graphviz SVG rendering

assets/viz/
├── state_utils.js       # Client-side state transformations (applyState, compressEdges, etc.)
├── theme_utils.js       # Theme detection and color parsing
├── reactflow.umd.js     # React Flow library
├── elk.bundled.js       # ELK layout library
└── custom.css           # Custom styling

tests/viz/
├── test_collapsed_pipeline_outputs_and_grouping.py  # Combined outputs, input grouping
├── test_collapsed_pipelines_no_dangling_edges.py    # Edge compression
├── test_separate_outputs.py                          # Both display modes
├── test_state_utils_visibility.py                    # Visibility logic
├── test_debug_tools.py                               # Debug utilities
└── ...
```

## Data Flow

### 1. Python Side (Build Time)

```python
# 1. GraphWalker traverses the pipeline
walker = GraphWalker(pipeline, depth=2, traverse_collapsed=True)
graph = walker.walk()  # Returns VisualizationGraph

# 2. UIHandler wraps for serialization
handler = UIHandler(pipeline, depth=2)
graph_data = handler.get_visualization_data(traverse_collapsed=True)

# 3. JSRenderer transforms to React Flow format
renderer = JSRenderer()
rf_data = renderer.render(graph_data, theme='dark', separate_outputs=False)

# 4. HTML generator embeds data
html = generate_widget_html(rf_data)
```

### 2. JavaScript Side (Runtime)

```javascript
// 1. Parse embedded data
const initialData = JSON.parse(document.getElementById('graph-data').textContent);

// 2. Apply state transformations
const stateResult = applyState(nodes, edges, { expansionState, separateOutputs, showTypes, theme });

// 3. Apply visibility (hide children of collapsed pipelines)
const nodesWithVis = applyVisibility(stateResult.nodes, expansionState);

// 4. Compress edges (remap to visible ancestors)
const compressedEdges = compressEdges(nodesWithVis, stateResult.edges);

// 5. Group inputs (combine inputs targeting same node)
const { nodes, edges } = groupInputs(nodesWithVis, compressedEdges);

// 6. ELK layout → React Flow render
```

---

## Debugging Guide

### Quick Debug Commands

```bash
# Generate test HTML
uv run python scripts/test_collapsed_pipeline.py

# Run all viz tests
uv run pytest tests/viz/ -v

# Run specific test
uv run pytest tests/viz/test_collapsed_pipelines_no_dangling_edges.py -v
```

### Browser Console Debug API

The visualization exposes a debug API via `HyperNodesVizState.debug`:

```javascript
// Enable verbose logging for edge compression and layout
HyperNodesVizState.debug.enableDebug()

// Analyze current graph state (nodes, edges, pipelines, dangling edges)
HyperNodesVizState.debug.analyzeState()

// Get current pipeline expansion state
HyperNodesVizState.debug.getExpansionState()

// Simulate edge compression with a specific expansion state
HyperNodesVizState.debug.simulateCompression({ 'rag_pipeline': false })

// Disable debug mode
HyperNodesVizState.debug.disableDebug()

// === Visual Debug Overlays ===

// Show debug overlays (node bounding boxes + edge connection points)
HyperNodesVizState.debug.showOverlays()

// Hide debug overlays
HyperNodesVizState.debug.hideOverlays()

// Inspect current layout (node positions, dimensions, edge paths)
HyperNodesVizState.debug.inspectLayout()
// Returns: { nodes: [{id, x, y, width, height, bottom}], edges: [...], edgePaths: [...] }

// Validate edge-node connections (checks if edges connect within node bounds)
HyperNodesVizState.debug.validateConnections()
// Returns: { valid: bool, issues: [{edge, type, issue, expected, actual}], summary: str }

// Get comprehensive debug report (runs all analyses)
HyperNodesVizState.debug.fullReport()
```

### Visual Debug Mode

Debug overlays can be enabled in three ways:

1. **UI Toggle**: Click the bug icon in the view controls panel
2. **Console**: `HyperNodesVizState.debug.showOverlays()`
3. **URL Parameter**: Add `?debug=overlays` or `?debug=true` to the URL

When enabled, the visualization shows:
- **Red dashed boxes** around each node with ID labels
- **Green circles** at edge source points
- **Blue circles** at edge target points
- **Coordinate labels** on edges showing (sourceX, sourceY) → (targetX, targetY)

### Python State Simulator

The `state_simulator` module replicates JavaScript state transformations in Python for testing:

```python
from hypernodes.viz import (
    UIHandler,
    simulate_state,
    verify_state,
    verify_edge_alignment,
    simulate_collapse_expand_cycle,
    diagnose_all_states,
)

# Get visualization graph
handler = UIHandler(pipeline, depth=99)
graph = handler.get_visualization_data(traverse_collapsed=True)

# Simulate a specific state (collapsed pipeline, combined outputs)
result = simulate_state(
    graph,
    expansion_state={"my_pipeline": False},
    separate_outputs=False,
)

# Verify edges are valid
alignment = verify_edge_alignment(result)
if not alignment["valid"]:
    print("Issues:", alignment["issues"])

# Test a complete collapse/expand cycle
cycle = simulate_collapse_expand_cycle(graph, "my_pipeline")
print(cycle["summary"])  # "Pipeline 'my_pipeline' collapse/expand cycle: PASS (0 total issues)"

# Diagnose all state combinations
all_states = diagnose_all_states(graph)
for key, state in all_states.items():
    if state["orphan_edges"]:
        print(f"{key}: orphan edges found!")
```

### Debugging Each Layer

#### 1. GraphWalker (`graph_walker.py`)

**What it does**: Traverses the pipeline DAG and creates nodes/edges.

**Debug approach**:
```python
from hypernodes.viz.graph_walker import GraphWalker

walker = GraphWalker(pipeline, depth=2, traverse_collapsed=True)
graph = walker.walk()

# Inspect nodes
for node in graph.nodes:
    print(f"{node.__class__.__name__}: {node.id} (parent={node.parent_id})")

# Inspect edges
for edge in graph.edges:
    print(f"  {edge.source} → {edge.target}")
```

**Common issues**:
- Missing boundary outputs → Check `_expand_pipeline_node` creates them
- Wrong parent assignment → Check `parent_id` parameter passing
- Missing type hints → Check `_extract_input_type` recursive lookup

#### 2. UIHandler (`ui_handler.py`)

**What it does**: Manages depth, expansion state, and serialization.

**Debug approach**:
```python
from hypernodes.viz.ui_handler import UIHandler

handler = UIHandler(pipeline, depth=2)
data = handler.get_visualization_data(traverse_collapsed=True)

# Check what nodes exist
print("Nodes:", [n.id for n in data.nodes])
print("Pipelines:", [n.id for n in data.nodes if hasattr(n, 'is_expanded')])
```

**Key parameters**:
- `depth`: How many levels to expand initially
- `traverse_collapsed=True`: Pre-fetch internal structure for interactive expand/collapse

#### 3. JSRenderer (`js/renderer.py`)

**What it does**: Transforms VisualizationGraph → React Flow node/edge format.

**Debug approach**:
```python
from hypernodes.viz.js.renderer import JSRenderer

renderer = JSRenderer()
rf_data = renderer.render(graph_data, theme="dark", separate_outputs=False)

# Inspect React Flow data
import json
print(json.dumps(rf_data, indent=2))
```

**Key outputs**:
- `nodes[].data.nodeType`: FUNCTION, PIPELINE, DATA, INPUT_GROUP, DUAL
- `nodes[].data.isExpanded`: For PIPELINE nodes
- `nodes[].data.sourceId`: For output DATA nodes (points to producer)
- `nodes[].parentNode`: For nested nodes
- `edges[].sourcePosition/targetPosition`: Should be "bottom"/"top"

#### 4. state_utils.js (Frontend)

**What it does**: Client-side state transformations.

**Debug with Node.js**:
```javascript
// scripts/debug_state.js
const utils = require('../assets/viz/state_utils.js');
const fs = require('fs');

const html = fs.readFileSync('outputs/test.html', 'utf-8');
const match = html.match(/<script id="graph-data"[^>]*>([\s\S]*?)<\/script>/);
const data = JSON.parse(match[1]);

// Test transformations
const result = utils.applyState(data.nodes, data.edges, {
    expansionState: new Map([["rag_pipeline", false]]),
    separateOutputs: false,
    showTypes: true,
    theme: "dark"
});
console.log(JSON.stringify(result, null, 2));
```

Run: `node scripts/debug_state.js`

**Key functions**:
- `applyState`: Applies theme, separateOutputs mode, combines outputs
- `applyVisibility`: Hides children of collapsed pipelines
- `compressEdges`: Remaps edges to visible ancestors when pipelines collapse
- `groupInputs`: Groups inputs targeting the same node

#### 5. Playwright Browser Testing

For interactive debugging:
```python
# Use Playwright MCP tools
mcp_playwright_browser_navigate(url="file:///path/to/test.html")
mcp_playwright_browser_wait_for(time=2)  # Wait for React/ELK to render
mcp_playwright_browser_console_messages()  # Check for errors
mcp_playwright_browser_evaluate(function="() => HyperNodesVizState.debug.analyzeState()")
```

---

## Key Patterns

### Boundary Outputs

When a pipeline is collapsed, its outputs appear at the parent level:

```
Expanded:                          Collapsed:
┌─ rag_pipeline ─────────┐        ┌─ rag_pipeline ────┐
│  generate_answer       │   →    │  → answer : str   │
│      → answer          │        └───────────────────┘
└────────────────────────┘
```

Boundary output nodes have `sourceId` pointing to the PIPELINE:
```javascript
{ id: "answer", data: { sourceId: "rag_pipeline", nodeType: "DATA" } }
```

### Edge Compression

When pipelines collapse, edges to internal nodes remap to the collapsed pipeline:

```javascript
// getVisibleAncestor walks up parent chain to find collapsed pipeline
const sourceVis = getVisibleAncestor(edge.source);  // e.g., "rag_pipeline"
const targetVis = getVisibleAncestor(edge.target);
// Remap: edge to internal node → edge to collapsed pipeline
```

### Input Grouping

Inputs targeting the same node are grouped into INPUT_GROUP:
```
Before: eval_pair → rag_pipeline, model_name → rag_pipeline
After:  [eval_pair, model_name, num_results] → rag_pipeline
```

---

## Common Issues & Fixes

| Issue | What to Check | Fix Location |
|-------|---------------|--------------|
| Missing edges after collapse | `compressEdges` output, `getVisibleAncestor` | `state_utils.js` |
| Hanging/dangling arrows | Handle positions, node visibility, `updateNodeInternals` | `html_generator.py` |
| Edge starts/ends outside node | Use `validateConnections()` to diagnose | `state_utils.js` debug API |
| Stale edge paths after layout | Layout version, edge ID updates | `html_generator.py` |
| Nodes not grouping | `groupInputs`, target matching | `state_utils.js` |
| Outputs not combined | `applyState`, `sourceId` values | `state_utils.js` |
| Wrong node positions | ELK layout, `parentNode` | `html_generator.py` |
| Types missing on inputs | `_extract_input_type` | `graph_walker.py` |
| Pipeline outputs not shown | `functionOutputs` collection | `state_utils.js` |

### Debugging Edge Alignment Issues

When edges appear to "hang" or not connect properly:

1. **Enable debug overlays**: `HyperNodesVizState.debug.showOverlays()` or toggle in UI
2. **Inspect layout**: `HyperNodesVizState.debug.inspectLayout()` to see node positions
3. **Validate connections**: `HyperNodesVizState.debug.validateConnections()` to check bounds
4. **Check edge paths**: Look for `delta.y` in issues to see how far off edges are
5. **Run full report**: `HyperNodesVizState.debug.fullReport()` for comprehensive analysis

Common causes:
- React Flow caching edge paths after node resize
- Layout not triggering `updateNodeInternals`
- Asynchronous ELK layout completing after render

---

## Do's and Don'ts

### DO's ✅

1. **Use `traverse_collapsed=True`** for interactive viz (pre-fetches internal structure)
2. **Filter boundary outputs based on expansion state** (expanded→hide, collapsed→show)
3. **Remap edges when hiding nodes** (use `compressEdges`)
4. **Test both display modes** (`separateOutputs=true/false`)
5. **Preserve expansion state across option changes** (store separately)
6. **Verify with Playwright** for interactive behavior

### DON'TS ❌

1. **Don't skip edges from expanded pipelines** without remapping
2. **Don't assume all DATA nodes with sourceId are function outputs** (check sourceNodeTypes)
3. **Don't filter edges before remapping** (order matters)
4. **Don't create zero-size handles** (use opacity-0 instead)
5. **Don't modify nodes in place** (use spread for React state)

---

## Running Tests

```bash
# All visualization tests
uv run pytest tests/viz/ -v

# Specific test categories
uv run pytest tests/viz/test_collapsed*.py -v      # Collapse behavior
uv run pytest tests/viz/test_separate_outputs.py -v # Display modes
uv run pytest tests/viz/test_debug_tools.py -v     # Debug utilities

# Edge alignment tests (requires Playwright)
pip install playwright && playwright install chromium
uv run pytest tests/viz/test_edge_alignment_playwright.py -v

# Python-only edge alignment tests (no browser)
uv run pytest tests/viz/test_edge_alignment_playwright.py::TestPythonEdgeAlignment -v
```

### Playwright Browser Tests

The `test_edge_alignment_playwright.py` file contains tests that validate edge-node connections in a real browser:

```python
# Run tests that verify edges connect properly after collapse/expand
pytest tests/viz/test_edge_alignment_playwright.py::TestPlaywrightEdgeAlignment -v

# These tests:
# 1. Generate HTML visualization
# 2. Open in headless Chromium
# 3. Click to collapse/expand pipelines
# 4. Use debug.validateConnections() to verify alignment
# 5. Assert no issues found
```

## Generate Test HTML

```python
from hypernodes.viz.ui_handler import UIHandler
from hypernodes.viz.js.renderer import JSRenderer
from hypernodes.viz.js.html_generator import generate_widget_html

handler = UIHandler(pipeline, depth=2)
graph_data = handler.get_visualization_data(traverse_collapsed=True)
renderer = JSRenderer()
rf_data = renderer.render(graph_data, theme="dark", separate_outputs=False, show_types=True)
html = generate_widget_html(rf_data)

with open("outputs/test.html", "w") as f:
    f.write(html)
```
