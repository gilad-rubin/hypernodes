# HyperNodes Architecture Documentation

**Project:** hypernodes v0.4.8  
**Type:** Python Library  
**Architecture Pattern:** Modular Library with Pluggable Execution Engines  
**Generated:** 2025-12-22

---

## Executive Summary

HyperNodes is a cache-first pipeline framework for ML/AI workflows. It enables building DAGs from decorated functions with automatic dependency resolution, intelligent caching, and pluggable execution engines. The architecture follows SOLID principles with a zero-dependency core and optional feature packages.

**Key Design Principles:**
- **Cache-first**: Content-addressable caching for reproducible pipelines
- **Think Singular**: Write logic for one item, scale with `.map()`
- **Hierarchical Composition**: Pipelines are nodes, enabling unlimited nesting
- **Pluggable Engines**: Sequential, distributed (Daft), or parallel (Dask) execution
- **Zero Dependencies**: Core functionality has no external dependencies

---

## Technology Stack

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Core Language** | Python | 3.10+ | Implementation language |
| **Package Manager** | uv | Latest | Modern package/project manager |
| **Build System** | Hatchling | Latest | PEP 517 compliant build backend |
| **Core Dependencies** | None | - | Zero-dependency core |
| **Distributed Engine** | Daft | >=0.6.11 | Optional: DataFrame-based distributed computing |
| **Parallel Engine** | Dask | Latest | Optional: Parallel map operations |
| **Batch Processing** | PyArrow | >=14.0.0 | Optional: Columnar data |
| **Visualization** | Graphviz | >=0.20 | Optional: Static pipeline viz |
| **Interactive Viz** | React Flow + ELK | Bundled | JavaScript-based interactive viz |
| **Notebook Support** | Jupyter + IPyWidgets | >=8.1.7 | Optional: Interactive widgets |
| **Telemetry** | Logfire | >=2.0.0 | Optional: Distributed tracing |
| **Progress** | tqdm + rich | Latest | Optional: Progress bars |
| **Testing** | pytest + playwright | >=8.4.2 | Test framework + browser testing |

---

## Architecture Pattern

**Pattern:** Modular Library with Protocol-Based Extension Points

```
┌─────────────────────────────────────────────────────────────┐
│                    User Code (Pipelines)                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Public API Layer                          │
│  Pipeline | @node | @branch | SeqEngine | DiskCache          │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Core Execution                            │
│  GraphBuilder → Orchestrator → NodeExecution                │
│  ├─ DAG Construction                                         │
│  ├─ Lifecycle Management                                     │
│  └─ Single Node Execution                                    │
└──────────┬──────────────────────────────────┬───────────────┘
           │                                  │
┌──────────▼────────────┐         ┌──────────▼───────────────┐
│  Cache System         │         │  Callback System         │
│  ├─ Signature Compute │         │  ├─ Progress Tracking    │
│  ├─ DiskCache         │         │  └─ Tracing (Logfire)    │
│  └─ Custom Backends   │         └──────────────────────────┘
└───────────────────────┘
           │
┌──────────▼────────────────────────────────────────────────┐
│                  Execution Engines (Protocol)              │
│  ├─ SeqEngine (built-in)                                   │
│  ├─ DaftEngine (integrations/daft/) - Distributed         │
│  └─ DaskEngine (integrations/dask/) - Parallel            │
└────────────────────────────────────────────────────────────┘
           │
┌──────────▼────────────────────────────────────────────────┐
│              Visualization System                          │
│  ├─ GraphWalker (traversal)                                │
│  ├─ UIHandler (state management)                           │
│  ├─ Graphviz Renderer (static SVG)                         │
│  └─ React Flow Renderer (interactive HTML)                 │
└────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Node (`node.py`)

Wraps a Python function with pipeline metadata.

**Key Properties:**
- `func`: The wrapped function
- `output_name`: Name(s) of outputs (str or tuple)
- `root_args`: Tuple of input parameter names
- `code_hash`: Cached SHA256 hash of function source
- `cache`: Whether to cache this node's output

**Usage:**
```python
@node(output_name="result", cache=True)
def process(x: int) -> int:
    return x * 2
```

### 2. Pipeline (`pipeline.py`)

Manages a DAG of nodes. Pure definition - no execution state.

**NEW ARCHITECTURE:** Pipeline no longer holds `cache` or `callbacks`. These are configured at the engine level.

**Key Methods:**
- `run()`: Execute pipeline once with given inputs
- `map()`: Execute pipeline multiple times over collections
- `as_node()`: Wrap pipeline as a node for nesting
- `visualize()`: Generate visualization (JS interactive or Graphviz SVG)
- `bind()`: Set default input values

### 3. Branch Nodes (`branch.py`)

Conditional execution routing based on boolean conditions.

**How it works:**
- Branch nodes produce mutually-exclusive "gate signals"
- Target nodes implicitly depend on these gate signals
- Only the winning path executes; nodes in the losing path are skipped
- Both paths can produce the same output name (exclusive producers)

**Visualization:** Rendered as diamond shapes with True/False edge labels

### 4. Execution Engines

#### Engine Protocol (`protocols.py`)

All engines implement the `Engine` protocol with `run()` and `map()` methods.

#### SeqEngine (`sequential_engine.py`)

Default sequential execution engine. Simple topological execution, no parallelism.

**Owns:**
- Cache backend instance
- Callback instances
- Branch routing logic (tracks satisfied gates)

#### DaftEngine (`integrations/daft/engine.py`)

High-performance distributed execution facade.

**Features:**
- Lazy DataFrame execution
- Auto-optimized batch sizes (64-1024)
- Auto-tuned parallelism (8-16x CPU cores)
- Stateful parameter handling
- Per-item caching in map mode

**Architecture:**
- **Engine**: Orchestration and caching
- **Operations** (`operations.py`): Modular strategies (Function, Batch, Pipeline, Dual)
- **CodeGen** (`codegen.py`): Tracks imports and generates standalone Daft scripts

#### DaskEngine (`integrations/dask/engine.py`)

Parallel map operations using Dask Bag.

### 5. Cache System (`cache.py`)

Content-addressable caching using computation signatures.

**Signature Formula:**
```
sig(node) = hash(code_hash + env_hash + inputs_hash + deps_hash)
```

**Cache Implementations:**
- `DiskCache`: Pickle-based filesystem cache
- Custom backends can be implemented

**Cache Hierarchy:**
1. **Engine Level**: `engine.cache` - the backend instance
2. **Node Level**: `node.cache` (True/False) - whether this node should be cached
3. **Effective**: Caching happens if `engine.cache is not None and node.cache is True`

### 6. Callback System (`callbacks.py`, `orchestrator.py`)

Lifecycle hooks for observability. All callbacks inherit from `PipelineCallback`.

**Lifecycle Events:**
- `on_pipeline_start/end`
- `on_node_start/end`
- `on_node_cached` (cache hit)
- `on_branch_decision` (branch node evaluated)
- `on_node_skipped` (node skipped due to branch routing)
- `on_map_start/end`
- `on_map_item_start/end`
- `on_nested_pipeline_start/end`

**Available Callbacks:**
- `ProgressCallback` (`telemetry/progress.py`): Live tqdm progress bars (auto-detects Jupyter vs CLI)
- `TelemetryCallback` (`telemetry/tracing.py`): Distributed tracing with Logfire

**Engine Compatibility:**
Callbacks can declare which engines they support via `supported_engines` property.

### 7. ExecutionOrchestrator (`orchestrator.py`)

Shared lifecycle management for all engines.

**Responsibilities:**
- CallbackDispatcher setup
- Pipeline metadata tracking
- Start/End event notifications
- Callback/engine compatibility validation

**Benefits:**
- Consistent behavior across all engines
- No code duplication
- Easy to add new engines

### 8. Visualization System (`viz/`)

Comprehensive visualization with two rendering modes.

**Architecture:**

```
Python:  GraphWalker → UIHandler → JSRenderer → html_generator
                                         ↓
Browser: JSON → applyState → applyVisibility → compressEdges → groupInputs → ELK → ReactFlow
```

**Key Components:**
- **GraphWalker** (`graph_walker.py`): Traverses pipeline DAG, generates flat node/edge structure
- **UIHandler** (`ui_handler.py`): Manages depth, expansion state, serialization
- **Renderers**:
  - **Graphviz** (`graphviz/renderer.py`): Static SVG generation
  - **React Flow** (`js/renderer.py`, `js/html_generator.py`): Interactive HTML
- **Assets** (`assets/`): Bundled JS/CSS (React 18, React Flow 11, ELK 0.8) - **NO CDN dependencies**

**Features:**
- Expand/collapse nested pipelines
- Separate or combined output display modes
- Type annotations on inputs
- Branch nodes rendered as diamonds
- Theme detection (auto-adapts to VS Code, JupyterLab, Marimo)
- Debug mode with edge/node validation

---

## Data Architecture

**No Database** - This is a library package, not an application.

**Caching Storage:**
- Content-addressable cache stored on disk (DiskCache)
- Pickle serialization for Python objects
- Cache keys are deterministic signatures

---

## API Design

### Public API Surface (`__init__.py`)

**Core Classes:**
- `Pipeline` - Main pipeline class
- `Node` - Node wrapper (created by `@node`)
- `SeqEngine` - Sequential execution engine

**Decorators:**
- `@node` - Convert function to node
- `@branch` - Create conditional branch node

**Cache:**
- `DiskCache` - Filesystem-based cache

**Callbacks:**
- `ProgressCallback` - Progress tracking
- `TelemetryCallback` - Distributed tracing

**Optional Engines:**
- `DaftEngine` (from `hypernodes.engines`)
- `DaskEngine` (from `hypernodes.engines`)

### Patterns

**Basic Pipeline:**
```python
from hypernodes import Pipeline, node, SeqEngine, DiskCache

@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

engine = SeqEngine(cache=DiskCache(path=".cache"))
pipeline = Pipeline(nodes=[double], engine=engine)
result = pipeline.run(inputs={"x": 5})  # {"doubled": 10}
```

**Mapping (Scale):**
```python
results = pipeline.map(
    inputs={"x": [1, 2, 3]},
    map_over="x"
)
# [{"doubled": 2}, {"doubled": 4}, {"doubled": 6}]
```

**Nesting:**
```python
inner = Pipeline(nodes=[clean])
outer = Pipeline(nodes=[inner.as_node(), analyze])
result = outer.run(inputs={"text": "hello"})
```

**Binding Defaults:**
```python
pipeline = Pipeline(nodes=[scale]).bind(factor=10)
result = pipeline.run(inputs={"value": 5})  # Uses bound factor=10
```

---

## Component Overview

### Source Tree

```
src/hypernodes/
├── __init__.py                  # Public API exports
├── pipeline.py                  # Pipeline class (DAG orchestration)
├── node.py                      # Node decorator and class
├── cache.py                     # Content-addressable caching
├── engines.py                   # Engine protocol definitions
├── sequential_engine.py         # Default sequential engine
├── callbacks.py                 # Callback protocol
├── graph_builder.py             # DAG construction
├── orchestrator.py              # Execution lifecycle
├── node_execution.py            # Single node execution
├── branch.py                    # Branch nodes (conditional)
├── dual_node.py                 # Dual singular/batch nodes
├── pipeline_node.py             # Nested pipeline wrapper
├── hypernode.py                 # HyperNode protocol
├── map_planner.py               # Map operation planning
├── batch_adapter.py             # Batch processing adapters
├── protocols.py                 # Protocol definitions
├── decorators.py                # Decorator utilities
├── exceptions.py                # Custom exceptions
│
├── integrations/                # Pluggable execution backends
│   ├── daft/                    # Distributed DataFrame execution
│   │   ├── engine.py            # DaftEngine facade
│   │   ├── operations.py        # Modular Daft operations
│   │   └── codegen.py           # Code generation
│   └── dask/                    # Parallel execution
│       └── engine.py            # DaskEngine implementation
│
├── viz/                         # Visualization system
│   ├── ui_handler.py            # State management
│   ├── graph_walker.py          # Graph traversal
│   ├── structures.py            # Data classes
│   ├── visualization_engine.py  # Renderer registry
│   ├── graphviz/                # Static SVG rendering
│   │   ├── renderer.py
│   │   └── style.py
│   ├── js/                      # Interactive HTML rendering
│   │   ├── html_generator.py    # HTML generation
│   │   └── renderer.py          # React Flow transformation
│   ├── assets/                  # Bundled JS/CSS (offline)
│   │   ├── react*.js
│   │   ├── reactflow.*
│   │   ├── elk.bundled.js
│   │   ├── state_utils.js
│   │   └── tailwind.min.css
│   └── state_simulator.py      # Python state simulator (testing)
│
└── telemetry/                   # Observability
    ├── progress.py              # Progress bars (tqdm/rich)
    ├── tracing.py               # Logfire distributed tracing
    ├── waterfall.py             # Waterfall diagrams
    └── environment.py           # Environment detection
```

### Module Responsibilities

| Module | Responsibility | Dependencies |
|--------|----------------|--------------|
| `pipeline.py` | DAG definition, run/map API | `graph_builder`, `node` |
| `node.py` | Function wrapping, metadata | None (core) |
| `cache.py` | Signature computation, storage | None (core) |
| `sequential_engine.py` | Sequential execution | `orchestrator`, `node_execution` |
| `graph_builder.py` | Dependency resolution, DAG construction | `node` |
| `orchestrator.py` | Lifecycle management, callbacks | `callbacks` |
| `node_execution.py` | Single node execution logic | `cache` |
| `branch.py` | Conditional routing | `node` |
| `viz/` | Visualization rendering | Graphviz (optional), React (bundled) |
| `telemetry/` | Progress and tracing | tqdm, logfire (optional) |
| `integrations/` | Alternative engines | daft, dask (optional) |

---

## Development Workflow

### Installation

```bash
# Install for development
uv sync

# Install with all features
uv add hypernodes[all]
```

### Running Tests

```bash
# All tests
uv run pytest

# Specific category
uv run pytest tests/test_execution.py
uv run pytest tests/viz/
uv run pytest tests/test_daft_*.py
```

### Building

```bash
# Build distribution
uv build

# Output: dist/hypernodes-0.4.8-py3-none-any.whl
```

---

## Deployment Architecture

**Distribution:** PyPI package

**Installation Methods:**
1. `pip install hypernodes`
2. `uv add hypernodes`

**Runtime Environments:**
- Local development
- Jupyter notebooks
- Cloud (with Modal integration)
- Distributed clusters (with Daft)

**No Server Component** - Pure library package

---

## Testing Strategy

### Test Organization

**Test Framework:** pytest with asyncio support

**Test Categories:**
- **Core Execution** (`test_execution.py`) - Basic pipeline functionality
- **Map Operations** (`test_map.py`) - Collection processing
- **Caching** (`test_caching.py`) - Cache behavior and invalidation
- **Callbacks** (`test_callbacks.py`) - Lifecycle hooks
- **Nested Pipelines** (`test_nested_pipelines.py`) - Composition
- **Branch Nodes** (`test_branch.py`, `test_branch_same_output.py`) - Conditional execution
- **Engines** (`test_daft_*.py`, `test_benchmarks_engines.py`) - Engine implementations
- **Visualization** (`viz/test_*.py` - 27 files) - Rendering, layout, interactions (includes Playwright browser tests)
- **Stateful Objects** (`test_stateful*.py`) - Stateful parameter handling
- **Binding** (`test_bind.py`) - Input binding

### Test Infrastructure

**Tools:**
- pytest >= 8.4.2
- playwright >= 1.56.0 (browser testing)

**Configuration:**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
norecursedirs = ["old", ".*", "build", "dist", "*.egg"]
```

---

## Security Considerations

**As a library package:**

1. **Code Execution:**
   - Executes user-provided functions (by design)
   - Cache uses pickle (trusted code only)

2. **Dependency Security:**
   - Zero-dependency core minimizes attack surface
   - Optional dependencies audited

3. **Sandboxing:**
   - No built-in sandboxing (library, not platform)
   - Users responsible for function safety

---

## Performance Characteristics

### Scaling Dimensions

**Single Machine (SeqEngine):**
- Simple topological execution
- Good for: Development, small datasets, debugging
- Scales with: Number of CPU cores (no parallelism in SeqEngine itself)

**Distributed (DaftEngine):**
- Lazy DataFrame execution
- Auto-batching with optimal batch sizes
- Auto-tuned parallelism (8-16x CPU cores)
- Good for: Large datasets, production workloads
- Scales with: Cluster size, data partitioning

**Parallel (DaskEngine):**
- Parallel map operations
- Good for: CPU-bound map tasks
- Scales with: Number of workers

### Caching Impact

**Cache Effectiveness:**
- First run: Full execution
- Repeated run (same inputs): Instant (cache hit)
- Incremental run (few new items): Only new items execute
- Code change: Only affected nodes re-execute

**Cache Overhead:**
- Signature computation: ~1ms per node
- Disk I/O: Depends on result size

---

## Extension Points

### 1. Custom Engines

Implement the `Engine` protocol:

```python
from hypernodes.protocols import Engine

class MyEngine(Engine):
    def run(self, pipeline, inputs):
        # Custom execution logic
        pass
    
    def map(self, pipeline, inputs, map_over, map_mode):
        # Custom map logic
        pass
```

### 2. Custom Cache Backends

Extend cache interface:

```python
from hypernodes.cache import CacheBackend

class RedisCache(CacheBackend):
    def get(self, signature):
        # Retrieve from Redis
        pass
    
    def set(self, signature, value):
        # Store in Redis
        pass
```

### 3. Custom Callbacks

Inherit from `PipelineCallback`:

```python
from hypernodes.callbacks import PipelineCallback

class MyCallback(PipelineCallback):
    def on_node_start(self, context):
        # Custom logic
        pass
```

### 4. Custom Visualization Renderers

Implement `VisualizationEngine` protocol:

```python
from hypernodes.viz.visualization_engine import VisualizationEngine

class MyRenderer(VisualizationEngine):
    def render(self, graph_data, **kwargs):
        # Custom rendering logic
        pass
```

---

## Known Limitations

1. **Cache Invalidation:**
   - Relies on source code hashing (may miss some edge cases)
   - Environment variables captured but not monitored

2. **Branch Node Limitations:**
   - Boolean conditions only (no multi-way branching)
   - Condition evaluation happens at runtime (can't optimize away)

3. **Visualization:**
   - Large graphs (>100 nodes) may be slow to render in browser
   - Graphviz has layout limitations for complex graphs

4. **Daft Engine:**
   - Requires getdaft package installation
   - Some Python features not serializable to Daft UDFs

---

## Future Roadmap

**Based on current architecture:**

- **Distributed Caching:** Redis/S3 cache backends
- **Multi-way Branching:** Beyond boolean conditions
- **Pipeline Optimization:** Static analysis and optimization passes
- **Streaming:** Support for streaming data sources
- **Type Safety:** Full type checking with mypy
- **Performance:** JIT compilation for hot paths

---

**Last Updated:** 2025-12-22  
**Version:** 0.4.8  
**Status:** Alpha (Development Status :: 3 - Alpha)

