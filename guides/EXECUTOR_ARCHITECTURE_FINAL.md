# HyperNodes Engine Architecture - Final Design

## Overview

The executor system has been refactored into an **Engine** architecture with clear separation between orchestration and execution.

---

## Core Concepts

### 1. Engine vs Executor

**Engine** (the orchestrator): Handles all the complex pipeline logic
- Dependency graph resolution and topological sorting
  - **Important**: The NetworkX DAG is constructed during `Pipeline.__init__()`, not during `.run()` or `.map()`
  - Graph construction happens once at pipeline creation time for optimal performance
- Per-node caching with signature computation
- Callback lifecycle management
- Selective execution based on output_name
- Implementations:
  - `HyperNodesEngine` - HyperNodes native node-by-node orchestration
  - `DaftEngine` - Framework-level execution using Daft DataFrames

**Executor** (the worker): Simple work submission following `concurrent.futures.Executor` protocol
- Just submits callables and returns futures - no pipeline knowledge
- HyperNodesEngine delegates to executors for parallelism:
  - `ThreadPoolExecutor(max_workers=N)` - Standard library thread pool
  - `ProcessPoolExecutor(max_workers=N)` - Standard library process pool
  - `AsyncExecutor(max_concurrent=N)` - Custom async executor for I/O-bound work
  - `"sequential"` - String alias for synchronous execution (no executor)
- Can be any class implementing the `concurrent.futures.Executor` protocol (`submit()`, `shutdown()`)

### 2. Architecture Diagram

```
┌─────────────────────────────────┐
│      Engine (ABC)               │  ← Base interface for orchestrators
│  - run(pipeline, inputs, ...)   │
│  - map(pipeline, inputs, ...)   │
└─────────────────────────────────┘
            ▲          ▲
            │          │
    ┌───────┴──────┐   │
    │              │   │
┌──────────────────────┐  ┌──────────────┐
│  HyperNodesEngine    │  │  DaftEngine  │
│                      │  │              │
│ Orchestrates:        │  │ (Framework)  │
│ - Dependency graph   │  │ Uses Daft    │
│ - Caching per node   │  │ DataFrame    │
│ - Callbacks          │  │ operations   │
│                      │  └──────────────┘
│ Delegates to:        │
│ ┌─────────────────┐ │
│ │ node_executor   │ │ ← ThreadPoolExecutor(max_workers=4)
│ │                 │ │   ProcessPoolExecutor(max_workers=4)
│ │ map_executor    │ │   AsyncExecutor(max_concurrent=10)
│ └─────────────────┘ │   "sequential" (default)
└──────────────────────┘
      ↓
Uses executors following concurrent.futures.Executor protocol
(Simple workers: submit(callable) → Future)
```

### 3. Pipeline Lifecycle: Construction vs Execution

**Construction Phase** (happens once during `Pipeline.__init__()`):
```python
pipeline = Pipeline(
    nodes=[node1, node2, node3],
    engine=engine,
    cache=cache,
)
# At this point:
# ✓ NetworkX DAG is fully constructed
# ✓ Dependencies are resolved
# ✓ Topological order is computed
# ✓ Pipeline is ready for execution
```

**Execution Phase** (happens during `.run()` or `.map()` calls):
```python
result = pipeline.run(inputs={"x": 5})
# At this point:
# ✓ Walks pre-constructed DAG in topological order
# ✓ Executes nodes with resolved dependencies
# ✓ Applies caching and callbacks
# ✗ Does NOT rebuild the graph
```

**Key Benefits:**
- Graph construction overhead is paid once, not per execution
- Multiple `.run()` or `.map()` calls reuse the same optimized DAG
- Enables fast repeated execution with different inputs

---

## API Design

### HyperNodesEngine

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from hypernodes import Pipeline, HyperNodesEngine
from hypernodes.executors import AsyncExecutor

# Option 1: String aliases (with sensible defaults)
engine = HyperNodesEngine(
    node_executor="sequential",    # Default: no parallelism
    map_executor="sequential",     # Default: no parallelism
)

# Option 2: Async for I/O-bound work
engine = HyperNodesEngine(
    node_executor="async",         # Creates AsyncExecutor(max_concurrent=2)
    map_executor="async",          # Creates AsyncExecutor(max_concurrent=2)
)

# Option 3: Executor instances for fine control
engine = HyperNodesEngine(
    node_executor=ThreadPoolExecutor(max_workers=4), #can also recieve "threaded" and place max_workers=2
    map_executor=ProcessPoolExecutor(max_workers=2), #can also recieve "parallel" and place max_workers=os.num_cpu() or something like that

# Option 4: Custom AsyncExecutor with different concurrency
engine = HyperNodesEngine(
    node_executor=AsyncExecutor(max_concurrent=20),
    map_executor="sequential",
)

# Use with pipeline
pipeline = Pipeline(
    nodes=[...],
    engine=engine,
    cache=cache,
    callbacks=callbacks,
)
```

### DaftEngine

```python
from hypernodes import Pipeline
from hypernodes.executors import DaftEngine

# Daft handles everything (graph, execution, optimization)
engine = DaftEngine(
    collect=True,        # Auto-collect results
    show_plan=False,     # Show execution plan
    debug=False,         # Debug mode
)

pipeline = Pipeline(
    nodes=[...],
    engine=engine,
    cache=cache,          # HyperNodes cache still used
    callbacks=callbacks,  # HyperNodes callbacks still fired
)
```

---

## String Aliases & Defaults

### Supported String Values

| String | Creates | Use Case |
|--------|---------|----------|
| `"sequential"` | No executor (sync) | Default, simple, predictable |
| `"async"` | `AsyncExecutor(max_concurrent=10)` | I/O-bound (API calls, file I/O) |
| `"threaded"` | `ThreadPoolExecutor(max_workers=cpu_count())` | Mixed I/O + CPU work |
| `"parallel"` | `ProcessPoolExecutor(max_workers=cpu_count())` | CPU-bound work |

### Default Parameters

**String Defaults:**
- `AsyncExecutor`: `max_concurrent=10`
- `ThreadPoolExecutor`: `max_workers=os.cpu_count()` or 4
- `ProcessPoolExecutor`: `max_workers=os.cpu_count()` or 4

**HyperNodesEngine Defaults:**
- `node_executor="sequential"` - No parallelism for nodes within `.run()`
- `map_executor="sequential"` - No parallelism for items in `.map()`

---

## Engine Interface

### Base Class (Abstract)

```python
class Engine(ABC):
    """Abstract base for all engines."""

    @abstractmethod
    def run(
        self,
        pipeline: Pipeline,
        inputs: Dict[str, Any],
        output_name: Optional[Union[str, List[str]]] = None,
        _ctx: Optional[CallbackContext] = None,  # Internal only
    ) -> Dict[str, Any]:
        """Execute pipeline with given inputs."""
        pass

    @abstractmethod
    def map(
        self,
        pipeline: Pipeline,
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        output_name: Optional[Union[str, List[str]]] = None,
        _ctx: Optional[CallbackContext] = None,  # Internal only
    ) -> List[Dict[str, Any]]:
        """Execute pipeline over multiple items."""
        pass
```

**Note:** `_ctx` is private and internal. Users never interact with it.

---

## AsyncExecutor Design

### Implementation

```python
from concurrent.futures import Future
import asyncio
from typing import Callable, Any

class AsyncExecutor:
    """Async executor for I/O-bound concurrent work.

    Compatible with concurrent.futures.Executor protocol but uses asyncio
    internally. Works in Jupyter notebooks by reusing existing event loops.

    Args:
        max_concurrent: Maximum number of concurrent tasks (default: 10)

    Example:
        >>> executor = AsyncExecutor(max_concurrent=20)
        >>> future = executor.submit(some_io_function, arg1, arg2)
        >>> result = future.result()
    """

    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent

    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit work to async executor.

        Returns a Future compatible with concurrent.futures protocol.
        """
        # Implementation handles Jupyter event loop reuse
        ...

    def shutdown(self, wait: bool = True):
        """Shutdown executor (for compatibility with Executor protocol)."""
        pass
```

### Jupyter Compatibility

The `AsyncExecutor` is designed to work in Jupyter notebooks without the common `RuntimeError: This event loop is already running` error:

1. Detects if an event loop is already running
2. Reuses existing loop instead of creating new one
3. Uses `nest_asyncio` if needed for nested scenarios

---

## Nested Map Parallelism Management

### Worker Reduction Strategy

HyperNodesEngine intelligently manages nested map operations to prevent resource explosion:

```python
def _calculate_effective_workers(self, num_items: int, map_depth: int) -> int:
    """Calculate effective worker count for nested maps."""
    if map_depth == 0:
        # Top-level map: use full worker count
        return min(self.max_workers, num_items)
    elif map_depth == 1:
        # First nested level: reduce to sqrt of max_workers
        return min(int(self.max_workers**0.5) or 1, num_items)
    else:
        # Deeper nesting: use sequential to avoid explosion
        return 1
```

**Example:**
- Outer map: 100 items with 10 workers
- Inner map (nested): 50 items with √10 ≈ 3 workers
- Total concurrent: 10 × 3 = 30 workers (not 500!)

---

## Usage Examples

### Example 1: Simple Sequential (Default)

```python
from hypernodes import Pipeline, HyperNodesEngine, node

@node(output_name="result")
def process(x: int) -> int:
    return x * 2

pipeline = Pipeline(
    nodes=[process],
    engine=HyperNodesEngine(),  # Both node and map are sequential
)

result = pipeline.run(inputs={"x": 5})
# {"result": 10}
```

### Example 2: Async for I/O-Bound Work

```python
import asyncio
from hypernodes import Pipeline, HyperNodesEngine, node

@node(output_name="data")
async def fetch_api(url: str) -> dict:
    # I/O-bound API call
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

pipeline = Pipeline(
    nodes=[fetch_api],
    engine=HyperNodesEngine(
        node_executor="async",  # Concurrent I/O for independent nodes
        map_executor="async",   # Concurrent I/O for map items
    ),
)

# Efficiently handles multiple concurrent API calls
results = pipeline.map(
    inputs={"urls": ["http://api1.com", "http://api2.com", ...]},
    map_over="urls",
)
```

### Example 3: Mixed Parallelism

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from hypernodes import Pipeline, HyperNodesEngine, node

@node(output_name="processed")
def cpu_heavy(data: list) -> list:
    # CPU-intensive work
    return [expensive_computation(x) for x in data]

@node(output_name="saved")
def save_to_db(processed: list) -> bool:
    # I/O-bound database save
    db.save(processed)
    return True

pipeline = Pipeline(
    nodes=[cpu_heavy, save_to_db],
    engine=HyperNodesEngine(
        node_executor=ThreadPoolExecutor(max_workers=4),  # Threads for mixed work
        map_executor=ProcessPoolExecutor(max_workers=2),  # Processes for CPU-heavy maps
    ),
)
```

### Example 4: Using Daft for Large-Scale Processing

```python
from hypernodes import Pipeline, node
from hypernodes.executors import DaftEngine

@node(output_name="cleaned")
def clean_text(text: str) -> str:
    return text.strip().lower()

@node(output_name="word_count")
def count_words(cleaned: str) -> int:
    return len(cleaned.split())

pipeline = Pipeline(
    nodes=[clean_text, count_words],
    engine=DaftEngine(collect=True),  # Daft handles everything
)

# Daft optimizes this into DataFrame operations
results = pipeline.map(
    inputs={"texts": ["Hello World", "Foo Bar", ...]},
    map_over="texts",
)
```

### Example 5: Nested Pipelines with Different Engines

```python
from hypernodes import Pipeline, HyperNodesEngine, node

# Inner pipeline: I/O-bound
inner = Pipeline(
    nodes=[fetch_data, parse_json],
    engine=HyperNodesEngine(
        node_executor="async",
        map_executor="async",
    ),
)

# Outer pipeline: CPU-bound
outer = Pipeline(
    nodes=[prepare, inner.as_node(map_over="urls"), aggregate],
    engine=HyperNodesEngine(
        node_executor="threaded",
        map_executor="parallel",
    ),
)

# Each pipeline has its own engine configuration
result = outer.run(inputs={"urls": [...]})
```

---

## Migration from Backend → Engine

### Old Code (Backend)

```python
from hypernodes import Pipeline, LocalBackend

pipeline = Pipeline(
    nodes=[...],
    backend=LocalBackend(
        node_execution="parallel",
        map_execution="threaded",
        max_workers=8,
    ),
)
```

### New Code (Engine)

```python
from hypernodes import Pipeline, HyperNodesEngine
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

pipeline = Pipeline(
    nodes=[...],
    engine=HyperNodesEngine(
        node_executor=ProcessPoolExecutor(max_workers=8),  # Or "parallel"
        map_executor=ThreadPoolExecutor(max_workers=8),    # Or "threaded"
    ),
)
```

---

## Implementation Checklist

### Phase 1: Core Refactoring ✅
- [x] Create `Executor` base class
- [x] Create `HyperNodesEngine` (rename from LocalExecutor)
- [x] Create `AsyncExecutor` class
- [x] Create `DaftEngine` (rename from DaftExecutor)
- [x] Remove `ModalExecutor` (guidance only)

### Phase 2: Engine Implementation 🔄
- [ ] Refactor `HyperNodesEngine` to accept executor instances or strings
- [ ] Implement string → executor mapping
- [ ] Update `_ctx` parameter naming
- [ ] Test AsyncExecutor in Jupyter

### Phase 3: Pipeline Integration
- [ ] Update `Pipeline.__init__` (`backend` → `engine`)
- [ ] Update `effective_engine` property
- [ ] Update `.run()` and `.map()` methods
- [ ] Update inheritance logic

### Phase 4: Testing & Documentation
- [ ] Update all test files
- [ ] Update documentation and examples
- [ ] Run full test suite
- [ ] Update README and guides

---

## Key Decisions Summary

| Decision | Rationale |
|----------|-----------|
| **Engine terminology** | Engines orchestrate (complex pipeline logic), executors are simple workers (concurrent.futures protocol) |
| **HyperNodesEngine** | Emphasizes it's the HyperNodes native orchestration layer |
| **DaftEngine** | Daft is a processing engine, not just an executor |
| **String aliases** | Convenience for common patterns with sensible defaults |
| **Executor instances** | Full control for advanced users - any class with concurrent.futures protocol |
| **AsyncExecutor** | First-class async support with Jupyter compatibility |
| **`max_concurrent`** | Proper terminology for async (not `max_workers`) |
| **`_ctx` parameter** | Private, users never interact with it |
| **Sequential default** | Simple, predictable, debuggable |
| **Worker reduction** | Intelligent nested map management prevents resource explosion |

---

## Notes

1. **No Modal Executor**: Modal integration is guidance-only via `guides/modal_functions.md`. Users wrap pipelines in Modal functions themselves.

2. **Cache & Callbacks**: Both `HyperNodesEngine` and `DaftEngine` use HyperNodes cache/callbacks for now. Future: may allow Daft to handle its own.

3. **Backwards Compatibility**: Not required. This is a breaking change but with clear migration path.

4. **Protocol Compatibility**: Worker executors (ThreadPoolExecutor, ProcessPoolExecutor, AsyncExecutor) follow `concurrent.futures.Executor` protocol. This keeps them simple and interchangeable.

5. **Jupyter First**: AsyncExecutor designed specifically to work in Jupyter notebooks without event loop issues.
