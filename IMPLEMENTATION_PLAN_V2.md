# HyperNodes Executor Architecture - Implementation Plan v2.0

Date: 2025-11-05

Status: Planning Phase - User-Facing API First

## Executive Summary

Refactor HyperNodes to cleanly separate WHERE code runs (Backend) from HOW it runs (Executor Strategy), with a simple, intuitive, and non-conflicting API.

**Key Principles:**

1. **Simple defaults work** - No config needed for basic use
2. **Progressive complexity** - Advanced features are opt-in
3. **Lazy initialization** - Resources allocated only when needed
4. **Clean separation** - WHERE (backend) is independent from HOW (executor)
5. **Familiar API** - Keep existing `DiskCache`, `ProgressCallback`, etc.
6. **Strategy Pattern** - A single `executor` parameter defines the entire "HOW"

## 1. Design Principles & Goals

### 1.1 Core Principles

**Principle 1: Simplicity First**

```python
# This just works - no config needed
# Defaults to executor=SequentialNodeExecutor()
pipeline = Pipeline(nodes=[load, process, save])
result = pipeline.run({"file": "data.csv"})

```

**Principle 2: Progressive Complexity**

```python
from concurrent.futures import ThreadPoolExecutor
from hypernodes.executor import SequentialNodeExecutor

# Add caching
pipeline.with_cache(DiskCache(".cache"))

# Add parallel map execution
pipeline.with_executor(
    SequentialNodeExecutor(map_executor=ThreadPoolExecutor(max_workers=8))
)

# Configure all at once
pipeline = Pipeline(
    nodes=[...],
    backend=ModalBackend(image=my_image, gpu="A100"),
    executor=SequentialNodeExecutor(map_executor=ThreadPoolExecutor(max_workers=8)),
    cache=DiskCache(".cache"),
    callbacks=[ProgressCallback()]
)

```

**Principle 3: Clean Separation (WHERE vs HOW)**

```python
# WHERE: Backend defines execution environment
backend = LocalBackend()  # This machine
backend = ModalBackend(gpu="A100")  # Modal with GPU

# HOW: ExecutorStrategy defines execution strategy
# A single object bundles all "HOW" logic
executor = SequentialNodeExecutor()  # Fully sequential
executor = SequentialNodeExecutor(map_executor=ThreadPoolExecutor(max_workers=8)) # Nodes sequential, Map parallel
executor = EagerNodeExecutor(map_executor=ThreadPoolExecutor(max_workers=8)) # Nodes eager/async, Map parallel
executor = DaftExecutor(runner="ray") # Entire pipeline handled by Daft

# Combine independently
pipeline = Pipeline(
    nodes=[...],
    backend=ModalBackend(gpu="A100"),  # WHERE
    executor=EagerNodeExecutor(map_executor=ThreadPoolExecutor(max_workers=20))  # HOW
)

```

**Principle 4: Familiar API with Lazy Init**

```python
# Keep existing API - just make them lazy internally
cache = DiskCache(".cache")  # NO file handles opened yet
callback = ProgressCallback()  # NO rich Progress created yet

# Resources allocated when pipeline.run() is called
pipeline.with_cache(cache).with_callbacks([callback])
result = pipeline.run(inputs)  # now the backend handles materialization

```

### 1.2 Goals

**Goal 1: Decouple Backend from Execution Strategy**

- Current: `LocalBackend(node_execution="threaded")` mixes WHERE and HOW
- Target: `LocalBackend()` + `executor=SequentialNodeExecutor(map_executor=...)` are separate

**Goal 2: Enable Serialization**

- Current: Cache/callbacks have file handles, can't pickle
- Target: Lazy init means config is always serializable

**Goal 3: Simplify Testing**

- Current: Hard to test execution strategies independently
- Target: `ExecutorStrategy` is a standalone, testable component

## 2. User-Facing API Examples

### 2.1 Simple (Beginner)

**Example 1: Zero Configuration (Default)**

```python
from hypernodes import Pipeline, node
from hypernodes.executor import SequentialNodeExecutor

@node(output_name="result")
def process(data: str) -> str:
    return data.upper()

# The user provides no execution strategy
pipeline = Pipeline(nodes=[process])
# pipeline.executor is SequentialNodeExecutor() by default
result = pipeline.run({"data": "hello"})

# Node Execution: Sequential
# Map Execution: Sequential

```

**Example 2: Add Caching**

```python
from hypernodes import Pipeline, DiskCache

pipeline = Pipeline(nodes=[process]).with_cache(DiskCache(".cache"))
# or Pipeline(nodes=[process], cache=DiskCache(".cache"))
result = pipeline.run({"data": "hello"})
result = pipeline.run({"data": "hello"}) # Cached

```

**Example 3: Add Progress Bar**

```python

from hypernodes import Pipeline
from hypernodes.telemetry import ProgressCallback

pipeline = Pipeline(nodes=[load, process, save])
pipeline.with_callbacks([ProgressCallback()])
result = pipeline.run({"file": "data.csv"}) # Shows progress

```

### 2.2 Intermediate

**Example 4: Parallel Map (Most Common)**

```python

from hypernodes import Pipeline
from hypernodes.executor import SequentialNodeExecutor
from concurrent.futures import ThreadPoolExecutor

# Create a worker pool for map items
map_pool = ThreadPoolExecutor(max_workers=8)

# Pass the pool to the 'SequentialNodeExecutor' strategy's 'map_executor'
pipeline = Pipeline(
    nodes=[fetch, process, save],
    executor=SequentialNodeExecutor(map_executor=map_pool)
)

result = pipeline.map({"url": ["http://...", ]}, map_over="url")

# Node Execution: Sequential (one-by-one)
# Map Execution: Parallel (in ThreadPoolExecutor)

```

**Example 5: Eager Nodes (Async Graph)**

```python

from hypernodes import Pipeline
from hypernodes.executor import EagerNodeExecutor

# Use the Eager strategy for concurrent node execution
pipeline = Pipeline(
    nodes=[fetch_api, process],
    executor=EagerNodeExecutor()
)
result = pipeline.run({"url": "..."})

# Node Execution: Eager (async/concurrent graph)
# Map Execution: Sequential (map_executor is None)

```

**Example 6: Max Parallelism (Eager Nodes + Parallel Map)**

```python

from hypernodes import Pipeline
from hypernodes.executor import EagerNodeExecutor
from concurrent.futures import ProcessPoolExecutor

# A pool for CPU-bound map items
cpu_map_pool = ProcessPoolExecutor(max_workers=4)

pipeline = Pipeline(
    nodes=[load, process, save],
    executor=EagerNodeExecutor(map_executor=cpu_map_pool)
)
result = pipeline.map({"files": ["file1.csv", "file2.csv"]})

# Node Execution: Eager (async/concurrent graph)
# Map Execution: Parallel (in ProcessPoolExecutor)

```

**Example 7: Global Configuration**

```python

from hypernodes import set_default_config, DiskCache, ProgressCallback
from hypernodes.executor import SequentialNodeExecutor
from concurrent.futures import ThreadPoolExecutor

# Set once for entire application
set_default_config(
    backend=LocalBackend(),
    executor=SequentialNodeExecutor(map_executor=ThreadPoolExecutor(max_workers=8)),
    cache=DiskCache(".cache"),
    callbacks=[ProgressCallback()]
)

# All new pipelines use these defaults
pipeline1 = Pipeline(nodes=[...])
pipeline2 = Pipeline(nodes=[...])

```

### 2.3 Advanced

**Example 8: Holistic Executor (Daft)**

```python

from hypernodes import Pipeline, DaftBackend
from hypernodes.executor import DaftExecutor

# The Daft strategy tells the backend to convert the pipeline
# The DaftBackend knows how to interpret this strategy
pipeline = Pipeline(
    nodes=[train, evaluate],
    backend=DaftBackend(),
    executor=DaftExecutor(runner="ray", ray_address="...")
)

# The DaftBackend's .run() method will use the Daft strategy
result = pipeline.run({"data": "train.csv"})

# Node Execution: Holistic (handled by Daft)
# Map Execution: Holistic (handled by Daft)

```

**Example 9: Modal with Eager Nodes and Parallel Map**

```python

import modal
from hypernodes import Pipeline, ModalBackend
from hypernodes.executor import EagerNodeExecutor
from concurrent.futures import ThreadPoolExecutor

image = modal.Image.debian_slim().pip_install("numpy", "pandas")

pipeline = Pipeline(
    nodes=[train, evaluate],
    backend=ModalBackend(image=image, gpu="A100"),
    executor=EagerNodeExecutor(map_executor=ThreadPoolExecutor(max_workers=16))
)

result = pipeline.run({"data": "train.csv"})

# Node Execution: Eager (async/concurrent graph on Modal)
# Map Execution: Parallel (in ThreadPoolExecutor on Modal)

```

**Example 10: Nested Pipelines with Different Configs**

```python

from hypernodes import Pipeline, ModalBackend, LocalBackend
from hypernodes.executor import SequentialNodeExecutor, EagerNodeExecutor

# Inner: GPU on Modal, sequential nodes
gpu_pipeline = Pipeline(
    nodes=[tokenize, embed],
    backend=ModalBackend(gpu="A100"),
    executor=SequentialNodeExecutor()
)

# Outer: CPU locally, eager nodes
outer = Pipeline(
    nodes=[load, gpu_pipeline.as_node(), save],
    backend=LocalBackend(),
    executor=EagerNodeExecutor()
)

result = outer.run({"files": ["doc1.txt", "doc2.txt"]})

```

**Example 11: Override in as_node()**

```python

from hypernodes import Pipeline, DiskCache
from hypernodes.executor import SequentialNodeExecutor, EagerNodeExecutor

# Inner pipeline with its own config
inner = Pipeline(
    nodes=[tokenize, embed],
    executor=EagerNodeExecutor(),
    cache=DiskCache(".cache")
)

# Override execution strategy and disable cache when wrapping as node
inner_node = inner.as_node(
    input_mapping={"doc": "text"},
    executor=SequentialNodeExecutor(),  # Override executor
    cache=None                      # Disable cache
)

# or
inner_node = (
    inner.as_node(input_mapping={"doc": "text"})
    .with_executor(SequentialNodeExecutor())
    .with_cache(None)
)

outer = Pipeline(nodes=[load, inner_node, save])
result = outer.run({"doc": "hello"})

```

**Example 12: Configuration Precedence**

```python

from hypernodes import Pipeline, set_default_config, DiskCache
from hypernodes.executor import SequentialNodeExecutor, EagerNodeExecutor
from concurrent.futures import ThreadPoolExecutor

# Level 1: Library default
# executor=SequentialNodeExecutor() cache=None
pipeline = Pipeline(nodes=[process])

# Level 2: Global default
set_default_config(
    executor=SequentialNodeExecutor(map_executor=ThreadPoolExecutor(max_workers=4)),
    cache=DiskCache(".cache")
)
# Now uses SequentialNodeExecutor(map_executor=...) and caching

# Level 3: Pipeline config
pipeline.with_executor(EagerNodeExecutor(map_executor=ThreadPoolExecutor(max_workers=8)))
# Now uses EagerNodeExecutor(map_executor=...)

# Level 4: as_node() override (highest for nested)
outer_node = pipeline.as_node(executor=SequentialNodeExecutor())
# This node uses SequentialNodeExecutor(), overriding pipeline config

```

## 3. Architecture

### 3.1 Three-Layer Design

```
Pipeline (Holds all config)
  ├── backend: Backend (WHERE)
  ├── executor: ExecutorStrategy (HOW)
  ├── cache: Cache (lazy init internally)
  ├── callbacks: List[Callback] (lazy init internally)
  └── retries, timeout

Backend (WHERE)
  ├── LocalBackend() - just environment
  └── ModalBackend(image, gpu, ...) - infrastructure only

ExecutorStrategy (HOW) - A single object defines the "how"
  ├── TaskExecutor (Base for task-based execution)
  │     ├── map_executor: concurrent.futures.Executor
  │     │
  │     ├── SequentialNodeExecutor(map_executor=...)
  │     │       - Node Execution: Sequential
  │     │       - Map Execution: Parallel (if map_executor set)
  │     │
  │     └── EagerNodeExecutor(map_executor=...)
  │             - Node Execution: Eager (async)
  │             - Map Execution: Parallel (if map_executor set)
  │
  └── HolisticExecutor (Base for framework-based execution)
        │
        └── DaftExecutor(runner=...)
                - Node Execution: Handled by Daft
                - Map Execution: Handled by Daft

Cache (Familiar API, lazy internally)
  ├── DiskCache(path) - stores path, opens on first use
  └── MemoryCache(max_size) - stores config, allocates on use

Callback (Familiar API, lazy internally)
  ├── ProgressCallback(style) - stores config, creates UI on use
  └── TelemetryCallback(service) - stores config, connects on use

```

### 3.2 Config Precedence

```
as_node() override:   inner.as_node(executor=X, cache=Y) [HIGHEST]
        ↓
Pipeline config:      pipeline.with_executor(X)
        ↓
Global config:        set_default_config(executor=X, cache=Y)
        ↓
Library default:      LocalBackend + executor=SequentialNodeExecutor() [LOWEST]

Note: pipeline.run() is clean - no call-site overrides

```

## 4. Implementation Plan

### Phase 1: Core Abstractions (Week 1)

**Files to Create:**

- `src/hypernodes/executor.py` - `ExecutorStrategy` base class and subclasses.
- `src/hypernodes/config.py` - Global config management.

**ExecutorStrategy Design:**

```python

# src/hypernodes/executor.py

from abc import ABC
from concurrent.futures import Executor as ConcurrentExecutor

# --- Base Classes ---

class ExecutorStrategy(ABC):
    """
    Base class for defining HOW a pipeline executes.
    This is the Strategy Pattern. The Pipeline is the Context,
    and this is the Strategy.
    """
    pass

class TaskExecutor(ExecutorStrategy):
    """
    An execution strategy that runs discrete tasks (nodes and map items).
    This is the base for sequential, eager, thread-pooled, etc.
    """
    def __init__(
        self,
        map_executor: ConcurrentExecutor | None = None
    ):
        """
        Args:
            map_executor: A concurrent.futures.Executor (like ThreadPoolExecutor)
                          to parallelize items in 'pipeline.map()'.
                          If None, map items run sequentially.
        """
        self.map_executor = map_executor

class HolisticExecutor(ExecutorStrategy):
    """
    An execution strategy that translates the entire pipeline
    into another framework (e.g., Daft, Spark).
    """
    pass

# --- Concrete Task Strategies ---

class SequentialNodeExecutor(TaskExecutor):
    """
    The default strategy.
    - Graph Nodes: Run one-by-one.
    - Map Items: Run one-by-one (or in 'map_executor' if provided).
    """
    def __init__(self, map_executor: ConcurrentExecutor | None = None):
        super().__init__(map_executor)

class EagerNodeExecutor(TaskExecutor):
    """
    An eager/async strategy.
    - Graph Nodes: Run asynchronously as soon as inputs are ready.
    - Map Items: Run one-by-one (or in 'map_executor' if provided).
    """
    def __init__(self, map_executor: ConcurrentExecutor | None = None):
        super().__init__(map_executor)

# --- Concrete Holistic Strategies ---

class DaftExecutor(HolisticExecutor):
    """
    Executes the entire pipeline as a Daft plan.
    It does NOT use a 'map_executor' because Daft handles
    all parallelism internally.
    """
    def __init__(self, runner: str = "local", ray_address: str | None = None):
        self.runner = runner
        self.ray_address = ray_address

```

**Global Config Design:**

```python

# src/hypernodes/config.py

from typing import Optional, List
from .executor import ExecutorStrategy

# Global config storage
_global_backend: Optional['Backend'] = None
_global_executor: Optional[ExecutorStrategy] = None
_global_cache: Optional['Cache'] = None
_global_callbacks: List['PipelineCallback'] = []
# ... other globals

def set_default_config(
    backend: Optional['Backend'] = None,
    executor: Optional[ExecutorStrategy] = None,
    cache: Optional['Cache'] = None,
    callbacks: Optional[List['PipelineCallback']] = None,
    # ... other params
) -> None:
    """Set global default configuration."""
    global _global_backend, _global_executor, _global_cache, _global_callbacks

    if backend is not None:
        _global_backend = backend
    if executor is not None:
        _global_executor = executor
    if cache is not None:
        _global_cache = cache
    if callbacks is not None:
        _global_callbacks = callbacks

def get_default_executor() -> Optional[ExecutorStrategy]:
    """Get global default execution strategy."""
    return _global_executor

# ... other get_default...() and reset_default_config() ...

```

### Phase 2: Lazy Initialization (Week 2)

**Make Cache Lazy:**

```python

# src/hypernodes/cache.py (UPDATE)
# ... (Design as specified in original plan, no changes needed) ...
class Cache(ABC):
    def __init__(self):
        self._materialized = False
    # ...

```

**Make Callbacks Lazy:**

```python

# src/hypernodes/callbacks.py (UPDATE)
# ... (Design as specified in original plan, no changes needed) ...
class PipelineCallback(ABC):
    def __init__(self):
        self._materialized = False
    # ...

```

### Phase 3: Backend Refactor (Week 2)

**Refactor LocalBackend:**

```python

# src/hypernodes/backend.py (REFACTORED)
from .executor import ExecutorStrategy, TaskExecutor, SequentialNodeExecutor, EagerNodeExecutor, DaftExecutor

class Backend(ABC):
    # ...
    @abstractmethod
    def run(self, pipeline, inputs, executor: ExecutorStrategy, ctx, output_name):
        pass

    @abstractmethod
    def map(self, pipeline, items, inputs, executor: ExecutorStrategy, ctx, output_name):
        pass

class LocalBackend(Backend):
    """Local execution. Interprets TaskExecutor objects."""

    def run(self, pipeline, inputs, executor: ExecutorStrategy, ctx, output_name):
        """Execute pipeline locally."""
        if not isinstance(executor, TaskExecutor):
            raise TypeError(
                f"LocalBackend can only handle TaskExecutor (SequentialNodeExecutor, EagerNodeExecutor), "
                f"not {type(executor).__name__}. Use DaftBackend for Daft."
            )

        # ... (Materialize cache/callbacks) ...

        if isinstance(executor, SequentialNodeExecutor):
            # ... run graph generation-by-generation ...
            for node in nodes_to_execute:
                result = node(**node_inputs) # Simplified
                # ...
        elif isinstance(executor, EagerNodeExecutor):
            # ... use asyncio.run(self._run_eager(...)) ...
            pass

    def map(self, pipeline, items, inputs, executor: ExecutorStrategy, ctx, output_name):
        """Execute map using the strategy's map_executor."""
        if not isinstance(executor, TaskExecutor):
            raise TypeError(f"LocalBackend cannot .map() with {type(executor).__name__}")

        def run_item(item):
            return pipeline.run({**inputs, **item}, _ctx=ctx)

        map_runner = executor.map_executor
        if map_runner is None:
            # Run sequentially
            return [run_item(item) for item in items]
        else:
            # Run in parallel using the provided worker pool
            futures = [map_runner.submit(run_item, item) for item in items]
            return [f.result() for f in futures]

```

**Refactor ModalBackend:**

```python

# src/hypernodes/backend.py (UPDATED)

class ModalBackend(Backend):
    # ... (init) ...

    def run(self, pipeline, inputs, executor: ExecutorStrategy, ctx, output_name):
        # ...
        # Serialize pipeline + execution strategy
        payload = self._serialize(pipeline, inputs, executor, ctx, output_name)
        # ... (submit to modal) ...

    def map(self, pipeline, items, inputs, executor: ExecutorStrategy, ctx, output_name):
        # ...
        # The map logic is handled remotely by the backend's .run_item()
        # The map_executor from the ExecutorStrategy will be used on the remote side
        # ...
        pass

    def _serialize(self, pipeline, inputs, executor, ctx, output_name):
        # ...
        payload = (pipeline, inputs, executor, ctx.data, output_name)
        return cloudpickle.dumps(payload)

    @self._app.function(...)
    def _remote_execute(serialized_payload: bytes) -> bytes:
        pipeline, inputs, executor, ctx_data, output_name = cloudpickle.loads(serialized_payload)

        # Use LocalBackend for actual execution on the remote worker
        local_backend = LocalBackend()
        ctx = CallbackContext()
        ctx.data.update(ctx_data)

        # The remote worker uses the deserialized execution strategy
        results = local_backend.run(pipeline, inputs, executor, ctx, output_name)
        return cloudpickle.dumps(results)

```

### Phase 4: Pipeline Integration (Week 3)

**Update Pipeline:**

```python

# src/hypernodes/pipeline.py (UPDATED)
from .executor import ExecutorStrategy, SequentialNodeExecutor

class Pipeline:
    def __init__(
        self,
        nodes,
        backend: Optional[Backend] = None,
        executor: Optional[ExecutorStrategy] = None,
        cache: Optional[Cache] = None,
        callbacks: Optional[List[PipelineCallback]] = None,
        name: Optional[str] = None,
        parent: Optional['Pipeline'] = None,
    ):
        self.nodes = nodes
        self.name = name
        self._parent = parent

        self.backend = backend
        self.executor = executor
        self.cache = cache
        self.callbacks = callbacks or []
        # ...

    @property
    def effective_executor(self) -> ExecutorStrategy:
        """Get effective execution strategy with precedence."""
        from .config import get_default_executor

        # Own config
        if self.executor is not None:
            return self.executor

        # Parent config
        if self._parent is not None:
            return self._parent.effective_executor

        # Global default
        global_executor = get_default_executor()
        if global_executor is not None:
            return global_executor

        # Library default
        return SequentialNodeExecutor()

    # ... (effective_backend, effective_cache, effective_callbacks) ...

    def run(self, inputs, output_name=None, _ctx=None):
        """Execute pipeline."""
        backend = self.effective_backend
        executor = self.effective_executor

        return backend.run(self, inputs, executor, _ctx, output_name)

    def map(self, items, inputs=None, output_name=None, _ctx=None):
        """Execute pipeline over a list of items."""
        backend = self.effective_backend
        executor = self.effective_executor

        return backend.map(self, items, inputs or {}, executor, _ctx, output_name)

    # --- Fluent builders ---
    def with_backend(self, backend: Backend) -> 'Pipeline':
        self.backend = backend
        return self

    def with_executor(self, executor: ExecutorStrategy) -> 'Pipeline':
        self.executor = executor
        return self

    def with_cache(self, cache: Cache) -> 'Pipeline':
        self.cache = cache
        return self
    # ...

```

### Phase 5: DaftExecutor (Week 4 - Separate)

**DaftBackend (optional):**

```python

# src/hypernodes/backend.py (ADD)
from .executor import DaftExecutor, ExecutorStrategy

class DaftBackend(Backend):
    """Backend for Daft execution."""

    def run(self, pipeline, inputs, executor: ExecutorStrategy, ctx, output_name):
        """Execute pipeline using Daft."""
        if not isinstance(executor, DaftExecutor):
            raise TypeError(
                f"DaftBackend requires a DaftExecutor strategy, "
                f"got {type(executor).__name__}."
            )

        # 'executor' is our Daft object with config
        print(f"Translating pipeline to Daft plan (runner={executor.runner})...")

        # ... daft_plan = self._convert_to_daft(pipeline, executor) ...
        # ... result = daft_plan.run(inputs) ...
        return result

    def map(self, pipeline, items, inputs, executor: ExecutorStrategy, ctx, output_name):
        """Execute map using Daft."""
        # Daft handles map naturally via DataFrame operations
        return self.run(pipeline, {**inputs, "items": items}, executor, ctx, output_name)

```

### Phase 6: Testing & Documentation (Week 5)

**Test Structure:**

```
tests/
├── test_executor.py
│   ├── test_sequential_executor
│   ├── test_eager_executor
│   ├── test_task_executor_with_map_executor
│   └── test_daft_executor
├── test_runtime_config.py
│   ├── test_config_precedence
│   └── test_global_config
├── test_lazy_initialization.py
│   ├── ...
├── test_backend_refactor.py
│   ├── test_local_backend_interprets_strategy
│   ├── test_modal_backend_serializes_strategy
│   └── test_daft_backend_requires_daft_strategy
└── test_integration.py
    ├── ...

```

## 5. Migration Guide

### Old API → New API

**Before:**

```python

# Old way mixed node/map execution
backend = LocalBackend(node_execution="threaded", max_workers=8)
pipeline = Pipeline(nodes=[...], backend=backend, cache=DiskCache(".cache"))

```

**After (Option 1: Constructor):**

```python

from hypernodes.executor import SequentialNodeExecutor
from concurrent.futures import ThreadPoolExecutor

# New way is explicit:
# - Node execution is 'Sequential'
# - Map execution is 'ThreadPoolExecutor(8)'
pipeline = Pipeline(
    nodes=[...],
    executor=SequentialNodeExecutor(map_executor=ThreadPoolExecutor(max_workers=8)),
    cache=DiskCache(".cache")
)

```

**After (Option 2: Fluent):**

```python

from hypernodes.executor import SequentialNodeExecutor
from concurrent.futures import ThreadPoolExecutor

pipeline = (
    Pipeline(nodes=[...])
    .with_executor(SequentialNodeExecutor(map_executor=ThreadPoolExecutor(max_workers=8)))
    .with_cache(DiskCache(".cache"))
)

```

**After (Option 3: Global):**

```python

from hypernodes import set_default_config
from hypernodes.executor import SequentialNodeExecutor
from concurrent.futures import ThreadPoolExecutor

set_default_config(
    executor=SequentialNodeExecutor(map_executor=ThreadPoolExecutor(max_workers=8)),
    cache=DiskCache(".cache")
)

# All new pipelines use this
pipeline = Pipeline(nodes=[...])

```

## 6. Timeline

| **Week** | **Phase** | **Deliverables** |
| --- | --- | --- |
| 1 | Core Abstractions | `ExecutorStrategy` classes, Global config functions |
| 2 | Lazy Init + Backend Refactor | Cache/Callback lazy, `Backend` update to interpret strategies |
| 3 | Pipeline Integration | `Pipeline` class updated with `executor` param |
| 4 | DaftBackend + as_node() | `DaftBackend` + `DaftExecutor` strategy, `as_node()` overrides |
| 5 | Testing & Docs | Tests, migration guide |

**Total:** 5 weeks (1 engineer)

## 7. Key Design Decisions

### 7.1 Why No RuntimeConfig Class?

**Decision:** No `RuntimeConfig` wrapper - just individual parameters (`backend`, `executor`, `cache`, `callbacks`).

**Rationale:**

- Simpler for users - no extra concept to learn.
- Each concern is explicit and separate.
- `pipeline.with_executor(...)` is clearer than `pipeline.with_config(RuntimeConfig(executor=...))`
- Precedence still works with individual params.

### 7.2 Why Keep Familiar API (DiskCache, not CacheSpec)?

**Decision:** Keep `DiskCache`, `ProgressCallback`, etc. - make them lazy internally.

**Rationale:**

- Users already know these classes.
- Easier migration (just add lazy init internally).
- Implementation detail (lazy init) hidden from users.

### 7.3 Why No Call-Site Overrides in run()?

**Decision:** Keep `pipeline.run(inputs)` clean - no `config=` parameter.

**Rationale:**

- Simpler API surface.
- Configuration is done before execution (constructor or `.with_*()`).
- For nested pipelines, use `as_node(executor=..., cache=...)` to override.
- Clearer separation: configure → execute.

### 7.4 Why Separate Daft?

**Decision:** `DaftExecutor` is a `HolisticExecutor`, completely separate from `TaskExecutor`.

**Rationale:**

- Fundamentally different. `TaskExecutor` (like `EagerNodeExecutor`) *runs* nodes. `HolisticExecutor` (like `DaftExecutor`) *translates* nodes.
- Makes conflicts impossible. You can't have `EagerNodeExecutor(map_executor=...)` and `DaftExecutor()` at the same time. You choose one.

### 7.5 Why No RemoteBackend Yet?

**Decision:** Skip generic RemoteBackend for now.

**Rationale:**

- Dask, Ray, Spark each need custom integration.
- Can add specific backends (`DaskBackend`) and strategies (`DaskExecutor`) later.

### 7.6 Why `ExecutorStrategy` Objects?

**Decision:** Use the Strategy Pattern (`executor=SequentialNodeExecutor(...)`) instead of simple strings (`node_scheduler="eager"`, `map_executor=...`).

**Rationale:**

- **Adheres to Single Responsibility Principle:** The `Pipeline` is no longer responsible for *combining* and *validating* conflicting execution parameters. That logic is now encapsulated *within* the `ExecutorStrategy` object itself.
- **No Conflicts:** It's impossible for a user to provide `executor=DaftExecutor()` *and* a `map_executor`. The `DaftExecutor` object doesn't have a `map_executor` parameter. All "HOW" logic is bundled in one place.
- **Extensible:** Adding a new `SparkExecutor` is clean. You just add a new class. You don't need to update the `Pipeline`'s `__init__` with more `if/elif` logic.
- **Clarity:** It's immediately obvious what the execution model is just by looking at the object: `EagerNodeExecutor(map_executor=ThreadPoolExecutor())`.

### 7.7 Why `set_default_config()` Not Separate Functions?

**Decision:** Single `set_default_config(executor=..., cache=..., callbacks=...)` function.

**Rationale:**

- One place to set global defaults.
- Clear that all params are optional.
- Easier to set multiple defaults at once.

**End of Implementation Plan**