Phase 1: Core Executor Abstractions

## Class Structure (Pseudo-code)

```python
# src/hypernodes/executor.py

from abc import ABC
from typing import Optional, Literal
from concurrent.futures import Executor as ConcurrentExecutor

# --- Base Classes (Strategy Pattern) ---

class ExecutorStrategy(ABC):
    """Base strategy for HOW pipeline executes.

    Single Responsibility: Define execution strategy interface
    Open/Closed: Open for extension (new strategies), closed for modification
    """
    pass


class TaskExecutor(ExecutorStrategy):
    """Executes discrete tasks (nodes + map items).

    Single Responsibility: Bundle node + map execution config
    """
    def __init__(self, map_executor: Optional[ConcurrentExecutor] = None):
        self.map_executor = map_executor


class FrameworkExecutor(ExecutorStrategy):
    """Translates entire pipeline to framework (Daft, Spark, etc).

    Single Responsibility: Framework-specific execution
    Liskov Substitution: Can replace ExecutorStrategy without breaking code
    """
    pass


# --- Concrete Strategies ---

class LocalExecutor(TaskExecutor):
    """Configurable local task execution.

    Single Responsibility: Local execution with configurable strategy
    Args:
        node_execution: "sequential" or "async"
        map_executor: Optional concurrent.futures.Executor for parallel map
    """
    def __init__(
        self,
        node_execution: Literal["sequential", "async"] = "sequential",
        map_executor: Optional[ConcurrentExecutor] = None
    ):
        super().__init__(map_executor)
        self.node_execution = node_execution


class DaftExecutor(FrameworkExecutor):
    """Entire pipeline handled by Daft.

    Single Responsibility: Daft-specific execution config
    Interface Segregation: No map_executor (Daft handles internally)
    """
    def __init__(self, runner: str = "local", ray_address: Optional[str] = None):
        self.runner = runner
        self.ray_address = ray_address
```

```python
# src/hypernodes/config.py

from typing import Optional, List
from .executor import ExecutorStrategy

# Module-level state (Dependency Inversion: depend on abstractions)
_global_backend: Optional['Backend'] = None
_global_executor: Optional[ExecutorStrategy] = None
_global_cache: Optional['Cache'] = None
_global_callbacks: List['PipelineCallback'] = []


def set_default_config(
    backend: Optional['Backend'] = None,
    executor: Optional[ExecutorStrategy] = None,
    cache: Optional['Cache'] = None,
    callbacks: Optional[List['PipelineCallback']] = None,
) -> None:
    """Single function to set all defaults.
    
    Single Responsibility: Global config management
    """
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
    """Get global executor."""
    return _global_executor


def get_default_backend() -> Optional['Backend']:
    """Get global backend."""
    return _global_backend


def get_default_cache() -> Optional['Cache']:
    """Get global cache."""
    return _global_cache


def get_default_callbacks() -> List['PipelineCallback']:
    """Get global callbacks."""
    return _global_callbacks.copy()


def reset_default_config() -> None:
    """Reset all defaults."""
    global _global_backend, _global_executor, _global_cache, _global_callbacks
    _global_backend = None
    _global_executor = None
    _global_cache = None
    _global_callbacks = []
```

## Smoke Tests

```python
# tests/test_executor_strategy.py

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from hypernodes.executor import (
    ExecutorStrategy,
    TaskExecutor,
    FrameworkExecutor,
    LocalExecutor,
    DaftExecutor,
)


def test_local_executor_default():
    """Smoke: LocalExecutor with defaults (sequential, no map_executor)."""
    executor = LocalExecutor()

    assert isinstance(executor, ExecutorStrategy)
    assert isinstance(executor, TaskExecutor)
    assert executor.node_execution == "sequential"
    assert executor.map_executor is None


def test_local_executor_async():
    """Smoke: LocalExecutor with async node execution."""
    executor = LocalExecutor(node_execution="async")

    assert executor.node_execution == "async"
    assert executor.map_executor is None


def test_local_executor_with_thread_pool():
    """Smoke: LocalExecutor with ThreadPoolExecutor for map."""
    pool = ThreadPoolExecutor(max_workers=4)
    executor = LocalExecutor(map_executor=pool)

    assert executor.node_execution == "sequential"
    assert executor.map_executor is pool


def test_local_executor_async_with_process_pool():
    """Smoke: LocalExecutor with async nodes + ProcessPoolExecutor for map."""
    pool = ProcessPoolExecutor(max_workers=2)
    executor = LocalExecutor(node_execution="async", map_executor=pool)

    assert executor.node_execution == "async"
    assert isinstance(executor, TaskExecutor)
    assert executor.map_executor is pool


def test_daft_executor_local():
    """Smoke: DaftExecutor with local runner."""
    executor = DaftExecutor(runner="local")

    assert isinstance(executor, ExecutorStrategy)
    assert isinstance(executor, FrameworkExecutor)
    assert not isinstance(executor, TaskExecutor)  # Different hierarchy
    assert executor.runner == "local"
    assert executor.ray_address is None


def test_daft_executor_ray():
    """Smoke: DaftExecutor with ray runner."""
    executor = DaftExecutor(runner="ray", ray_address="ray://localhost:10001")

    assert executor.runner == "ray"
    assert executor.ray_address == "ray://localhost:10001"


def test_executor_hierarchy():
    """Smoke: Verify class hierarchy."""
    local = LocalExecutor()
    daft = DaftExecutor()

    # All are ExecutorStrategy
    assert isinstance(local, ExecutorStrategy)
    assert isinstance(daft, ExecutorStrategy)

    # TaskExecutor vs FrameworkExecutor separation
    assert isinstance(local, TaskExecutor)
    assert not isinstance(daft, TaskExecutor)

    assert isinstance(daft, FrameworkExecutor)
    assert not isinstance(local, FrameworkExecutor)
```

```python
# tests/test_global_config.py

from hypernodes.config import (
    set_default_config,
    get_default_executor,
    get_default_backend,
    get_default_cache,
    get_default_callbacks,
    reset_default_config,
)
from hypernodes.executor import LocalExecutor
from hypernodes.backend import LocalBackend
from hypernodes.cache import DiskCache
from hypernodes.callbacks import PipelineCallback


def test_set_and_get_executor():
    """Smoke: Set/get global executor."""
    reset_default_config()

    executor = LocalExecutor()
    set_default_config(executor=executor)

    assert get_default_executor() is executor


def test_set_and_get_backend():
    """Smoke: Set/get global backend."""
    reset_default_config()

    backend = LocalBackend()
    set_default_config(backend=backend)

    assert get_default_backend() is backend


def test_set_and_get_cache():
    """Smoke: Set/get global cache."""
    reset_default_config()

    cache = DiskCache(".cache")
    set_default_config(cache=cache)

    assert get_default_cache() is cache


def test_set_multiple_configs_at_once():
    """Smoke: Set all configs in one call."""
    reset_default_config()

    executor = LocalExecutor()
    backend = LocalBackend()
    cache = DiskCache(".cache")
    callback = PipelineCallback()

    set_default_config(
        executor=executor,
        backend=backend,
        cache=cache,
        callbacks=[callback]
    )

    assert get_default_executor() is executor
    assert get_default_backend() is backend
    assert get_default_cache() is cache
    assert callback in get_default_callbacks()


def test_reset_clears_all():
    """Smoke: Reset clears all defaults."""
    set_default_config(
        executor=LocalExecutor(),
        backend=LocalBackend(),
        cache=DiskCache(".cache"),
    )

    reset_default_config()

    assert get_default_executor() is None
    assert get_default_backend() is None
    assert get_default_cache() is None
    assert get_default_callbacks() == []
```

Phase 2: Lazy Initialization

## Class Structure (Pseudo-code)

```python
# src/hypernodes/cache.py (UPDATED)

from abc import ABC, abstractmethod
from typing import Any, Optional

class Cache(ABC):
    """Abstract cache with lazy initialization.
    
    Single Responsibility: Cache interface
    Open/Closed: Open for extension (new cache backends)
    """
    def __init__(self):
        self._materialized = False
    
    @abstractmethod
    def _materialize(self) -> None:
        """Allocate resources (files, connections, etc).
        
        Single Responsibility: Resource allocation
        """
        pass
    
    def _ensure_materialized(self) -> None:
        """Ensure cache is ready to use."""
        if not self._materialized:
            self._materialize()
            self._materialized = True
    
    @abstractmethod
    def get(self, signature: str) -> Optional[Any]:
        """Get cached value."""
        pass
    
    @abstractmethod
    def put(self, signature: str, output: Any) -> None:
        """Store cached value."""
        pass


class DiskCache(Cache):
    """Disk-based cache with lazy file opening.
    
    Single Responsibility: Disk-based caching
    """
    def __init__(self, path: str):
        super().__init__()
        # Store config only - no file operations yet
        self.path = Path(path)
        self.blob_dir = None
        self.meta_file = None
        self.meta_store = None
    
    def _materialize(self) -> None:
        """Open files and load metadata.
        
        Single Responsibility: File initialization
        """
        self.path.mkdir(parents=True, exist_ok=True)
        self.blob_dir = self.path / "blobs"
        self.blob_dir.mkdir(exist_ok=True)
        self.meta_file = self.path / "meta.json"
        self.meta_store = self._load_meta()
    
    def get(self, signature: str) -> Optional[Any]:
        """Get from cache (materializes if needed)."""
        self._ensure_materialized()
        # ... existing get logic ...
    
    def put(self, signature: str, output: Any) -> None:
        """Put to cache (materializes if needed)."""
        self._ensure_materialized()
        # ... existing put logic ...
```

```python
# src/hypernodes/callbacks.py (UPDATED)

from abc import ABC

class PipelineCallback(ABC):
    """Base callback with lazy initialization.
    
    Single Responsibility: Callback lifecycle interface
    Open/Closed: Open for extension (new callbacks)
    """
    def __init__(self):
        self._materialized = False
    
    def _materialize(self) -> None:
        """Allocate resources (progress bars, tracers, etc).
        
        Single Responsibility: Resource allocation
        Override in subclasses if needed.
        """
        pass
    
    def _ensure_materialized(self) -> None:
        """Ensure callback is ready to use."""
        if not self._materialized:
            self._materialize()
            self._materialized = True
    
    def on_pipeline_start(self, pipeline_id: str, inputs: Dict, ctx: CallbackContext) -> None:
        """Called on pipeline start (materializes if needed)."""
        self._ensure_materialized()
        # Subclass implements logic here
    
    # ... other lifecycle methods call _ensure_materialized() first ...
```

## Smoke Tests

```python
# tests/test_lazy_cache.py

import tempfile
from pathlib import Path
from hypernodes.cache import DiskCache


def test_diskcache_lazy_initialization():
    """Smoke: DiskCache doesn't create files in __init__."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache"
        
        # Create cache - should NOT create directory yet
        cache = DiskCache(str(cache_path))
        
        assert not cache_path.exists()  # Lazy!
        assert cache._materialized is False
        
        # First operation triggers materialization
        cache.put("test_sig", {"result": 42})
        
        assert cache_path.exists()  # Now created
        assert cache._materialized is True


def test_diskcache_works_after_materialization():
    """Smoke: DiskCache works normally after lazy init."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskCache(str(Path(tmpdir) / "cache"))
        
        # Put triggers materialization
        cache.put("sig1", {"value": 100})
        
        # Get should work
        result = cache.get("sig1")
        assert result == {"value": 100}
        
        # Cache miss
        assert cache.get("sig_nonexistent") is None


def test_diskcache_multiple_operations():
    """Smoke: Multiple operations don't re-materialize."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskCache(str(Path(tmpdir) / "cache"))
        
        cache.put("sig1", 1)
        materialize_count = 1
        
        cache.put("sig2", 2)
        cache.get("sig1")
        cache.get("sig2")
        
        # _materialized flag prevents re-initialization
        assert cache._materialized is True
```

```python
# tests/test_lazy_callbacks.py

from hypernodes.callbacks import PipelineCallback, CallbackContext


class TestCallback(PipelineCallback):
    """Test callback that tracks materialization."""
    
    def __init__(self):
        super().__init__()
        self.materialize_count = 0
        self.start_count = 0
    
    def _materialize(self):
        """Track materialization."""
        self.materialize_count += 1
    
    def on_pipeline_start(self, pipeline_id, inputs, ctx):
        """Track calls."""
        super().on_pipeline_start(pipeline_id, inputs, ctx)
        self.start_count += 1


def test_callback_lazy_initialization():
    """Smoke: Callback doesn't materialize in __init__."""
    callback = TestCallback()
    
    assert callback._materialized is False
    assert callback.materialize_count == 0


def test_callback_materializes_on_first_use():
    """Smoke: Callback materializes on first lifecycle event."""
    callback = TestCallback()
    ctx = CallbackContext()
    
    # First call triggers materialization
    callback.on_pipeline_start("test_pipeline", {}, ctx)
    
    assert callback._materialized is True
    assert callback.materialize_count == 1
    assert callback.start_count == 1


def test_callback_doesnt_rematerialize():
    """Smoke: Subsequent calls don't re-materialize."""
    callback = TestCallback()
    ctx = CallbackContext()
    
    callback.on_pipeline_start("pipeline1", {}, ctx)
    callback.on_pipeline_start("pipeline2", {}, ctx)
    callback.on_pipeline_start("pipeline3", {}, ctx)
    
    # Only materialized once
    assert callback.materialize_count == 1
    # But called three times
    assert callback.start_count == 3
```

Phase 3: Backend Refactor

## Class Structure (Pseudo-code)

```python
# src/hypernodes/backend.py (REFACTORED)

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from .executor import ExecutorStrategy, TaskExecutor, LocalExecutor

class Backend(ABC):
    """Abstract backend - WHERE code runs.

    Single Responsibility: Define execution environment interface
    Dependency Inversion: Depends on ExecutorStrategy abstraction
    """
    @abstractmethod
    def run(
        self,
        pipeline: 'Pipeline',
        inputs: Dict[str, Any],
        executor: ExecutorStrategy,  # NEW: Strategy injected
        ctx: Optional['CallbackContext'] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute pipeline.

        Single Responsibility: Execute pipeline in this environment
        """
        pass

    @abstractmethod
    def map(
        self,
        pipeline: 'Pipeline',
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        executor: ExecutorStrategy,  # NEW: Strategy injected
        ctx: Optional['CallbackContext'] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute map operation."""
        pass


class LocalBackend(Backend):
    """Local execution - interprets ExecutorStrategy.

    Single Responsibility: Local execution environment
    Open/Closed: Closed for modification (no __init__ params),
                 open for extension (new executor strategies)
    Dependency Inversion: Depends on ExecutorStrategy abstraction
    """
    def __init__(self):
        """No configuration - strategy injected at runtime."""
        pass

    def run(
        self,
        pipeline: 'Pipeline',
        inputs: Dict[str, Any],
        executor: ExecutorStrategy,
        ctx: Optional['CallbackContext'] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute locally using given strategy.

        Single Responsibility: Dispatch to strategy-specific execution
        """
        # Validate executor type
        if not isinstance(executor, TaskExecutor):
            raise TypeError(
                f"LocalBackend can only handle TaskExecutor, "
                f"got {type(executor).__name__}. "
                f"Use DaftBackend for DaftExecutor."
            )

        # Materialize cache and callbacks (lazy init)
        self._materialize_resources(pipeline)

        # Dispatch based on node_execution mode (Strategy Pattern)
        if not isinstance(executor, LocalExecutor):
            raise TypeError(
                f"LocalBackend requires LocalExecutor, "
                f"got {type(executor).__name__}"
            )

        if executor.node_execution == "sequential":
            return self._run_sequential(pipeline, inputs, ctx, output_name)
        elif executor.node_execution == "async":
            return self._run_async(pipeline, inputs, ctx, output_name)
        else:
            raise ValueError(f"Unsupported node_execution mode: {executor.node_execution}")
    
    def map(
        self,
        pipeline: 'Pipeline',
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        executor: ExecutorStrategy,
        ctx: Optional['CallbackContext'] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute map using executor's map_executor.
        
        Single Responsibility: Map execution dispatch
        """
        if not isinstance(executor, TaskExecutor):
            raise TypeError(f"LocalBackend.map() requires TaskExecutor")
        
        def run_item(item):
            return self.run(pipeline, {**inputs, **item}, executor, ctx, output_name)
        
        # Use executor's map_executor (concurrent.futures.Executor)
        if executor.map_executor is None:
            # Sequential
            return [run_item(item) for item in items]
        else:
            # Parallel using provided executor
            futures = [executor.map_executor.submit(run_item, item) for item in items]
            return [f.result() for f in futures]
    
    def _materialize_resources(self, pipeline: 'Pipeline') -> None:
        """Trigger lazy initialization of cache and callbacks.
        
        Single Responsibility: Resource materialization
        """
        cache = pipeline.effective_cache
        if cache is not None and hasattr(cache, '_ensure_materialized'):
            cache._ensure_materialized()
        
        callbacks = pipeline.effective_callbacks
        for callback in callbacks:
            if hasattr(callback, '_ensure_materialized'):
                callback._ensure_materialized()
    
    # Keep existing _run_sequential, _run_async, etc. unchanged
    def _run_sequential(self, pipeline, inputs, ctx, output_name):
        """Existing implementation - no changes."""
        pass
    
    def _run_async(self, pipeline, inputs, ctx, output_name):
        """Existing implementation - no changes."""
        pass
```

```python
# src/hypernodes/backend_compat.py (TEMPORARY SHIM)

from typing import Literal, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from .backend import LocalBackend
from .executor import LocalExecutor

class LocalBackendV1(LocalBackend):
    """Temporary compatibility shim for old API.

    Single Responsibility: Backward compatibility
    WILL BE REMOVED in Phase 7
    """
    def __init__(
        self,
        node_execution: Literal["sequential", "async", "threaded", "parallel"] = "sequential",
        map_execution: Literal["sequential", "async", "threaded", "parallel"] = "sequential",
        max_workers: Optional[int] = None,
    ):
        super().__init__()

        # Map old API to ExecutorStrategy
        map_exec = None
        if map_execution == "threaded":
            map_exec = ThreadPoolExecutor(max_workers=max_workers or 4)
        elif map_execution == "parallel":
            map_exec = ProcessPoolExecutor(max_workers=max_workers or 4)

        # Map node_execution to LocalExecutor node_execution mode
        if node_execution in ("sequential", "threaded", "parallel"):
            self._executor = LocalExecutor(node_execution="sequential", map_executor=map_exec)
        elif node_execution == "async":
            self._executor = LocalExecutor(node_execution="async", map_executor=map_exec)
        else:
            raise ValueError(f"Unknown node_execution: {node_execution}")

    def run(self, pipeline, inputs, ctx=None, output_name=None):
        """Old signature - inject executor."""
        return super().run(pipeline, inputs, self._executor, ctx, output_name)

    def map(self, pipeline, items, inputs, ctx=None, output_name=None):
        """Old signature - inject executor."""
        return super().map(pipeline, items, inputs, self._executor, ctx, output_name)
```

## Smoke Tests

```python
# tests/test_local_backend_executor.py

from concurrent.futures import ThreadPoolExecutor
from hypernodes import Pipeline, node
from hypernodes.backend import LocalBackend
from hypernodes.executor import LocalExecutor


@node(output_name="result")
def add_one(x: int) -> int:
    return x + 1


def test_local_backend_sequential_executor():
    """Smoke: LocalBackend with LocalExecutor (sequential)."""
    pipeline = Pipeline(nodes=[add_one])
    backend = LocalBackend()
    executor = LocalExecutor(node_execution="sequential")

    result = backend.run(pipeline, {"x": 5}, executor)

    assert result == {"result": 6}


def test_local_backend_async_executor():
    """Smoke: LocalBackend with LocalExecutor (async)."""
    pipeline = Pipeline(nodes=[add_one])
    backend = LocalBackend()
    executor = LocalExecutor(node_execution="async")

    result = backend.run(pipeline, {"x": 10}, executor)

    assert result == {"result": 11}


def test_local_backend_map_with_thread_pool():
    """Smoke: LocalBackend.map() uses executor's map_executor."""
    pipeline = Pipeline(nodes=[add_one])
    backend = LocalBackend()
    pool = ThreadPoolExecutor(max_workers=2)
    executor = LocalExecutor(map_executor=pool)

    items = [{"x": 1}, {"x": 2}, {"x": 3}]
    results = backend.map(pipeline, items, {}, executor)

    assert results == [{"result": 2}, {"result": 3}, {"result": 4}]
    pool.shutdown()


def test_local_backend_rejects_daft_executor():
    """Smoke: LocalBackend rejects DaftExecutor."""
    from hypernodes.executor import DaftExecutor

    pipeline = Pipeline(nodes=[add_one])
    backend = LocalBackend()
    executor = DaftExecutor(runner="local")

    with pytest.raises(TypeError, match="can only handle TaskExecutor"):
        backend.run(pipeline, {"x": 5}, executor)
```

```python
# tests/test_backend_compatibility.py

from hypernodes import Pipeline, node
from hypernodes.backend_compat import LocalBackendV1


@node(output_name="result")
def add_one(x: int) -> int:
    return x + 1


def test_local_backend_v1_sequential():
    """Smoke: LocalBackendV1 with old API."""
    pipeline = Pipeline(nodes=[add_one])
    backend = LocalBackendV1(node_execution="sequential")
    
    result = backend.run(pipeline, {"x": 5})
    
    assert result == {"result": 6}


def test_local_backend_v1_async():
    """Smoke: LocalBackendV1 with async nodes."""
    pipeline = Pipeline(nodes=[add_one])
    backend = LocalBackendV1(node_execution="async")
    
    result = backend.run(pipeline, {"x": 10})
    
    assert result == {"result": 11}


def test_local_backend_v1_threaded_map():
    """Smoke: LocalBackendV1 with threaded map."""
    pipeline = Pipeline(nodes=[add_one])
    backend = LocalBackendV1(
        node_execution="sequential",
        map_execution="threaded",
        max_workers=2
    )
    
    items = [{"x": 1}, {"x": 2}, {"x": 3}]
    results = backend.map(pipeline, items, {})
    
    assert results == [{"result": 2}, {"result": 3}, {"result": 4}]
```

Phase 4: Pipeline Integration

## Class Structure (Pseudo-code)

```python
# src/hypernodes/pipeline.py (UPDATED)

from typing import Optional, List, Dict, Any, Union
from .backend import Backend, LocalBackend
from .executor import ExecutorStrategy, LocalExecutor
from .cache import Cache
from .callbacks import PipelineCallback

class Pipeline:
    """Pipeline with individual config parameters.

    Single Responsibility: Pipeline orchestration
    Dependency Inversion: Depends on Backend/Executor abstractions
    """
    def __init__(
        self,
        nodes: List,
        backend: Optional[Backend] = None,
        executor: Optional[ExecutorStrategy] = None,  # NEW
        cache: Optional[Cache] = None,
        callbacks: Optional[List[PipelineCallback]] = None,
        name: Optional[str] = None,
        parent: Optional['Pipeline'] = None,
    ):
        """Initialize pipeline with individual config params.

        Single Responsibility: Pipeline configuration
        """
        self.nodes = nodes
        self.backend = backend
        self.executor = executor  # NEW
        self.cache = cache
        self.callbacks = callbacks
        self.name = name
        self._parent = parent
        # ... existing graph building logic ...

    @property
    def effective_executor(self) -> ExecutorStrategy:
        """Get effective executor with precedence.

        Precedence: self > parent > global > library default
        Single Responsibility: Executor resolution
        """
        from .config import get_default_executor

        # Level 1: Own config
        if self.executor is not None:
            return self.executor

        # Level 2: Parent config
        if self._parent is not None:
            return self._parent.effective_executor

        # Level 3: Global config
        global_executor = get_default_executor()
        if global_executor is not None:
            return global_executor

        # Level 4: Library default
        return LocalExecutor()
    
    @property
    def effective_backend(self) -> Backend:
        """Get effective backend with precedence."""
        from .config import get_default_backend
        
        if self.backend is not None:
            return self.backend
        if self._parent is not None:
            return self._parent.effective_backend
        
        global_backend = get_default_backend()
        if global_backend is not None:
            return global_backend
        
        return LocalBackend()  # Library default
    
    # ... existing effective_cache, effective_callbacks ...
    
    def run(
        self,
        inputs: Dict[str, Any],
        output_name: Union[str, List[str], None] = None,
        _ctx=None,
    ) -> Dict[str, Any]:
        """Execute pipeline.
        
        Single Responsibility: Pipeline execution entry point
        """
        backend = self.effective_backend
        executor = self.effective_executor  # NEW
        
        # Backend interprets executor (Strategy Pattern)
        return backend.run(self, inputs, executor, _ctx, output_name)
    
    def with_executor(self, executor: ExecutorStrategy) -> 'Pipeline':
        """Fluent builder for executor.
        
        Single Responsibility: Configuration builder
        """
        self.executor = executor
        return self
    
    # ... existing with_backend, with_cache, with_callbacks ...
```

## Smoke Tests

```python
# tests/test_pipeline_executor.py

from concurrent.futures import ThreadPoolExecutor
from hypernodes import Pipeline, node
from hypernodes.executor import LocalExecutor


@node(output_name="result")
def add_one(x: int) -> int:
    return x + 1


def test_pipeline_uses_executor_from_constructor():
    """Smoke: Pipeline uses executor from constructor."""
    executor = LocalExecutor()
    pipeline = Pipeline(nodes=[add_one], executor=executor)

    assert pipeline.executor is executor
    assert pipeline.effective_executor is executor


def test_pipeline_uses_default_executor():
    """Smoke: Pipeline uses default if no executor provided."""
    pipeline = Pipeline(nodes=[add_one])

    assert pipeline.executor is None
    # Should get LocalExecutor as library default
    assert isinstance(pipeline.effective_executor, LocalExecutor)


def test_pipeline_with_executor_fluent():
    """Smoke: with_executor() fluent builder."""
    pipeline = Pipeline(nodes=[add_one])
    executor = LocalExecutor(node_execution="async")

    pipeline.with_executor(executor)

    assert pipeline.executor is executor


def test_pipeline_run_passes_executor_to_backend():
    """Smoke: run() passes executor to backend."""
    executor = LocalExecutor()
    pipeline = Pipeline(nodes=[add_one], executor=executor)

    result = pipeline.run({"x": 5})

    assert result == {"result": 6}


def test_pipeline_map_with_executor():
    """Smoke: map() uses executor's map_executor."""
    pool = ThreadPoolExecutor(max_workers=2)
    executor = LocalExecutor(map_executor=pool)
    pipeline = Pipeline(nodes=[add_one], executor=executor)

    results = pipeline.map({"x": [1, 2, 3]}, map_over="x")

    assert results == {"result": [2, 3, 4]}
    pool.shutdown()
```

```python
# tests/test_pipeline_config_precedence.py

from hypernodes import Pipeline, node
from hypernodes.config import set_default_config, reset_default_config
from hypernodes.executor import LocalExecutor


@node(output_name="result")
def add_one(x: int) -> int:
    return x + 1


def test_precedence_pipeline_over_global():
    """Smoke: Pipeline config overrides global config."""
    reset_default_config()

    global_executor = LocalExecutor()
    set_default_config(executor=global_executor)

    pipeline_executor = LocalExecutor(node_execution="async")
    pipeline = Pipeline(nodes=[add_one], executor=pipeline_executor)

    assert pipeline.effective_executor is pipeline_executor  # Not global


def test_precedence_global_over_default():
    """Smoke: Global config overrides library default."""
    reset_default_config()

    global_executor = LocalExecutor(node_execution="async")
    set_default_config(executor=global_executor)

    pipeline = Pipeline(nodes=[add_one])  # No executor specified

    assert pipeline.effective_executor is global_executor


def test_precedence_nested_pipeline():
    """Smoke: Child inherits from parent."""
    parent_executor = LocalExecutor()
    parent = Pipeline(nodes=[add_one], executor=parent_executor, name="parent")

    # Create child with parent reference (simulates nesting)
    child = Pipeline(nodes=[add_one], name="child", parent=parent)

    assert child.effective_executor is parent_executor
Phase 5: as_node() Config Overrides
Class Structure (Pseudo-code)
# src/hypernodes/pipeline.py (UPDATED)

class PipelineNode(Node):
    """Wraps Pipeline with config overrides.
    
    Single Responsibility: Pipeline-to-Node adaptation with config
    """
    def __init__(
        self,
        pipeline: 'Pipeline',
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        map_over: Optional[Union[str, List[str]]] = None,
        name: Optional[str] = None,
        # NEW: Config overrides
        backend: Optional[Backend] = None,
        executor: Optional[ExecutorStrategy] = None,
        cache: Optional[Cache] = None,
        callbacks: Optional[List[PipelineCallback]] = None,
    ):
        """Wrap pipeline with optional config overrides.
        
        Single Responsibility: Config override at node level
        """
        self.pipeline = pipeline
        self.input_mapping = input_mapping or {}
        self.output_mapping = output_mapping or {}
        self.map_over = map_over
        self.name = name
        
        # Apply config overrides (highest precedence)
        if backend is not None:
            self.pipeline.backend = backend
        if executor is not None:
            self.pipeline.executor = executor
        if cache is not None:
            self.pipeline.cache = cache
        if callbacks is not None:
            self.pipeline.callbacks = callbacks
    
    # ... existing properties and __call__ unchanged ...


class Pipeline:
    # ... existing code ...
    
    def as_node(
        self,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        map_over: Optional[Union[str, List[str]]] = None,
        name: Optional[str] = None,
        # NEW: Config overrides
        backend: Optional[Backend] = None,
        executor: Optional[ExecutorStrategy] = None,
        cache: Optional[Cache] = None,
        callbacks: Optional[List[PipelineCallback]] = None,
    ) -> PipelineNode:
        """Wrap as node with optional config overrides.
        
        Single Responsibility: Node wrapping with config
        """
        return PipelineNode(
            pipeline=self,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            map_over=map_over,
            name=name,
            backend=backend,
            executor=executor,
            cache=cache,
            callbacks=callbacks,
        )
```

## Smoke Tests

```python
# tests/test_as_node_overrides.py

from hypernodes import Pipeline, node
from hypernodes.executor import LocalExecutor
from hypernodes.cache import DiskCache


@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2


def test_as_node_override_executor():
    """Smoke: as_node() can override executor."""
    inner_executor = LocalExecutor()
    inner = Pipeline(nodes=[double], executor=inner_executor)

    # Override executor when wrapping as node
    override_executor = LocalExecutor(node_execution="async")
    node_wrapper = inner.as_node(executor=override_executor)

    assert node_wrapper.pipeline.executor is override_executor


def test_as_node_override_cache():
    """Smoke: as_node() can override cache."""
    inner = Pipeline(nodes=[double], cache=None)

    # Enable caching at node level
    cache = DiskCache(".cache")
    node_wrapper = inner.as_node(cache=cache)

    assert node_wrapper.pipeline.cache is cache


def test_as_node_override_multiple_configs():
    """Smoke: as_node() can override multiple configs."""
    inner = Pipeline(nodes=[double])

    executor = LocalExecutor(node_execution="async")
    cache = DiskCache(".cache")

    node_wrapper = inner.as_node(
        executor=executor,
        cache=cache,
        name="custom_double"
    )

    assert node_wrapper.pipeline.executor is executor
    assert node_wrapper.pipeline.cache is cache
    assert node_wrapper.name == "custom_double"


def test_as_node_no_override():
    """Smoke: as_node() without overrides preserves config."""
    executor = LocalExecutor()
    cache = DiskCache(".cache")
    inner = Pipeline(nodes=[double], executor=executor, cache=cache)

    node_wrapper = inner.as_node()

    assert node_wrapper.pipeline.executor is executor
    assert node_wrapper.pipeline.cache is cache
```

```python
# tests/test_nested_pipeline_config.py

from hypernodes import Pipeline, node
from hypernodes.executor import LocalExecutor


@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

@node(output_name="result")
def add_ten(doubled: int) -> int:
    return doubled + 10


def test_nested_pipeline_config_precedence():
    """Smoke: as_node() override has highest precedence."""
    # Inner pipeline with its own executor
    inner_executor = LocalExecutor()
    inner = Pipeline(nodes=[double], executor=inner_executor)

    # Outer pipeline with different executor
    outer_executor = LocalExecutor(node_execution="async")

    # Wrap inner with as_node() override
    override_executor = LocalExecutor()
    inner_node = inner.as_node(executor=override_executor)

    outer = Pipeline(
        nodes=[inner_node, add_ten],
        executor=outer_executor
    )

    # inner_node should use override, not outer's executor
    result = outer.run({"x": 5})
    assert result == {"result": 20}  # (5 * 2) + 10


def test_nested_pipeline_inherits_when_no_override():
    """Smoke: Nested pipeline inherits from parent when no override."""
    inner = Pipeline(nodes=[double])  # No executor specified
    inner_node = inner.as_node()  # No override

    outer_executor = LocalExecutor(node_execution="async")
    outer = Pipeline(nodes=[inner_node, add_ten], executor=outer_executor)

    # Inner should inherit outer's executor via parent reference
    # (This is handled by effective_executor property)
    result = outer.run({"x": 5})
    assert result == {"result": 20}
```

Phase 5: as_node() Config Overrides

## Class Structure (Pseudo-code)

```python
# src/hypernodes/pipeline.py (UPDATED)

class PipelineNode(Node):
    """Wraps Pipeline with config overrides.
    
    Single Responsibility: Pipeline-to-Node adaptation with config
    """
    def __init__(
        self,
        pipeline: 'Pipeline',
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        map_over: Optional[Union[str, List[str]]] = None,
        name: Optional[str] = None,
        # NEW: Config overrides
        backend: Optional[Backend] = None,
        executor: Optional[ExecutorStrategy] = None,
        cache: Optional[Cache] = None,
        callbacks: Optional[List[PipelineCallback]] = None,
    ):
        """Wrap pipeline with optional config overrides.
        
        Single Responsibility: Config override at node level
        """
        self.pipeline = pipeline
        self.input_mapping = input_mapping or {}
        self.output_mapping = output_mapping or {}
        self.map_over = map_over
        self.name = name
        
        # Apply config overrides (highest precedence)
        if backend is not None:
            self.pipeline.backend = backend
        if executor is not None:
            self.pipeline.executor = executor
        if cache is not None:
            self.pipeline.cache = cache
        if callbacks is not None:
            self.pipeline.callbacks = callbacks
    
    # ... existing properties and __call__ unchanged ...


class Pipeline:
    # ... existing code ...
    
    def as_node(
        self,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        map_over: Optional[Union[str, List[str]]] = None,
        name: Optional[str] = None,
        # NEW: Config overrides
        backend: Optional[Backend] = None,
        executor: Optional[ExecutorStrategy] = None,
        cache: Optional[Cache] = None,
        callbacks: Optional[List[PipelineCallback]] = None,
    ) -> PipelineNode:
        """Wrap as node with optional config overrides.
        
        Single Responsibility: Node wrapping with config
        """
        return PipelineNode(
            pipeline=self,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            map_over=map_over,
            name=name,
            backend=backend,
            executor=executor,
            cache=cache,
            callbacks=callbacks,
        )
```

## SOLID Principles Summary

### Single Responsibility Principle ✅
- ExecutorStrategy: Define execution strategy
- Backend: Define execution environment
- Pipeline: Orchestrate node execution
- Cache: Handle caching
- Callbacks: Handle lifecycle events
- config.py: Manage global configuration

### Open/Closed Principle ✅
- Open for extension: Add new ExecutorStrategy subclasses without modifying Backend
- Closed for modification: LocalBackend doesn't change when adding SparkExecutor

### Liskov Substitution Principle ✅
- Any ExecutorStrategy can be used where parent type expected
- Any Backend can be used where Backend expected
- TaskExecutor and FrameworkExecutor are distinct hierarchies (no mixing)

### Interface Segregation Principle ✅
- TaskExecutor has map_executor (needs it)
- FrameworkExecutor does NOT have map_executor (doesn't need it)
- Separate hierarchies prevent interface pollution

### Dependency Inversion Principle ✅
- Backend depends on ExecutorStrategy abstraction (not concrete classes)
- Pipeline depends on Backend abstraction (not LocalBackend)
- Pipeline depends on ExecutorStrategy abstraction (injected at runtime)

This design is clean, testable, and follows SOLID principles throughout! 🎯