# Phase 1: Core Abstractions

## Overview

Create the foundational abstractions for the new execution configuration architecture: `RuntimeConfig`, `Executor` interface, and global configuration management. These are the building blocks for the entire refactor.

## Goals

1. Define `RuntimeConfig` - the immutable configuration bundle
2. Define `Executor` abstract base class - the HOW of execution
3. Implement basic executors (Sequential, Async, ThreadPool, ProcessPool)
4. Implement global configuration management
5. Implement lazy initialization patterns for cache and callbacks

## File 1: `src/hypernodes/config.py`

### RuntimeConfig Class

```python
"""Runtime configuration for pipeline execution."""

from dataclasses import dataclass, field
from typing import Optional, List, Callable, Union, Any
from contextlib import contextmanager

from .cache import Cache


@dataclass(frozen=True)
class CacheSpec:
    """Lazy cache specification for serializable configuration.
    
    Instead of storing live Cache objects (which may have file handles,
    network connections, etc.), store specifications that can be
    instantiated when needed.
    
    Attributes:
        kind: Type of cache ("disk", "memory", "redis", etc.)
        path: Path for disk cache (optional)
        options: Additional options (max_size, ttl, etc.)
    """
    kind: str
    path: Optional[str] = None
    options: dict = field(default_factory=dict)
    
    def create(self) -> Cache:
        """Instantiate the actual cache from this specification.
        
        Returns:
            Cache instance
        """
        if self.kind == "disk":
            from .cache import DiskCache
            if self.path is None:
                raise ValueError("DiskCache requires 'path' parameter")
            return DiskCache(path=self.path)
        else:
            raise ValueError(f"Unknown cache kind: {self.kind}")


@dataclass(frozen=True)
class RuntimeConfig:
    """Immutable runtime configuration for pipeline execution.
    
    Bundles all execution-related configuration in one place:
    - backend: WHERE code runs (local, Modal, remote cluster)
    - executor: HOW code runs (sequential, async, threaded, parallel)
    - cache: Caching strategy (can be Cache, CacheSpec, or factory)
    - callbacks: Observability hooks (list of callbacks or factories)
    - retries: Retry policy for failed nodes
    - timeout: Execution timeout in seconds
    
    This object is immutable and serializable (if cache/callbacks are specs).
    """
    backend: Optional[Any] = None  # Backend instance (LocalBackend, ModalBackend, etc.)
    executor: Optional[Any] = None  # Executor instance
    cache: Optional[Union[Cache, CacheSpec, Callable[[], Cache]]] = None
    callbacks: Optional[List[Union[Any, Callable]]] = None  # PipelineCallback instances or factories
    retries: int = 0
    timeout: Optional[float] = None
    
    def materialize_cache(self) -> Optional[Cache]:
        """Materialize cache from spec/factory if needed.
        
        Returns:
            Actual Cache instance or None
        """
        if self.cache is None:
            return None
        
        if isinstance(self.cache, CacheSpec):
            return self.cache.create()
        
        if callable(self.cache) and not isinstance(self.cache, Cache):
            # Factory function
            return self.cache()
        
        # Already a Cache instance
        return self.cache
    
    def materialize_callbacks(self) -> List[Any]:
        """Materialize callbacks from factories if needed.
        
        Returns:
            List of actual callback instances
        """
        if self.callbacks is None:
            return []
        
        materialized = []
        for cb in self.callbacks:
            if callable(cb) and not hasattr(cb, 'on_pipeline_start'):
                # Factory function - call it
                materialized.append(cb())
            else:
                # Already a callback instance
                materialized.append(cb)
        
        return materialized
    
    @staticmethod
    @contextmanager
    def use(config: 'RuntimeConfig'):
        """Context manager for temporary configuration override.
        
        Args:
            config: RuntimeConfig to use temporarily
            
        Example:
            >>> with RuntimeConfig.use(modal_config):
            ...     pipeline.run({"x": 1})  # Uses modal_config
        """
        from . import get_default_config, set_default_config
        
        old_config = get_default_config()
        set_default_config(config)
        try:
            yield config
        finally:
            set_default_config(old_config)


# Global configuration management
_global_default_config: Optional[RuntimeConfig] = None


def set_default_config(config: RuntimeConfig) -> None:
    """Set the global default configuration.
    
    This configuration will be used by all pipelines that don't
    specify their own configuration.
    
    Args:
        config: RuntimeConfig to use as default
        
    Example:
        >>> from hypernodes import set_default_config, RuntimeConfig, LocalBackend
        >>> from hypernodes.executor import ThreadPoolExecutor
        >>> 
        >>> config = RuntimeConfig(
        ...     backend=LocalBackend(),
        ...     executor=ThreadPoolExecutor(max_workers=8),
        ...     cache=CacheSpec(kind="disk", path=".cache")
        ... )
        >>> set_default_config(config)
    """
    global _global_default_config
    _global_default_config = config


def get_default_config() -> RuntimeConfig:
    """Get the global default configuration.
    
    Returns:
        Current global RuntimeConfig, or library default if not set
        
    Library defaults:
        - backend: LocalBackend()
        - executor: ThreadPoolExecutor(max_workers=4)
        - cache: None
        - callbacks: []
    """
    global _global_default_config
    
    if _global_default_config is not None:
        return _global_default_config
    
    # Library defaults
    from .backend import LocalBackend
    from .executor import ThreadPoolExecutor
    
    return RuntimeConfig(
        backend=LocalBackend(),
        executor=ThreadPoolExecutor(max_workers=4),
        cache=None,
        callbacks=None,
        retries=0,
        timeout=None
    )
```

## File 2: `src/hypernodes/executor.py`

### Executor Abstract Base Class and Implementations

```python
"""Executor abstractions for pipeline execution strategies."""

import asyncio
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import (
    ThreadPoolExecutor as StdThreadPoolExecutor,
    ProcessPoolExecutor as StdProcessPoolExecutor,
    as_completed,
)
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .callbacks import CallbackContext

if TYPE_CHECKING:
    from .pipeline import Pipeline


class Executor(ABC):
    """Abstract base class for execution strategies.
    
    Executors define HOW code runs (sequential, parallel, distributed, etc.)
    while Backends define WHERE code runs (local, cloud, remote cluster).
    
    An Executor handles both:
    1. Pipeline DAG execution (run_pipeline)
    2. Map operations (map_items)
    
    Some executors (Daft, Dask) naturally handle both. Others can be
    composed using CompositeExecutor.
    """
    
    @abstractmethod
    def run_pipeline(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute a pipeline DAG with this execution strategy.
        
        Args:
            pipeline: The pipeline to execute
            inputs: Input dictionary for root arguments
            ctx: Callback context for tracking execution
            output_name: Optional output name(s) to compute
            
        Returns:
            Dictionary of pipeline outputs
        """
        pass
    
    @abstractmethod
    def map_items(
        self,
        pipeline: "Pipeline",
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute pipeline over multiple items.
        
        Args:
            pipeline: The pipeline to execute
            items: List of input dictionaries (one per item)
            inputs: Shared inputs for all items
            ctx: Callback context for tracking execution
            output_name: Optional output name(s) to compute
            
        Returns:
            List of output dictionaries (one per item)
        """
        pass


class SequentialExecutor(Executor):
    """Sequential execution - one operation at a time.
    
    Simplest executor. Nodes execute in topological order,
    items process one at a time. Predictable and easy to debug.
    """
    
    def run_pipeline(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute pipeline sequentially."""
        from .backend import LocalBackend
        
        # Delegate to LocalBackend's sequential execution
        # (We'll refactor LocalBackend to extract this logic)
        backend = LocalBackend()
        return backend._run_sequential(pipeline, inputs, ctx, output_name)
    
    def map_items(
        self,
        pipeline: "Pipeline",
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute map sequentially - one item at a time."""
        results = []
        for i, item in enumerate(items):
            if ctx:
                ctx.set("_in_map", True)
                ctx.set("_map_item_index", i)
                ctx.set("_map_item_start_time", time.time())
            
            # Merge item with shared inputs
            item_inputs = {**inputs, **item}
            
            # Execute pipeline for this item
            result = self.run_pipeline(pipeline, item_inputs, ctx, output_name)
            results.append(result)
            
            if ctx:
                ctx.set("_in_map", False)
        
        return results


class AsyncExecutor(Executor):
    """Async execution using asyncio for concurrency.
    
    Best for I/O-bound work. Independent nodes execute concurrently,
    and items are processed concurrently up to max_workers limit.
    
    Args:
        max_workers: Maximum concurrent operations (default: 10)
    """
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
    
    def run_pipeline(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute pipeline with async concurrency."""
        # Run async execution in event loop
        return asyncio.run(self._run_async(pipeline, inputs, ctx, output_name))
    
    async def _run_async(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Async implementation - will be extracted from LocalBackend."""
        from .backend import LocalBackend
        
        backend = LocalBackend()
        return await backend._run_async(pipeline, inputs, ctx, output_name)
    
    def map_items(
        self,
        pipeline: "Pipeline",
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute map with async concurrency."""
        return asyncio.run(self._map_async(pipeline, items, inputs, ctx, output_name))
    
    async def _map_async(
        self,
        pipeline: "Pipeline",
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Async map implementation."""
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_item(i: int, item: Dict[str, Any]):
            async with semaphore:
                # Create context for this item
                item_ctx = CallbackContext()
                if ctx:
                    for key, value in ctx.data.items():
                        if not key.startswith("_"):
                            item_ctx.data[key] = value
                
                item_ctx.set("_in_map", True)
                item_ctx.set("_map_item_index", i)
                item_ctx.set("_map_item_start_time", time.time())
                
                # Merge inputs
                item_inputs = {**inputs, **item}
                
                # Execute in executor (to avoid nested event loops)
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: self.run_pipeline(pipeline, item_inputs, item_ctx, output_name)
                )
                
                return result
        
        # Create tasks for all items
        tasks = [process_item(i, item) for i, item in enumerate(items)]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        return list(results)


class ThreadPoolExecutor(Executor):
    """Thread-based parallel execution.
    
    Good for I/O + CPU mixed workloads. Overcomes GIL for I/O operations.
    
    Args:
        max_workers: Maximum worker threads (default: CPU count)
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or os.cpu_count() or 4
        self._executor: Optional[StdThreadPoolExecutor] = None
    
    def _get_executor(self) -> StdThreadPoolExecutor:
        """Lazy initialize executor."""
        if self._executor is None:
            self._executor = StdThreadPoolExecutor(max_workers=self.max_workers)
        return self._executor
    
    def run_pipeline(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute pipeline with thread-based parallelism."""
        from .backend import LocalBackend
        
        backend = LocalBackend()
        return backend._run_threaded(pipeline, inputs, ctx, output_name)
    
    def map_items(
        self,
        pipeline: "Pipeline",
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute map with thread-based parallelism."""
        executor = self._get_executor()
        
        def process_item(i: int, item: Dict[str, Any]):
            # Create context for this item
            item_ctx = CallbackContext()
            if ctx:
                for key, value in ctx.data.items():
                    if not key.startswith("_"):
                        item_ctx.data[key] = value
            
            item_ctx.set("_in_map", True)
            item_ctx.set("_map_item_index", i)
            item_ctx.set("_map_item_start_time", time.time())
            
            # Merge inputs
            item_inputs = {**inputs, **item}
            
            # Execute pipeline
            result = self.run_pipeline(pipeline, item_inputs, item_ctx, output_name)
            
            return (i, result)
        
        # Submit all items
        futures = [
            executor.submit(process_item, i, item) for i, item in enumerate(items)
        ]
        
        # Collect results in order
        results = [None] * len(items)
        for future in as_completed(futures):
            i, result = future.result()
            results[i] = result
        
        return results
    
    def shutdown(self):
        """Shutdown the thread pool."""
        if self._executor:
            self._executor.shutdown(wait=True)


class ProcessPoolExecutor(Executor):
    """Process-based true parallel execution.
    
    Best for CPU-bound work. Utilizes multiple CPU cores.
    Requires picklable functions and data.
    
    Args:
        max_workers: Maximum worker processes (default: CPU count)
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or os.cpu_count() or 4
        self._executor: Optional[StdProcessPoolExecutor] = None
    
    def _get_executor(self) -> StdProcessPoolExecutor:
        """Lazy initialize executor."""
        if self._executor is None:
            self._executor = StdProcessPoolExecutor(max_workers=self.max_workers)
        return self._executor
    
    def run_pipeline(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute pipeline with process-based parallelism."""
        from .backend import LocalBackend
        
        backend = LocalBackend()
        return backend._run_parallel(pipeline, inputs, ctx, output_name)
    
    def map_items(
        self,
        pipeline: "Pipeline",
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute map with process-based parallelism."""
        executor = self._get_executor()
        
        def process_item(i: int, item: Dict[str, Any]):
            # Create context for this item
            item_ctx = CallbackContext()
            if ctx:
                for key, value in ctx.data.items():
                    if not key.startswith("_"):
                        item_ctx.data[key] = value
            
            item_ctx.set("_in_map", True)
            item_ctx.set("_map_item_index", i)
            item_ctx.set("_map_item_start_time", time.time())
            
            # Merge inputs
            item_inputs = {**inputs, **item}
            
            # Execute pipeline
            from .backend import LocalBackend
            backend = LocalBackend()
            result = backend._run_sequential(pipeline, item_inputs, item_ctx, output_name)
            
            return (i, result)
        
        # Submit all items
        futures = [
            executor.submit(process_item, i, item) for i, item in enumerate(items)
        ]
        
        # Collect results in order
        results = [None] * len(items)
        for future in as_completed(futures):
            i, result = future.result()
            results[i] = result
        
        return results
    
    def shutdown(self):
        """Shutdown the process pool."""
        if self._executor:
            self._executor.shutdown(wait=True)
```

## Verification Tests

### Test File: `tests/test_phase1_config.py`

```python
"""Tests for Phase 1: Core Abstractions (RuntimeConfig and basic executors)."""

import pytest
import time
from hypernodes import node, Pipeline
from hypernodes.config import (
    RuntimeConfig,
    CacheSpec,
    set_default_config,
    get_default_config,
)
from hypernodes.executor import (
    SequentialExecutor,
    AsyncExecutor,
    ThreadPoolExecutor,
    ProcessPoolExecutor,
)
from hypernodes.backend import LocalBackend
from hypernodes.cache import DiskCache
import tempfile
import shutil


# Test nodes
@node(output_name="doubled")
def double(x: int) -> int:
    """Simple computation."""
    return x * 2


@node(output_name="result")
def add_one(doubled: int) -> int:
    """Depends on double."""
    return doubled + 1


@node(output_name="slow_result")
def slow_compute(x: int) -> int:
    """Slow computation for testing parallelism."""
    time.sleep(0.1)
    return x * 2


# Test 1: RuntimeConfig creation and immutability
def test_runtime_config_creation():
    """Test creating RuntimeConfig with various parameters."""
    config = RuntimeConfig(
        backend=LocalBackend(),
        executor=SequentialExecutor(),
        cache=None,
        callbacks=None,
        retries=3,
        timeout=60.0,
    )
    
    assert config.backend is not None
    assert config.executor is not None
    assert config.retries == 3
    assert config.timeout == 60.0
    
    # Test immutability
    with pytest.raises(Exception):  # dataclass frozen
        config.retries = 5


# Test 2: CacheSpec lazy initialization
def test_cache_spec():
    """Test CacheSpec creation and materialization."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create spec
        spec = CacheSpec(kind="disk", path=temp_dir)
        
        # Spec is not a Cache instance
        assert not isinstance(spec, DiskCache)
        
        # Materialize it
        cache = spec.create()
        assert isinstance(cache, DiskCache)
        assert cache.path.name == temp_dir.split("/")[-1]
        
    finally:
        shutil.rmtree(temp_dir)


# Test 3: RuntimeConfig.materialize_cache
def test_runtime_config_materialize_cache():
    """Test cache materialization from spec."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Config with CacheSpec
        config = RuntimeConfig(
            backend=LocalBackend(),
            executor=SequentialExecutor(),
            cache=CacheSpec(kind="disk", path=temp_dir),
        )
        
        # Materialize
        cache = config.materialize_cache()
        assert isinstance(cache, DiskCache)
        
    finally:
        shutil.rmtree(temp_dir)


# Test 4: RuntimeConfig with cache factory
def test_runtime_config_cache_factory():
    """Test cache materialization from factory function."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Config with factory
        config = RuntimeConfig(
            backend=LocalBackend(),
            executor=SequentialExecutor(),
            cache=lambda: DiskCache(path=temp_dir),
        )
        
        # Materialize
        cache = config.materialize_cache()
        assert isinstance(cache, DiskCache)
        
    finally:
        shutil.rmtree(temp_dir)


# Test 5: Global config management
def test_global_config():
    """Test set_default_config and get_default_config."""
    # Get default
    default = get_default_config()
    assert default.backend is not None
    assert default.executor is not None
    
    # Set custom config
    custom_config = RuntimeConfig(
        backend=LocalBackend(),
        executor=SequentialExecutor(),
        retries=5,
    )
    set_default_config(custom_config)
    
    # Verify it's set
    current = get_default_config()
    assert current.retries == 5
    
    # Reset to default
    set_default_config(default)


# Test 6: RuntimeConfig.use context manager
def test_runtime_config_context_manager():
    """Test temporary config override with context manager."""
    # Get current default
    original = get_default_config()
    
    # Create temporary config
    temp_config = RuntimeConfig(
        backend=LocalBackend(),
        executor=SequentialExecutor(),
        retries=10,
    )
    
    # Use context manager
    with RuntimeConfig.use(temp_config):
        current = get_default_config()
        assert current.retries == 10
    
    # Verify restored
    restored = get_default_config()
    assert restored.retries == original.retries


# Test 7: SequentialExecutor
def test_sequential_executor():
    """Test SequentialExecutor execution."""
    pipeline = Pipeline(nodes=[double, add_one])
    
    config = RuntimeConfig(
        backend=LocalBackend(),
        executor=SequentialExecutor(),
    )
    
    # Note: Pipeline API update happens in Phase 5
    # For now, test executor directly
    executor = SequentialExecutor()
    result = executor.run_pipeline(pipeline, {"x": 5})
    
    assert result["doubled"] == 10
    assert result["result"] == 11


# Test 8: SequentialExecutor map
def test_sequential_executor_map():
    """Test SequentialExecutor map operation."""
    pipeline = Pipeline(nodes=[double])
    
    executor = SequentialExecutor()
    items = [{"x": 1}, {"x": 2}, {"x": 3}]
    results = executor.map_items(pipeline, items, {})
    
    assert len(results) == 3
    assert results[0]["doubled"] == 2
    assert results[1]["doubled"] == 4
    assert results[2]["doubled"] == 6


# Test 9: ThreadPoolExecutor
def test_thread_pool_executor():
    """Test ThreadPoolExecutor execution."""
    pipeline = Pipeline(nodes=[double, add_one])
    
    executor = ThreadPoolExecutor(max_workers=4)
    result = executor.run_pipeline(pipeline, {"x": 5})
    
    assert result["doubled"] == 10
    assert result["result"] == 11
    
    executor.shutdown()


# Test 10: ThreadPoolExecutor map (verify parallelism)
def test_thread_pool_executor_map_parallel():
    """Test ThreadPoolExecutor map with parallelism speedup."""
    pipeline = Pipeline(nodes=[slow_compute])
    
    # Sequential baseline
    seq_executor = SequentialExecutor()
    items = [{"x": i} for i in range(5)]
    
    start = time.time()
    seq_executor.map_items(pipeline, items, {})
    seq_duration = time.time() - start
    
    # Parallel execution
    thread_executor = ThreadPoolExecutor(max_workers=5)
    
    start = time.time()
    thread_executor.map_items(pipeline, items, {})
    parallel_duration = time.time() - start
    
    # Parallel should be faster (5 items * 0.1s = 0.5s sequential)
    # With 5 workers, should be ~0.1s
    assert parallel_duration < seq_duration * 0.5
    
    thread_executor.shutdown()


# Test 11: AsyncExecutor
def test_async_executor():
    """Test AsyncExecutor execution."""
    pipeline = Pipeline(nodes=[double, add_one])
    
    executor = AsyncExecutor(max_workers=10)
    result = executor.run_pipeline(pipeline, {"x": 5})
    
    assert result["doubled"] == 10
    assert result["result"] == 11


# Test 12: AsyncExecutor map
def test_async_executor_map():
    """Test AsyncExecutor map operation."""
    pipeline = Pipeline(nodes=[double])
    
    executor = AsyncExecutor(max_workers=10)
    items = [{"x": i} for i in range(10)]
    results = executor.map_items(pipeline, items, {})
    
    assert len(results) == 10
    assert all(results[i]["doubled"] == i * 2 for i in range(10))


# Test 13: ProcessPoolExecutor (requires picklable functions)
def test_process_pool_executor():
    """Test ProcessPoolExecutor execution."""
    pipeline = Pipeline(nodes=[double, add_one])
    
    executor = ProcessPoolExecutor(max_workers=2)
    result = executor.run_pipeline(pipeline, {"x": 5})
    
    assert result["doubled"] == 10
    assert result["result"] == 11
    
    executor.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Success Criteria

- [ ] `RuntimeConfig` class created with all required fields
- [ ] `CacheSpec` class created with `create()` method
- [ ] Global config management functions work correctly
- [ ] `RuntimeConfig.use()` context manager works
- [ ] `Executor` abstract base class defined
- [ ] `SequentialExecutor` implemented and tested
- [ ] `AsyncExecutor` implemented and tested
- [ ] `ThreadPoolExecutor` implemented and tested
- [ ] `ProcessPoolExecutor` implemented and tested
- [ ] All 13 verification tests pass
- [ ] Cache lazy initialization works (CacheSpec and factory)
- [ ] Callback lazy initialization works (factory pattern)

## Dependencies

- Python 3.8+
- `dataclasses` (built-in)
- `asyncio` (built-in)
- `concurrent.futures` (built-in)
- Existing `cache.py`, `callbacks.py`, `backend.py` modules

## Next Phase

Phase 2 will refactor the Backend classes to use these new abstractions.

