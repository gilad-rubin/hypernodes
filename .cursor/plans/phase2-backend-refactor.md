# Phase 2: Backend Refactoring

## Overview

Simplify the `Backend` abstract base class and refactor existing backend implementations (`LocalBackend`, `ModalBackend`) to cleanly separate infrastructure (WHERE) from execution strategy (HOW). Backends will now delegate execution logic to Executors.

## Goals

1. Simplify `Backend` ABC - remove execution strategy concerns
2. Refactor `LocalBackend` - make it trivial, delegate to executors
3. Update `ModalBackend` - integrate with RuntimeConfig and executors
4. Create `RemoteBackend` for Dask/Ray/Spark clusters (stub)
5. Extract `PipelineExecutionEngine` for shared execution logic

## Key Principle

**Backend = WHERE code runs**
- `LocalBackend` = local machine
- `ModalBackend` = Modal cloud with GPU/memory/etc
- `RemoteBackend` = remote cluster (Dask/Ray/Spark)

**Executor = HOW code runs** (from Phase 1)
- Already implemented: Sequential, Async, ThreadPool, ProcessPool
- Future: Daft, Dask, Ray

## File Changes

### 1. Simplify `src/hypernodes/backend.py`

#### Current Problems
- `LocalBackend` has 1670 lines with all execution logic
- Mixes infrastructure concerns with execution strategy
- `node_execution` and `map_execution` parameters conflate WHERE and HOW

#### Refactored Backend ABC

```python
"""Execution backends for running pipelines."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .callbacks import CallbackContext

if TYPE_CHECKING:
    from .pipeline import Pipeline


class Backend(ABC):
    """Abstract base class for pipeline execution backends.
    
    Backends define WHERE code runs (local machine, cloud, remote cluster).
    They delegate to Executors (from RuntimeConfig) for HOW code runs.
    
    All backends must implement run() and map() methods.
    """
    
    @abstractmethod
    def run(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute a pipeline with the given inputs.
        
        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values for root arguments
            ctx: Optional callback context
            output_name: Optional output name(s) to compute
            
        Returns:
            Dictionary containing the requested pipeline outputs
        """
        pass
    
    @abstractmethod
    def map(
        self,
        pipeline: "Pipeline",
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a pipeline over multiple items.
        
        Args:
            pipeline: The pipeline to execute
            items: List of input dictionaries (one per item)
            inputs: Shared inputs for all items
            ctx: Optional callback context
            output_name: Optional output name(s) to compute
            
        Returns:
            List of output dictionaries (one per item)
        """
        pass
```

#### Refactored LocalBackend

```python
class LocalBackend(Backend):
    """Local execution backend.
    
    Executes pipelines on the local machine. Delegates all execution
    logic to the Executor from RuntimeConfig.
    
    This backend is trivial - it just extracts the executor from
    the pipeline's configuration and uses it.
    """
    
    def __init__(self):
        """Initialize LocalBackend.
        
        No configuration needed - all execution strategy comes
        from RuntimeConfig.executor.
        """
        pass
    
    def run(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute pipeline using configured executor.
        
        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values
            ctx: Callback context (created if None)
            output_name: Optional output name(s) to compute
            
        Returns:
            Dictionary of pipeline outputs
        """
        # Get executor from pipeline's effective config
        executor = pipeline.effective_config.executor
        
        if executor is None:
            # Fallback: use default sequential executor
            from .executor import SequentialExecutor
            executor = SequentialExecutor()
        
        # Delegate to executor
        return executor.run_pipeline(pipeline, inputs, ctx, output_name)
    
    def map(
        self,
        pipeline: "Pipeline",
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute map using configured executor.
        
        Args:
            pipeline: The pipeline to execute
            items: List of input dictionaries
            inputs: Shared inputs for all items
            ctx: Callback context (created if None)
            output_name: Optional output name(s) to compute
            
        Returns:
            List of output dictionaries
        """
        # Get executor from pipeline's effective config
        executor = pipeline.effective_config.executor
        
        if executor is None:
            # Fallback: use default sequential executor
            from .executor import SequentialExecutor
            executor = SequentialExecutor()
        
        # Delegate to executor
        return executor.map_items(pipeline, items, inputs, ctx, output_name)
```

### 2. Extract PipelineExecutionEngine

Create `src/hypernodes/execution_engine.py` for reusable execution logic:

```python
"""Reusable pipeline execution engine.

This module contains the core execution logic that can be used by
any backend or executor. It handles:
- Topological execution order
- Dependency resolution
- Cache checking/storage
- Callback invocations
- Node signature computation
"""

import hashlib
import time
from typing import Any, Dict, List, Optional, Set, Union

import networkx as nx

from .cache import compute_signature, hash_code, hash_inputs
from .callbacks import CallbackContext


def _get_node_id(node) -> str:
    """Get a consistent node ID for callbacks and logging."""
    if hasattr(node, "name") and node.name:
        return node.name
    if hasattr(node, "func") and hasattr(node.func, "__name__"):
        return node.func.__name__
    if hasattr(node, "id"):
        return node.id
    if hasattr(node, "__name__"):
        return node.__name__
    return str(node)


def execute_pipeline_sequential(
    pipeline,
    inputs: Dict[str, Any],
    ctx: Optional[CallbackContext] = None,
    output_name: Union[str, List[str], None] = None,
) -> Dict[str, Any]:
    """Execute a pipeline sequentially with caching and callbacks.
    
    This is the core execution logic extracted from LocalBackend.
    Can be reused by any backend/executor that needs sequential execution.
    
    Args:
        pipeline: Pipeline to execute
        inputs: Input dictionary
        ctx: Callback context (created if None)
        output_name: Optional output name(s) to compute
        
    Returns:
        Dictionary of outputs
    """
    # Create or reuse callback context
    if ctx is None:
        ctx = CallbackContext()
    
    # Get configuration
    config = pipeline.effective_config
    callbacks = config.materialize_callbacks() if config.callbacks else []
    cache = config.materialize_cache() if config.cache else None
    
    # Set pipeline metadata
    node_ids = [_get_node_id(n) for n in pipeline.execution_order]
    ctx.set_pipeline_metadata(
        pipeline.id,
        {
            "total_nodes": len(pipeline.execution_order),
            "node_ids": node_ids,
            "pipeline_name": pipeline.name or pipeline.id,
        },
    )
    
    # Push pipeline onto hierarchy stack
    ctx.push_pipeline(pipeline.id)
    
    # Compute required nodes based on output_name
    required_nodes = pipeline._compute_required_nodes(output_name)
    nodes_to_execute = (
        required_nodes if required_nodes is not None else pipeline.execution_order
    )
    
    # Normalize output_name to set for filtering
    if output_name is None:
        requested_outputs = None
    elif isinstance(output_name, str):
        requested_outputs = {output_name}
    else:
        requested_outputs = set(output_name)
    
    # Trigger pipeline start callbacks
    pipeline_start_time = time.time()
    for callback in callbacks:
        callback.on_pipeline_start(pipeline.id, inputs, ctx)
    
    try:
        # Start with provided inputs
        available_values = dict(inputs)
        outputs = {}
        node_signatures = {}
        
        # Execute nodes in topological order
        for node in nodes_to_execute:
            # Get node inputs
            node_inputs = {
                param: available_values[param] for param in node.parameters
            }
            
            # Compute node signature for caching
            code_hash = hash_code(node.func)
            inputs_hash = hash_inputs(node_inputs)
            
            # Compute dependencies hash
            deps_signatures = []
            for param in node.parameters:
                if param in node_signatures:
                    deps_signatures.append(node_signatures[param])
            deps_hash = ":".join(sorted(deps_signatures))
            
            signature = compute_signature(
                code_hash=code_hash,
                inputs_hash=inputs_hash,
                deps_hash=deps_hash,
            )
            
            # Check cache if enabled
            cache_enabled = cache is not None and node.cache
            result = None
            
            if cache_enabled:
                result = cache.get(signature)
                if result is not None:
                    # Cache hit
                    for callback in callbacks:
                        callback.on_node_cached(_get_node_id(node), signature, ctx)
            
            if result is None:
                # Execute node
                node_start_time = time.time()
                for callback in callbacks:
                    callback.on_node_start(_get_node_id(node), node_inputs, ctx)
                
                try:
                    result = node(**node_inputs)
                    
                    node_duration = time.time() - node_start_time
                    for callback in callbacks:
                        callback.on_node_end(
                            _get_node_id(node),
                            {node.output_name: result},
                            node_duration,
                            ctx,
                        )
                
                except Exception as e:
                    for callback in callbacks:
                        callback.on_error(_get_node_id(node), e, ctx)
                    raise
                
                # Store in cache
                if cache_enabled:
                    cache.put(signature, result)
            
            # Store output and signature
            outputs[node.output_name] = result
            available_values[node.output_name] = result
            node_signatures[node.output_name] = signature
        
        # Filter outputs if specific outputs were requested
        if requested_outputs is not None:
            outputs = {k: v for k, v in outputs.items() if k in requested_outputs}
        
        # Trigger pipeline end callbacks
        pipeline_duration = time.time() - pipeline_start_time
        for callback in callbacks:
            callback.on_pipeline_end(pipeline.id, outputs, pipeline_duration, ctx)
        
        return outputs
    
    finally:
        # Pop pipeline from hierarchy stack
        ctx.pop_pipeline()
```

### 3. Update ModalBackend

```python
class ModalBackend(Backend):
    """Remote execution backend on Modal Labs serverless infrastructure.
    
    Executes pipelines on Modal's cloud with configurable resources.
    Integrates with RuntimeConfig for unified configuration.
    
    Args:
        image: Modal image with dependencies (required)
        gpu: GPU type ("A100", "A10G", "T4", "any", None)
        memory: Memory limit ("32GB", "256GB", etc.)
        cpu: CPU cores (1.0, 2.0, 8.0, etc.)
        timeout: Max execution time in seconds (default: 3600)
        volumes: Volume mounts {"/path": modal.Volume}
        secrets: Modal secrets for API keys, etc.
        max_concurrent: Max parallel containers for .map() (default: 100)
    """
    
    def __init__(
        self,
        image: Any,
        gpu: Optional[str] = None,
        memory: Optional[str] = None,
        cpu: Optional[float] = None,
        timeout: int = 3600,
        volumes: Optional[Dict[str, Any]] = None,
        secrets: Optional[list] = None,
        max_concurrent: int = 100,
    ):
        # Lazy import modal
        self.modal = _import_modal()
        self.cloudpickle = _import_cloudpickle()
        
        # Validate image
        if not isinstance(image, self.modal.Image):
            raise TypeError(f"image must be a modal.Image, got {type(image)}")
        
        self.image = image
        self.gpu = gpu
        self.memory = memory
        self.cpu = cpu
        self.timeout = timeout
        self.volumes = volumes or {}
        self.secrets = secrets or []
        self.max_concurrent = max_concurrent
        
        # Create Modal app
        self._app = self.modal.App(name="hypernodes-pipeline")
        
        # Create remote function
        self._create_remote_function()
    
    def _create_remote_function(self):
        """Create Modal function with specified resources."""
        # Build function kwargs
        function_kwargs = {
            "image": self.image,
            "timeout": self.timeout,
            "serialized": True,
        }
        
        if self.gpu:
            function_kwargs["gpu"] = self.gpu
        if self.memory:
            function_kwargs["memory"] = self.memory
        if self.cpu:
            function_kwargs["cpu"] = self.cpu
        if self.volumes:
            function_kwargs["volumes"] = self.volumes
        if self.secrets:
            function_kwargs["secrets"] = self.secrets
        
        # Create remote execution function
        @self._app.function(**function_kwargs)
        def _remote_execute(serialized_payload: bytes) -> bytes:
            """Executes pipeline on Modal infrastructure."""
            import cloudpickle
            
            # Deserialize pipeline, inputs, config, output_name
            pipeline, inputs, ctx_data, config, output_name = cloudpickle.loads(
                serialized_payload
            )
            
            # Reconstruct callback context
            from hypernodes.callbacks import CallbackContext
            
            ctx = CallbackContext()
            if ctx_data:
                for key, value in ctx_data.items():
                    ctx.data[key] = value
            
            # Get executor from config
            executor = config.executor
            if executor is None:
                from hypernodes.executor import SequentialExecutor
                executor = SequentialExecutor()
            
            # Execute using executor
            results = executor.run_pipeline(pipeline, inputs, ctx, output_name)
            
            # Serialize and return results
            return cloudpickle.dumps(results)
        
        self._remote_execute = _remote_execute
    
    def _serialize_payload(
        self,
        pipeline,
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> bytes:
        """Serialize pipeline, inputs, config for remote execution."""
        # Extract context data
        ctx_data = {}
        if ctx:
            excluded_prefixes = (
                "progress_bar:",
                "span:",
                "map_span",
                "map_item_span:",
                "current_span",
                "map_node_bars",
            )
            ctx_data = {
                k: v
                for k, v in ctx.data.items()
                if not any(k.startswith(prefix) for prefix in excluded_prefixes)
                and isinstance(v, (str, int, float, bool, list, dict, type(None)))
            }
        
        # Get config (without backend to avoid recursion)
        config = pipeline.effective_config
        
        # Replace backend with LocalBackend for remote execution
        original_backend = config.backend
        from .config import RuntimeConfig
        remote_config = RuntimeConfig(
            backend=LocalBackend(),  # Execute locally on Modal
            executor=config.executor,
            cache=config.cache,
            callbacks=None,  # Don't serialize callbacks
            retries=config.retries,
            timeout=config.timeout,
        )
        
        try:
            # Serialize everything
            payload = (pipeline, inputs, ctx_data, remote_config, output_name)
            result = self.cloudpickle.dumps(payload)
        finally:
            pass
        
        return result
    
    def run(
        self,
        pipeline,
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute pipeline on Modal."""
        if ctx is None:
            ctx = CallbackContext()
        
        # Serialize payload
        serialized_payload = self._serialize_payload(pipeline, inputs, ctx, output_name)
        
        # Submit to Modal
        with self._app.run():
            result_bytes = self._remote_execute.remote(serialized_payload)
        
        # Deserialize results
        results = self.cloudpickle.loads(result_bytes)
        
        return results
    
    def map(
        self,
        pipeline,
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute map on Modal."""
        results = []
        for item in items:
            merged = {**inputs, **item}
            results.append(self.run(pipeline, merged, ctx, output_name))
        return results
```

### 4. Create RemoteBackend (Stub)

```python
class RemoteBackend(Backend):
    """Generic remote execution backend for clusters.
    
    Connects to remote compute clusters (Dask, Ray, Spark) and
    executes pipelines there.
    
    Args:
        kind: Cluster type ("dask", "ray", "spark")
        address: Cluster scheduler address
        security: Security configuration (optional)
    """
    
    def __init__(
        self,
        kind: str,
        address: str,
        security: Optional[Any] = None,
    ):
        self.kind = kind
        self.address = address
        self.security = security
        
        if kind not in ("dask", "ray", "spark"):
            raise ValueError(f"Unsupported cluster kind: {kind}")
    
    def run(
        self,
        pipeline,
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute pipeline on remote cluster."""
        raise NotImplementedError(
            f"RemoteBackend for {self.kind} not yet implemented"
        )
    
    def map(
        self,
        pipeline,
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute map on remote cluster."""
        raise NotImplementedError(
            f"RemoteBackend for {self.kind} not yet implemented"
        )
```

## Verification Tests

### Test File: `tests/test_phase2_backends.py`

```python
"""Tests for Phase 2: Backend Refactoring."""

import pytest
from hypernodes import node, Pipeline
from hypernodes.backend import LocalBackend, ModalBackend, RemoteBackend
from hypernodes.config import RuntimeConfig
from hypernodes.executor import SequentialExecutor, ThreadPoolExecutor


@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2


@node(output_name="result")
def add_one(doubled: int) -> int:
    return doubled + 1


# Test 1: LocalBackend delegates to executor
def test_local_backend_delegates_to_executor():
    """Test that LocalBackend extracts executor from config."""
    pipeline = Pipeline(nodes=[double, add_one])
    
    # Set config with specific executor
    config = RuntimeConfig(
        backend=LocalBackend(),
        executor=SequentialExecutor(),
    )
    
    # Set on pipeline
    pipeline = pipeline.with_config(config)
    
    # Run
    result = pipeline.run({"x": 5})
    
    assert result["doubled"] == 10
    assert result["result"] == 11


# Test 2: LocalBackend with different executors
def test_local_backend_different_executors():
    """Test LocalBackend works with different executors."""
    pipeline = Pipeline(nodes=[double])
    
    # Test with Sequential
    config1 = RuntimeConfig(
        backend=LocalBackend(),
        executor=SequentialExecutor(),
    )
    result1 = LocalBackend().run(pipeline.with_config(config1), {"x": 5})
    assert result1["doubled"] == 10
    
    # Test with ThreadPool
    config2 = RuntimeConfig(
        backend=LocalBackend(),
        executor=ThreadPoolExecutor(max_workers=2),
    )
    result2 = LocalBackend().run(pipeline.with_config(config2), {"x": 5})
    assert result2["doubled"] == 10


# Test 3: LocalBackend map delegates to executor
def test_local_backend_map():
    """Test LocalBackend map delegates to executor."""
    pipeline = Pipeline(nodes=[double])
    
    config = RuntimeConfig(
        backend=LocalBackend(),
        executor=SequentialExecutor(),
    )
    pipeline = pipeline.with_config(config)
    
    items = [{"x": 1}, {"x": 2}, {"x": 3}]
    results = LocalBackend().map(pipeline, items, {})
    
    assert len(results) == 3
    assert results[0]["doubled"] == 2
    assert results[1]["doubled"] == 4
    assert results[2]["doubled"] == 6


# Test 4: LocalBackend fallback executor
def test_local_backend_fallback_executor():
    """Test LocalBackend uses fallback if no executor in config."""
    pipeline = Pipeline(nodes=[double])
    
    # Config without executor
    config = RuntimeConfig(
        backend=LocalBackend(),
        executor=None,
    )
    pipeline = pipeline.with_config(config)
    
    # Should use SequentialExecutor as fallback
    result = LocalBackend().run(pipeline, {"x": 5})
    assert result["doubled"] == 10


# Test 5: RemoteBackend raises NotImplementedError
def test_remote_backend_not_implemented():
    """Test RemoteBackend raises NotImplementedError."""
    backend = RemoteBackend(kind="dask", address="tcp://scheduler:8786")
    pipeline = Pipeline(nodes=[double])
    
    with pytest.raises(NotImplementedError):
        backend.run(pipeline, {"x": 5})
    
    with pytest.raises(NotImplementedError):
        backend.map(pipeline, [{"x": 1}], {})


# Test 6: RemoteBackend validates kind
def test_remote_backend_validates_kind():
    """Test RemoteBackend validates cluster kind."""
    with pytest.raises(ValueError):
        RemoteBackend(kind="invalid", address="tcp://localhost:8786")


# Test 7: Backend abstraction
def test_backend_is_abstract():
    """Test Backend cannot be instantiated directly."""
    from hypernodes.backend import Backend
    
    with pytest.raises(TypeError):
        Backend()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Migration Notes

### Breaking Changes

1. **LocalBackend constructor**: No more `node_execution`, `map_execution`, `max_workers`, `executor` parameters
   - **Before**: `LocalBackend(node_execution="threaded", max_workers=8)`
   - **After**: Use `RuntimeConfig(backend=LocalBackend(), executor=ThreadPoolExecutor(max_workers=8))`

2. **Backend responsibilities**: Backends now only define WHERE, not HOW
   - Execution logic moved to Executors
   - Configuration bundled in RuntimeConfig

3. **ModalBackend**: Removed `node_execution`, `map_execution` parameters
   - Use RuntimeConfig with appropriate executor instead

### Code Removed

- `LocalBackend._run_sequential()` → moved to `execution_engine.execute_pipeline_sequential()`
- `LocalBackend._run_async()` → logic moved to `AsyncExecutor`
- `LocalBackend._run_threaded()` → logic moved to `ThreadPoolExecutor`
- `LocalBackend._run_parallel()` → logic moved to `ProcessPoolExecutor`
- All `_map_*()` methods → logic moved to respective executors
- `PipelineExecutionEngine` class → extracted to `execution_engine.py`

## Success Criteria

- [ ] `Backend` ABC simplified (only run/map methods)
- [ ] `LocalBackend` refactored to ~50 lines (from 1670)
- [ ] `LocalBackend` delegates to executors correctly
- [ ] `execution_engine.py` created with reusable logic
- [ ] `ModalBackend` updated to work with RuntimeConfig
- [ ] `RemoteBackend` stub created
- [ ] All 7 verification tests pass
- [ ] No functionality lost from old implementation

## Dependencies

- Phase 1 must be complete (RuntimeConfig, Executor classes)
- Existing cache, callbacks, node, pipeline modules

## Next Phase

Phase 3 will convert DaftBackend to DaftExecutor.

