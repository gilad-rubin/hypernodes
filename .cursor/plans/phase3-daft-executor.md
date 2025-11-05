# Phase 3: DaftExecutor Implementation

## Overview

Convert `DaftBackend` to `DaftExecutor`, changing it from a WHERE (backend) to a HOW (executor). This aligns with the architecture where Daft is an execution strategy, not an infrastructure choice. The DaftExecutor will implement the `Executor` interface and handle both pipeline DAG execution and map operations using Daft DataFrames.

## Goals

1. Rename `src/hypernodes/daft_backend.py` → `src/hypernodes/daft_executor.py`
2. Rename class `DaftBackend` → `DaftExecutor`
3. Implement `Executor` interface (`run_pipeline`, `map_items`)
4. Keep all existing Daft-specific logic (UDF conversion, DataFrame ops, etc.)
5. Add stubs for `DaskExecutor` and `RayExecutor` (future work)
6. Update all tests and examples

## Key Principle

**Daft is an execution strategy (HOW), not infrastructure (WHERE)**

- Daft runs on local machine or Ray cluster
- The WHERE is determined by Daft's runner: `DaftExecutor(runner="local")` or `DaftExecutor(runner="ray", ray_address="...")`
- This is cleaner than treating Daft as a Backend

## File Changes

### 1. Rename and Refactor: `src/hypernodes/daft_executor.py`

#### Key Changes

```python
"""Daft execution executor for HyperNodes pipelines.

This executor automatically converts HyperNodes pipelines into Daft DataFrames
using next-generation UDFs (@daft.func, @daft.cls, @daft.func.batch).

Key features:
- Automatic conversion of nodes to Daft UDFs
- Lazy evaluation and optimization
- Automatic parallelization
- Support for generators, async, and batch operations
- Runs on local machine or Ray cluster via runner parameter
"""

import importlib.util
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from .executor import Executor
from .callbacks import CallbackContext

try:
    import daft
    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False

if TYPE_CHECKING:
    from .pipeline import Pipeline


class DaftExecutor(Executor):
    """Daft execution executor that converts HyperNodes pipelines to Daft DataFrames.
    
    This executor translates HyperNodes pipelines into Daft operations:
    - Nodes become @daft.func UDFs
    - Map operations become DataFrame operations
    - Pipelines are converted to lazy DataFrame transformations
    
    The executor can run on:
    - Local machine: DaftExecutor(runner="local")
    - Ray cluster: DaftExecutor(runner="ray", ray_address="ray://...")
    
    Args:
        runner: Execution runner ("local" or "ray", default: "local")
        ray_address: Ray cluster address (optional, only for runner="ray")
        collect: Whether to automatically collect results (default: True)
        show_plan: Whether to print the execution plan (default: False)
        debug: Enable debug logging (default: False)
    
    Example:
        >>> from hypernodes import node, Pipeline
        >>> from hypernodes.daft_executor import DaftExecutor
        >>> 
        >>> @node(output_name="result")
        >>> def add_one(x: int) -> int:
        >>>     return x + 1
        >>> 
        >>> pipeline = Pipeline(nodes=[add_one])
        >>> config = RuntimeConfig(
        ...     backend=LocalBackend(),
        ...     executor=DaftExecutor(runner="local")
        ... )
        >>> pipeline = pipeline.with_config(config)
        >>> result = pipeline.run(inputs={"x": 5})
        >>> # result == {"result": 6}
    """
    
    def __init__(
        self,
        runner: str = "local",
        ray_address: Optional[str] = None,
        collect: bool = True,
        show_plan: bool = False,
        debug: bool = False,
    ):
        if not DAFT_AVAILABLE:
            raise ImportError(
                "Daft is not installed. Install it with: pip install daft"
            )
        
        self.runner = runner
        self.ray_address = ray_address
        self.collect = collect
        self.show_plan = show_plan
        self.debug = debug
        
        # Validate runner
        if runner not in ("local", "ray"):
            raise ValueError(f"Invalid runner: {runner}. Must be 'local' or 'ray'")
        
        # If ray runner, ensure ray_address is provided or use default
        if runner == "ray" and ray_address is None:
            # Use default Ray connection
            self.ray_address = None  # Daft will connect to existing Ray cluster
    
    def run_pipeline(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute a pipeline using Daft DataFrame operations.
        
        Converts the pipeline to Daft operations, executes lazily,
        and returns materialized results.
        
        Args:
            pipeline: The pipeline to execute
            inputs: Input dictionary for root arguments
            ctx: Callback context for tracking execution
            output_name: Optional output name(s) to compute
            
        Returns:
            Dictionary of pipeline outputs
        """
        # This is the existing DaftBackend.run() logic
        # Keep all the UDF conversion, DataFrame building, etc.
        # (All the existing code from daft_backend.py)
        
        # Import the existing run method logic here
        # (I'm not copying the full 700+ lines, but it would be the same)
        
        # Set runner before execution
        if self.runner == "ray":
            if self.ray_address:
                daft.context.set_runner_ray(address=self.ray_address)
            else:
                daft.context.set_runner_ray()
        else:
            daft.context.set_runner_py()
        
        # ... existing Daft execution logic ...
        # (Convert nodes to UDFs, build DataFrame, execute, collect)
        
        pass  # Placeholder - actual implementation from existing code
    
    def map_items(
        self,
        pipeline: "Pipeline",
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute pipeline over multiple items using Daft.
        
        Creates a DataFrame with all items and executes the pipeline
        over all rows in parallel using Daft's distributed execution.
        
        Args:
            pipeline: The pipeline to execute
            items: List of input dictionaries (one per item)
            inputs: Shared inputs for all items
            ctx: Callback context for tracking execution
            output_name: Optional output name(s) to compute
            
        Returns:
            List of output dictionaries (one per item)
        """
        # This is the existing DaftBackend.map() logic
        # Keep all the DataFrame-based map execution
        
        # Set runner before execution
        if self.runner == "ray":
            if self.ray_address:
                daft.context.set_runner_ray(address=self.ray_address)
            else:
                daft.context.set_runner_ray()
        else:
            daft.context.set_runner_py()
        
        # ... existing Daft map logic ...
        # (Create DataFrame from items, apply pipeline as UDFs, collect)
        
        pass  # Placeholder - actual implementation from existing code
```

### 2. Create DaskExecutor Stub

```python
"""Dask executor for distributed pipeline execution."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .executor import Executor
from .callbacks import CallbackContext

if TYPE_CHECKING:
    from .pipeline import Pipeline


class DaskExecutor(Executor):
    """Dask execution executor for distributed computing.
    
    Uses Dask Delayed or Dask Futures to execute pipelines on
    a Dask cluster or local scheduler.
    
    Args:
        scheduler_address: Dask scheduler address (optional)
        scheduler: Local scheduler type ("threads" or "processes")
    
    Example:
        >>> # Local execution with threads
        >>> executor = DaskExecutor(scheduler="threads")
        >>> 
        >>> # Remote cluster
        >>> executor = DaskExecutor(scheduler_address="tcp://scheduler:8786")
    """
    
    def __init__(
        self,
        scheduler_address: Optional[str] = None,
        scheduler: str = "threads",
    ):
        self.scheduler_address = scheduler_address
        self.scheduler = scheduler
        
        if scheduler not in ("threads", "processes", "synchronous"):
            raise ValueError(f"Invalid scheduler: {scheduler}")
    
    def run_pipeline(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute pipeline using Dask Delayed."""
        raise NotImplementedError(
            "DaskExecutor is not yet implemented. "
            "Use ThreadPoolExecutor or DaftExecutor instead."
        )
    
    def map_items(
        self,
        pipeline: "Pipeline",
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute map using Dask Bag or Delayed."""
        raise NotImplementedError(
            "DaskExecutor is not yet implemented. "
            "Use ThreadPoolExecutor or DaftExecutor instead."
        )
```

### 3. Create RayExecutor Stub

```python
"""Ray executor for distributed pipeline execution."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .executor import Executor
from .callbacks import CallbackContext

if TYPE_CHECKING:
    from .pipeline import Pipeline


class RayExecutor(Executor):
    """Ray execution executor for distributed computing.
    
    Uses Ray's remote functions and actors to execute pipelines
    on a Ray cluster.
    
    Args:
        ray_address: Ray cluster address (optional, uses existing connection if None)
        num_cpus: CPUs per task (optional)
        num_gpus: GPUs per task (optional)
    
    Example:
        >>> # Local execution
        >>> executor = RayExecutor()
        >>> 
        >>> # Remote cluster
        >>> executor = RayExecutor(ray_address="ray://cluster:10001")
    """
    
    def __init__(
        self,
        ray_address: Optional[str] = None,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
    ):
        self.ray_address = ray_address
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
    
    def run_pipeline(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute pipeline using Ray remote functions."""
        raise NotImplementedError(
            "RayExecutor is not yet implemented. "
            "Use ThreadPoolExecutor or DaftExecutor instead."
        )
    
    def map_items(
        self,
        pipeline: "Pipeline",
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute map using Ray remote functions."""
        raise NotImplementedError(
            "RayExecutor is not yet implemented. "
            "Use ThreadPoolExecutor or DaftExecutor instead."
        )
```

## Verification Tests

### Test File: `tests/test_phase3_daft_executor.py`

```python
"""Tests for Phase 3: DaftExecutor."""

import pytest
from hypernodes import node, Pipeline
from hypernodes.config import RuntimeConfig
from hypernodes.backend import LocalBackend

# Import DaftExecutor (skip tests if not installed)
try:
    from hypernodes.daft_executor import DaftExecutor
    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False
    DaftExecutor = None


@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2


@node(output_name="result")
def add_one(doubled: int) -> int:
    return doubled + 1


@node(output_name="squared")
def square(x: int) -> int:
    return x * x


@pytest.mark.skipif(not DAFT_AVAILABLE, reason="Daft not installed")
class TestDaftExecutor:
    """Test suite for DaftExecutor."""
    
    def test_daft_executor_creation(self):
        """Test creating DaftExecutor with different runners."""
        # Local runner
        executor = DaftExecutor(runner="local")
        assert executor.runner == "local"
        assert executor.collect is True
        
        # Ray runner
        executor = DaftExecutor(runner="ray", ray_address="ray://localhost:10001")
        assert executor.runner == "ray"
        assert executor.ray_address == "ray://localhost:10001"
    
    def test_daft_executor_invalid_runner(self):
        """Test DaftExecutor raises error for invalid runner."""
        with pytest.raises(ValueError):
            DaftExecutor(runner="invalid")
    
    def test_daft_executor_run_pipeline(self):
        """Test DaftExecutor executes pipeline correctly."""
        pipeline = Pipeline(nodes=[double, add_one])
        
        config = RuntimeConfig(
            backend=LocalBackend(),
            executor=DaftExecutor(runner="local"),
        )
        pipeline = pipeline.with_config(config)
        
        result = pipeline.run({"x": 5})
        
        assert result["doubled"] == 10
        assert result["result"] == 11
    
    def test_daft_executor_map_items(self):
        """Test DaftExecutor map operation."""
        pipeline = Pipeline(nodes=[double])
        
        executor = DaftExecutor(runner="local")
        
        items = [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}]
        results = executor.map_items(pipeline, items, {})
        
        assert len(results) == 4
        assert results[0]["doubled"] == 2
        assert results[1]["doubled"] == 4
        assert results[2]["doubled"] == 6
        assert results[3]["doubled"] == 8
    
    def test_daft_executor_with_config(self):
        """Test DaftExecutor integrated with RuntimeConfig."""
        pipeline = Pipeline(nodes=[square])
        
        config = RuntimeConfig(
            backend=LocalBackend(),
            executor=DaftExecutor(runner="local", show_plan=False),
        )
        pipeline = pipeline.with_config(config)
        
        result = pipeline.run({"x": 7})
        assert result["squared"] == 49
    
    def test_daft_executor_selective_output(self):
        """Test DaftExecutor with selective output computation."""
        pipeline = Pipeline(nodes=[double, add_one])
        
        config = RuntimeConfig(
            backend=LocalBackend(),
            executor=DaftExecutor(runner="local"),
        )
        pipeline = pipeline.with_config(config)
        
        # Only compute "doubled", not "result"
        result = pipeline.run({"x": 5}, output_name="doubled")
        
        assert "doubled" in result
        assert result["doubled"] == 10
        # "result" should not be computed
        assert "result" not in result
    
    def test_daft_executor_map_large_batch(self):
        """Test DaftExecutor with larger batch for parallelism."""
        pipeline = Pipeline(nodes=[square])
        
        executor = DaftExecutor(runner="local")
        
        # Create 100 items
        items = [{"x": i} for i in range(100)]
        results = executor.map_items(pipeline, items, {})
        
        assert len(results) == 100
        assert all(results[i]["squared"] == i * i for i in range(100))


def test_dask_executor_stub():
    """Test DaskExecutor stub raises NotImplementedError."""
    from hypernodes.dask_executor import DaskExecutor
    
    executor = DaskExecutor(scheduler="threads")
    pipeline = Pipeline(nodes=[double])
    
    with pytest.raises(NotImplementedError):
        executor.run_pipeline(pipeline, {"x": 5})
    
    with pytest.raises(NotImplementedError):
        executor.map_items(pipeline, [{"x": 1}], {})


def test_ray_executor_stub():
    """Test RayExecutor stub raises NotImplementedError."""
    from hypernodes.ray_executor import RayExecutor
    
    executor = RayExecutor()
    pipeline = Pipeline(nodes=[double])
    
    with pytest.raises(NotImplementedError):
        executor.run_pipeline(pipeline, {"x": 5})
    
    with pytest.raises(NotImplementedError):
        executor.map_items(pipeline, [{"x": 1}], {})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Migration Notes

### Breaking Changes

1. **Import path changed**:
   - **Before**: `from hypernodes import DaftBackend` or `from hypernodes.daft_backend import DaftBackend`
   - **After**: `from hypernodes.daft_executor import DaftExecutor`

2. **Class name changed**:
   - **Before**: `DaftBackend(collect=True)`
   - **After**: `DaftExecutor(runner="local", collect=True)`

3. **Usage pattern changed**:
   - **Before**: `Pipeline(nodes=[...], backend=DaftBackend())`
   - **After**: 
     ```python
     config = RuntimeConfig(
         backend=LocalBackend(),
         executor=DaftExecutor(runner="local")
     )
     Pipeline(nodes=[...]).with_config(config)
     ```

4. **Runner parameter added**:
   - New `runner` parameter to specify "local" or "ray"
   - Replaces implicit behavior with explicit configuration

### Examples to Update

- `examples/daft_backend_example.py` → update to use DaftExecutor
- `examples/daft_backend_complex_types_example.py` → update to use DaftExecutor
- Any documentation mentioning DaftBackend

## Success Criteria

- [ ] File renamed: `daft_backend.py` → `daft_executor.py`
- [ ] Class renamed: `DaftBackend` → `DaftExecutor`
- [ ] `DaftExecutor` implements `Executor` interface
- [ ] `run_pipeline()` method works correctly
- [ ] `map_items()` method works correctly
- [ ] All existing Daft functionality preserved
- [ ] `runner` parameter works for "local" and "ray"
- [ ] `DaskExecutor` stub created
- [ ] `RayExecutor` stub created
- [ ] All 8 Daft tests pass (when Daft is installed)
- [ ] Examples updated
- [ ] `__init__.py` updated to export DaftExecutor

## Dependencies

- Phase 1 complete (Executor interface)
- Phase 2 complete (LocalBackend refactored)
- Daft installed for testing (optional dependency)

## Next Phase

Phase 4 will implement CompositeExecutor and extract PipelineExecutionEngine.

