"""Execution engine for HyperNodes pipelines.

This module provides the Engine abstraction and HypernodesEngine implementation.

Responsibilities:
- Resolve executor specifications (strings → instances)
- Manage executor lifecycle (creation, reuse, shutdown)
- Delegate to orchestrator for actual execution
- Public API entry points (run, map)
- Callback context lifecycle management

Design note: This is a thin facade. All orchestration logic lives in orchestrator.py.
"""

import os
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Union,
)

from .callbacks import CallbackContext
from .engine_orchestrator import (
    MapOrchestrator,
    PipelineOrchestrator,
    _pipeline_supports_async_native,
)
from .execution_context import get_callback_context, set_callback_context
from .executors import DEFAULT_WORKERS, AsyncExecutor, SequentialExecutor
from .map_planner import MapPlanner

# Import loky for robust parallel execution with cloudpickle support
try:
    from loky import get_reusable_executor

    _LOKY_AVAILABLE = True
except ImportError:
    _LOKY_AVAILABLE = False
    get_reusable_executor = None  # type: ignore

if TYPE_CHECKING:
    from .pipeline import Pipeline


# ============================================================================
# Protocol Definitions
# ============================================================================


class Executor(Protocol):
    """Protocol for executors (compatible with concurrent.futures.Executor).

    Custom executors must implement this interface:
    - submit(): Schedule a function for execution and return a Future
    - shutdown(): Cleanup resources

    This matches the concurrent.futures.Executor interface, so you can use:
    - ThreadPoolExecutor
    - ProcessPoolExecutor
    - AsyncExecutor (custom)
    - SequentialExecutor (custom)
    - Or any custom implementation
    """

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Schedule a callable to be executed and return a Future.

        Args:
            fn: Function to execute
            *args: Positional arguments for fn
            **kwargs: Keyword arguments for fn

        Returns:
            Future representing the execution
        """
        ...

    def shutdown(self, wait: bool = True) -> None:
        """Cleanup executor resources.

        Args:
            wait: If True, wait for pending tasks to complete
        """
        ...


class Engine(Protocol):
    """Protocol for pipeline execution engines.

    Engines implement strategies for executing pipelines using different executors.
    """

    def run(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute a pipeline with the given inputs.

        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values for root arguments
            output_name: Optional output name(s) to compute

        Returns:
            Dictionary containing the requested pipeline outputs
        """
        ...

    def map(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        map_over: List[str],
        map_mode: str,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, List[Any]]:
        """Execute a pipeline over multiple items.

        Args:
            pipeline: The pipeline to execute
            inputs: Input dictionary containing both varying and fixed parameters
            map_over: List of parameter names to map over
            map_mode: "zip" or "product"
            output_name: Optional output name(s) to compute

        Returns:
            Dictionary mapping output names to lists of values
        """
        ...


# ============================================================================
# Helper Classes
# ============================================================================


@dataclass
class ExecutorInfo:
    """Container for executor with ownership metadata.

    Attributes:
        executor: The executor instance implementing the Executor protocol
        is_reusable: True if executor is from a reusable pool (e.g., loky)
        spec: Original specification (string or executor instance)
    """

    executor: Executor
    is_reusable: bool
    spec: Union[str, Executor]


# ============================================================================
# Context Management
# ============================================================================


@contextmanager
def managed_callback_context():
    """Context manager for callback context lifecycle.

    Creates a new context if none exists, cleans up on exit.
    Yields existing context if already present.
    """
    ctx = get_callback_context()
    context_created_here = ctx is None

    if context_created_here:
        ctx = CallbackContext()
        set_callback_context(ctx)

    try:
        yield ctx
    finally:
        if context_created_here:
            set_callback_context(None)


# ============================================================================
# Executor Factory (Strategy Pattern)
# ============================================================================


def _create_sequential_executor(max_workers: int) -> Any:
    """Create a sequential executor."""
    return SequentialExecutor()


def _create_async_executor(max_workers: int) -> Any:
    """Create an async executor."""
    return AsyncExecutor(max_workers=DEFAULT_WORKERS["async"])


def _create_threaded_executor(max_workers: int) -> Any:
    """Create a threaded executor."""
    return ThreadPoolExecutor(max_workers=max_workers)


def _create_parallel_executor(
    max_workers: int, loky_timeout: Optional[int]
) -> tuple[Any, bool]:
    """Create a parallel executor (loky or ProcessPoolExecutor).

    Returns:
        Tuple of (executor, is_reusable)
    """
    if _LOKY_AVAILABLE and get_reusable_executor is not None:
        try:
            if loky_timeout is not None:
                return (
                    get_reusable_executor(
                        max_workers=max_workers, timeout=loky_timeout
                    ),
                    True,
                )
            else:
                return get_reusable_executor(max_workers=max_workers), True
        except TypeError:
            # Fallback: older loky versions without timeout
            return get_reusable_executor(max_workers=max_workers), True
    else:
        return ProcessPoolExecutor(max_workers=max_workers), False


# Executor factory lookup table
_EXECUTOR_FACTORIES: Dict[str, Callable[[int], Any]] = {
    "sequential": _create_sequential_executor,
    "async": _create_async_executor,
    "threaded": _create_threaded_executor,
}


def create_executor(
    spec: Union[str, Executor],
    max_workers: int,
    role: str,
    loky_timeout: Optional[int] = None,
) -> ExecutorInfo:
    """Create an executor from a specification.

    Args:
        spec: Either a string ("sequential", "async", "threaded", "parallel")
            or a custom executor instance implementing the Executor protocol
        max_workers: Maximum workers for parallel executors
        role: "node" or "map" - for validation
        loky_timeout: Timeout for loky reusable executor (seconds)

    Returns:
        ExecutorInfo with executor instance and metadata

    Raises:
        ValueError: If spec is invalid or parallel not allowed for nodes
    """
    # User-provided executor instance
    if not isinstance(spec, str):
        return ExecutorInfo(executor=spec, is_reusable=False, spec=spec)

    # Validate node-level parallel
    if spec == "parallel" and role == "node":
        raise ValueError(
            "Node-level 'parallel' is disabled. Use node_executor='threaded' "
            "for node concurrency or map_executor='parallel' for parallel map."
        )

    # Handle parallel separately (returns is_reusable)
    if spec == "parallel":
        executor, is_reusable = _create_parallel_executor(max_workers, loky_timeout)
        return ExecutorInfo(executor=executor, is_reusable=is_reusable, spec=spec)

    # Use factory lookup
    factory = _EXECUTOR_FACTORIES.get(spec)
    if factory is None:
        raise ValueError(
            f"Invalid executor spec: {spec}. "
            f"Must be 'sequential', 'async', 'threaded', or 'parallel'"
        )

    executor = factory(max_workers)
    return ExecutorInfo(executor=executor, is_reusable=False, spec=spec)


# ============================================================================
# Async Strategy (Dependency Injection)
# ============================================================================


class AsyncStrategy(Protocol):
    """Protocol for async execution strategies."""

    def should_use_async(
        self, pipeline: "Pipeline", node_executor: Any
    ) -> tuple[bool, str]:
        """Check if async execution should be used.

        Returns:
            Tuple of (should_use_async, runner_strategy)
        """
        ...


class PerCallAsyncStrategy:
    """Per-call async strategy (status quo)."""

    def should_use_async(
        self, pipeline: "Pipeline", node_executor: Any
    ) -> tuple[bool, str]:
        """Never use native async path."""
        return False, "per_call"


class ThreadLocalAsyncStrategy:
    """Thread-local async strategy."""

    def should_use_async(
        self, pipeline: "Pipeline", node_executor: Any
    ) -> tuple[bool, str]:
        """Never use native async path (thread-local handling in sync context)."""
        return False, "thread_local"


class AsyncNativeStrategy:
    """Async-native strategy (end-to-end async when possible)."""

    def should_use_async(
        self, pipeline: "Pipeline", node_executor: Any
    ) -> tuple[bool, str]:
        """Use async if executor is AsyncExecutor and pipeline supports it."""
        if not isinstance(node_executor, AsyncExecutor):
            return False, "thread_local"

        if _pipeline_supports_async_native(pipeline):
            return True, "thread_local"

        return False, "thread_local"


class AutoAsyncStrategy:
    """Auto async strategy (hybrid detection)."""

    def should_use_async(
        self, pipeline: "Pipeline", node_executor: Any
    ) -> tuple[bool, str]:
        """Use async if executor is AsyncExecutor and pipeline supports it."""
        if not isinstance(node_executor, AsyncExecutor):
            return False, "thread_local"

        if _pipeline_supports_async_native(pipeline):
            return True, "thread_local"

        return False, "thread_local"


def create_async_strategy(strategy_name: str) -> AsyncStrategy:
    """Factory for async strategies.

    Args:
        strategy_name: One of "per_call", "thread_local", "async_native", "auto"

    Returns:
        AsyncStrategy instance

    Raises:
        ValueError: If strategy_name is invalid
    """
    strategies = {
        "per_call": PerCallAsyncStrategy(),
        "thread_local": ThreadLocalAsyncStrategy(),
        "async_native": AsyncNativeStrategy(),
        "auto": AutoAsyncStrategy(),
    }

    strategy = strategies.get(strategy_name)
    if strategy is None:
        raise ValueError(
            f"Invalid async_strategy: {strategy_name}. "
            f"Must be one of {list(strategies.keys())}"
        )

    return strategy


# ============================================================================
# HyperNodes Engine
# ============================================================================


class HypernodesEngine:
    """HyperNodes native execution engine with configurable executors.

    This is a thin facade that:
    1. Manages executor instances (creation, reuse, lifecycle)
    2. Delegates orchestration to PipelineOrchestrator
    3. Manages callback context lifecycle

    Args:
        node_executor: Executor for running nodes within a pipeline.
            Can be:
            - "sequential": SequentialExecutor (default)
            - "async": AsyncExecutor
            - "threaded": ThreadPoolExecutor
            - Or a custom Executor instance implementing the protocol
        map_executor: Executor for running map operations.
            Same options as node_executor, plus:
            - "parallel": ProcessPoolExecutor (loky if available)
            Defaults to "sequential".
        max_workers: Maximum workers for parallel executors.
            Defaults to CPU count.
        async_strategy: How to await async nodes when called from sync contexts.
            - "per_call": status quo (new event loop per await)
            - "thread_local": reuse thread-local loop
            - "async_native": prefer async pipelines end-to-end
            - "auto": hybrid detection (thread_local fallback)
        loky_timeout: Timeout for loky reusable executor (seconds)

    Example with custom executor:
        ```python
        from concurrent.futures import ThreadPoolExecutor

        # Use standard library executor
        engine = HypernodesEngine(
            node_executor=ThreadPoolExecutor(max_workers=8)
        )

        # Or create a custom executor implementing the Executor protocol
        class CustomExecutor:
            def submit(self, fn, *args, **kwargs):
                # Custom execution logic
                ...
                return future

            def shutdown(self, wait=True):
                # Cleanup logic
                ...

        engine = HypernodesEngine(node_executor=CustomExecutor())
        ```
    """

    def __init__(
        self,
        node_executor: Union[
            Literal["sequential", "async", "threaded"], Executor
        ] = "sequential",
        map_executor: Union[
            Literal["sequential", "async", "threaded", "parallel"], Executor
        ] = "sequential",
        max_workers: Optional[int] = None,
        async_strategy: str = "auto",
        loky_timeout: Optional[int] = 1200,
    ):
        self.max_workers = max_workers or os.cpu_count() or 4

        # Create async strategy (dependency injection)
        self.async_strategy = create_async_strategy(async_strategy)

        # Create executors
        self._node_executor_info = create_executor(
            node_executor, self.max_workers, role="node", loky_timeout=loky_timeout
        )
        self._map_executor_info = create_executor(
            map_executor, self.max_workers, role="map", loky_timeout=loky_timeout
        )

        # Create orchestrators and planner (stateless, reusable)
        self.pipeline_orchestrator = PipelineOrchestrator()
        self.map_orchestrator = MapOrchestrator(self.pipeline_orchestrator)
        self.map_planner = MapPlanner()

    @property
    def node_executor(self) -> Any:
        """Get the node executor instance."""
        return self._node_executor_info.executor

    @property
    def map_executor(self) -> Any:
        """Get the map executor instance."""
        return self._map_executor_info.executor

    def run(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute a pipeline with the configured node executor.

        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values for root arguments
            output_name: Optional output name(s) to compute

        Returns:
            Dictionary containing the requested pipeline outputs
        """
        with managed_callback_context() as ctx:
            should_use_async, runner_strategy = self.async_strategy.should_use_async(
                pipeline, self.node_executor
            )

            if should_use_async:
                return self._run_async(
                    pipeline, inputs, ctx, output_name, runner_strategy
                )
            else:
                return self.pipeline_orchestrator.execute(
                    pipeline, inputs, self.node_executor, ctx, output_name
                )

    def _run_async(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: CallbackContext,
        output_name: Union[str, List[str], None],
        runner_strategy: str,
    ) -> Dict[str, Any]:
        """Execute pipeline using async orchestrator."""
        from .async_utils import run_coroutine_sync

        return run_coroutine_sync(
            self.pipeline_orchestrator.execute_async(
                pipeline, inputs, ctx, output_name
            ),
            strategy=runner_strategy,
        )

    def map(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        map_over: List[str],
        map_mode: str,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, List[Any]]:
        """Execute pipeline across multiple items.

        Args:
            pipeline: The pipeline to execute
            inputs: Input dictionary containing both varying and fixed parameters
            map_over: List of parameter names to map over
            map_mode: "zip" or "product"
            output_name: Optional output name(s) to compute

        Returns:
            Dictionary mapping output names to lists of values
        """
        with managed_callback_context() as ctx:
            # Use MapPlanner to convert inputs into items
            items = self.map_planner.plan_execution(inputs, map_over, map_mode)

            # Separate fixed inputs (not in map_over)
            fixed_inputs = {k: v for k, v in inputs.items() if k not in map_over}

            # Execute map operation
            results = self.map_orchestrator.execute_map(
                pipeline, items, fixed_inputs, self.map_executor, ctx, output_name
            )

            # Transpose results from List[Dict] to Dict[List]
            return self._transpose_results(results)

    def _transpose_results(self, results: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Transpose results from List[Dict] to Dict[List].

        Args:
            results: List of output dictionaries (one per item)

        Returns:
            Dictionary mapping output names to lists of values
        """
        if not results:
            return {}

        # Get all output keys from first result
        output_keys = list(results[0].keys())

        # Transpose: collect all values for each key
        return {key: [result[key] for result in results] for key in output_keys}

    def shutdown(self, wait: bool = True):
        """Shutdown executors that we own.

        Only shuts down executors that were created by this engine
        (from string specs), not user-provided instances or reusable executors.

        Args:
            wait: If True, wait for pending tasks to complete
        """
        self._shutdown_executor(self._node_executor_info, wait)
        self._shutdown_executor(self._map_executor_info, wait)

    def _shutdown_executor(self, info: ExecutorInfo, wait: bool):
        """Shutdown a single executor if we own it."""
        # Only shutdown if we created it from string spec and it's not reusable
        if isinstance(info.spec, str) and not info.is_reusable:
            if hasattr(info.executor, "shutdown"):
                info.executor.shutdown(wait=wait)
