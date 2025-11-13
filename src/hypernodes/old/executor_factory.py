"""Executor factory for creating executors from specifications.

This module handles executor creation, including fallback logic for parallel execution.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union

from .executors import AsyncExecutor, SequentialExecutor
from .protocols import Executor

# Import loky for robust parallel execution with cloudpickle support
try:
    from loky import get_reusable_executor

    _LOKY_AVAILABLE = True
except ImportError:
    _LOKY_AVAILABLE = False
    get_reusable_executor = None  # type: ignore


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


def _create_sequential_executor(max_concurrency: int) -> Any:
    """Create a sequential executor."""
    return SequentialExecutor()


def _create_async_executor(max_concurrency: int) -> Any:
    """Create an async executor."""
    return AsyncExecutor(max_workers=max_concurrency)


def _create_threaded_executor(max_concurrency: int) -> Any:
    """Create a threaded executor."""
    return ThreadPoolExecutor(max_workers=max_concurrency)


def _create_parallel_executor(
    max_concurrency: int, loky_timeout: Optional[int]
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
                        max_workers=max_concurrency, timeout=loky_timeout
                    ),
                    True,
                )
            else:
                return get_reusable_executor(max_workers=max_concurrency), True
        except TypeError:
            # Fallback: older loky versions without timeout
            return get_reusable_executor(max_workers=max_concurrency), True
    else:
        return ProcessPoolExecutor(max_workers=max_concurrency), False


# Executor factory lookup table
_EXECUTOR_FACTORIES: Dict[str, Callable[[int], Any]] = {
    "sequential": _create_sequential_executor,
    "async": _create_async_executor,
    "threaded": _create_threaded_executor,
}


def create_executor(
    spec: Union[str, Executor],
    max_concurrency: int,
    role: str,
    loky_timeout: Optional[int] = None,
) -> ExecutorInfo:
    """Create an executor from a specification.

    Args:
        spec: Either a string ("sequential", "async", "threaded", "parallel")
            or a custom executor instance implementing the Executor protocol
        max_concurrency: Maximum concurrent tasks for parallel executors
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
        executor, is_reusable = _create_parallel_executor(max_concurrency, loky_timeout)
        return ExecutorInfo(executor=executor, is_reusable=is_reusable, spec=spec)

    # Use factory lookup
    factory = _EXECUTOR_FACTORIES.get(spec)
    if factory is None:
        raise ValueError(
            f"Invalid executor spec: {spec}. "
            f"Must be 'sequential', 'async', 'threaded', or 'parallel'"
        )

    executor = factory(max_concurrency)
    return ExecutorInfo(executor=executor, is_reusable=False, spec=spec)
