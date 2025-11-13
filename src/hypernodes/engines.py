"""Execution engines for HyperNodes pipelines.

This module provides a unified import location for all execution engines:
- Engine: Protocol for pipeline execution engines
- Executor: Protocol for concurrent.futures-compatible executors
- HypernodesEngine: Node-by-node execution with various parallelism
- DaftEngine: Distributed DataFrame-based execution (optional)

Example:
    >>> from hypernodes.engines import HypernodesEngine, DaftEngine, Executor
    >>>
    >>> # Use HypernodesEngine for node-by-node execution
    >>> engine = HypernodesEngine(node_executor="threaded")
    >>>
    >>> # Use DaftEngine for distributed execution
    >>> daft_engine = DaftEngine(collect=True)
    >>>
    >>> # Create a custom executor
    >>> class CustomExecutor:
    ...     def submit(self, fn, *args, **kwargs):
    ...         # Custom logic
    ...         ...
    ...     def shutdown(self, wait=True):
    ...         ...
    >>> engine = HypernodesEngine(node_executor=CustomExecutor())
"""

from .engine import Engine, Executor, HypernodesEngine
from .executors import DEFAULT_WORKERS, AsyncExecutor, SequentialExecutor

# Build __all__ dynamically
__all__ = [
    "Engine",
    "Executor",
    "HypernodesEngine",
    "SequentialExecutor",
    "AsyncExecutor",
    "DEFAULT_WORKERS",
]

# Optional engines
try:
    from .integrations.daft.engine import DaftEngine, fix_script_classes_for_modal

    __all__.extend(["DaftEngine", "fix_script_classes_for_modal"])
except ImportError:
    pass
