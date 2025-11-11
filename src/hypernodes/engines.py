"""Execution engines for HyperNodes pipelines.

This module provides a unified import location for all execution engines:
- Engine: Abstract base class
- HypernodesEngine: Node-by-node execution with various parallelism
- DaftEngine: Distributed DataFrame-based execution (optional)

Example:
    >>> from hypernodes.engines import HypernodesEngine, DaftEngine
    >>>
    >>> # Use HypernodesEngine for node-by-node execution
    >>> engine = HypernodesEngine(node_executor="threaded")
    >>>
    >>> # Use DaftEngine for distributed execution
    >>> daft_engine = DaftEngine(collect=True)
"""

from .engine import Engine, HypernodesEngine
from .executors import SequentialExecutor, AsyncExecutor, DEFAULT_WORKERS

# Build __all__ dynamically
__all__ = ["Engine", "HypernodesEngine", "SequentialExecutor", "AsyncExecutor", "DEFAULT_WORKERS"]

# Optional engines
try:
    from .integrations.daft.engine import DaftEngine, fix_script_classes_for_modal
    __all__.extend(["DaftEngine", "fix_script_classes_for_modal"])
except ImportError:
    pass
