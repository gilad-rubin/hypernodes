"""Execution engines for HyperNodes pipelines.

This module provides a unified import location for all execution engines:
- Engine: Protocol for pipeline execution engines
- SeqEngine: Simple sequential execution (default)
- DaskEngine: Parallel execution using Dask Bag (optional)
- DaftEngine: Distributed DataFrame-based execution (optional)

Example:
    >>> from hypernodes import Pipeline
    >>> from hypernodes.engines import SeqEngine, DaskEngine, DaftEngine
    >>>
    >>> # Use SeqEngine (default - no need to specify)
    >>> pipeline = Pipeline(nodes=[...])
    >>>
    >>> # Or explicitly:
    >>> engine = SeqEngine()
    >>> pipeline = Pipeline(nodes=[...], engine=engine)
    >>>
    >>> # Use DaskEngine for parallel map operations
    >>> dask_engine = DaskEngine()  # Auto-optimized
    >>> pipeline = Pipeline(nodes=[...], engine=dask_engine)
    >>>
    >>> # Use DaftEngine for distributed execution
    >>> daft_engine = DaftEngine(collect=True)
    >>> pipeline = Pipeline(nodes=[...], engine=daft_engine)
"""

from .sequential_engine import SeqEngine

# Build __all__ dynamically
__all__ = [
    "SeqEngine",
]

# Optional engines
try:
    from .integrations.dask.engine import DaskEngine

    __all__.extend(["DaskEngine"])
except ImportError:
    pass

try:
    from .integrations.daft.engine import DaftEngine

    __all__.extend(["DaftEngine"])
except ImportError:
    pass
