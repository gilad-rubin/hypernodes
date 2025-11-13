"""Execution engines for HyperNodes pipelines.

This module provides a unified import location for all execution engines:
- Engine: Protocol for pipeline execution engines
- SequentialEngine: Simple sequential execution (default)
- DaftEngine: Distributed DataFrame-based execution (optional)

Example:
    >>> from hypernodes import Pipeline
    >>> from hypernodes.engines import SequentialEngine, DaftEngine
    >>>
    >>> # Use SequentialEngine (default - no need to specify)
    >>> pipeline = Pipeline(nodes=[...])
    >>>
    >>> # Or explicitly:
    >>> engine = SequentialEngine()
    >>> pipeline = Pipeline(nodes=[...], engine=engine)
    >>>
    >>> # Use DaftEngine for distributed execution
    >>> daft_engine = DaftEngine(collect=True)
    >>> pipeline = Pipeline(nodes=[...], engine=daft_engine)
"""

from .protocols import Engine
from .sequential_engine import SequentialEngine

# Build __all__ dynamically
__all__ = [
    "Engine",
    "SequentialEngine",
]

# Optional engines
try:
    from .integrations.daft.engine import DaftEngine

    __all__.extend(["DaftEngine"])
except ImportError:
    pass
