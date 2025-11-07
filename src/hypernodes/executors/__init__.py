"""Pipeline execution engines.

This package contains engine implementations for running pipelines:
- Engine: Abstract base class for all engines
- HyperNodesEngine: Node-by-node execution with various parallelism strategies
- DaftEngine: Distributed DataFrame-based execution (optional, requires daft)

Engines are orchestrators that handle pipeline execution with different strategies.
They manage dependency resolution, caching, callbacks, and parallelism.
"""

from hypernodes.executors.base import Engine
from hypernodes.executors.local import HyperNodesEngine

# Build __all__ dynamically based on available dependencies
__all__ = ["Engine", "HyperNodesEngine"]

# Optional engines (may require additional dependencies)
try:
    from hypernodes.executors.daft import DaftEngine
    __all__.append("DaftEngine")
except ImportError:
    pass
