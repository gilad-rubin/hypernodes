"""HyperNodes: Hierarchical, Modular Pipeline System.

A cache-first pipeline framework for ML/AI workflows that enables:
- Building pipelines from decorated functions
- Automatic dependency resolution
- Nested pipeline composition
- Intelligent caching (future)
- Backend-agnostic execution (local, remote, parallel)

Example:
    >>> from hypernodes import node, Pipeline
    >>>
    >>> @node(output_name="doubled")
    >>> def double(x: int) -> int:
    ...     return x * 2
    >>>
    >>> @node(output_name="result")
    >>> def add_one(doubled: int) -> int:
    ...     return doubled + 1
    >>>
    >>> pipeline = Pipeline(nodes=[double, add_one])
    >>> result = pipeline.run(inputs={"x": 5})
    >>> print(result)
    {'doubled': 10, 'result': 11}
"""

from .cache import DiskCache
from .callbacks import CallbackContext, PipelineCallback
from .decorators import stateful
from .dual_node import DualNode
from .engines import SeqEngine
from .exceptions import (
    CycleError,
    DependencyError,
    ExecutionError,
    HyperNodesError,
)
from .hypernode import HyperNode
from .node import Node, node
from .pipeline import Pipeline
from .visualization import (
    DESIGN_STYLES,
    GraphvizStyle,
    visualize,
)

# Optional: DaftEngine (requires daft to be installed)
try:
    from .engines import DaftEngine

    _DAFT_AVAILABLE = True
except ImportError:
    _DAFT_AVAILABLE = False

__version__ = "0.1.0"

__all__ = [
    # Decorators
    "node",
    "stateful",
    # Classes
    "DualNode",
    "HyperNode",
    "Node",
    "Pipeline",
    # Engines
    "SeqEngine",
    # Cache & Callbacks
    "DiskCache",
    "PipelineCallback",
    "CallbackContext",
    # Visualization
    "GraphvizStyle",
    "DESIGN_STYLES",
    "visualize",
    # Exceptions
    "HyperNodesError",
    "CycleError",
    "DependencyError",
    "ExecutionError",
    # Note: telemetry module is available but not exported at top level
    # Use: from hypernodes.telemetry import ProgressCallback, TelemetryCallback
    # Note: DaftEngine is available if daft is installed
    # Use: from hypernodes.engines import DaftEngine
]

if _DAFT_AVAILABLE:
    __all__.append("DaftEngine")
