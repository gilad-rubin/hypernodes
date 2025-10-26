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

from .node import Node, node
from .pipeline import Pipeline
from .backend import LocalBackend, ModalBackend
from .cache import DiskCache
from .callbacks import PipelineCallback, CallbackContext
from .exceptions import (
    HyperNodesError,
    CycleError,
    DependencyError,
    ExecutionError,
)
from .visualization import (
    GraphvizStyle,
    DESIGN_STYLES,
    visualize,
    build_graph,
)

__version__ = "0.1.0"

__all__ = [
    # Decorators
    "node",
    # Classes
    "Node",
    "Pipeline",
    "LocalBackend",
    "DiskCache",
    "PipelineCallback",
    "CallbackContext",
    # Visualization
    "GraphvizStyle",
    "DESIGN_STYLES",
    "visualize",
    "build_graph",
    # Exceptions
    "HyperNodesError",
    "CycleError",
    "DependencyError",
    "ExecutionError",
    # Note: telemetry module is available but not exported at top level
    # Use: from hypernodes.telemetry import ProgressCallback, TelemetryCallback
]
