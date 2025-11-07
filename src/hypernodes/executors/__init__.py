"""Pipeline execution strategies.

This package contains executor implementations for running pipelines:
- Executor: Abstract base class
- LocalExecutor: Node-by-node execution with various parallelism strategies
- PipelineExecutionEngine: Reusable execution engine wrapper
- DaftExecutor: Distributed DataFrame-based execution
- ModalExecutor: Remote execution on Modal serverless infrastructure
"""

from hypernodes.executors.base import Executor
from hypernodes.executors.local import LocalExecutor, PipelineExecutionEngine

# Build __all__ dynamically based on available dependencies
__all__ = ["Executor", "LocalExecutor", "PipelineExecutionEngine"]

# Optional executors (may require additional dependencies)
try:
    from hypernodes.executors.daft import DaftExecutor
    __all__.append("DaftExecutor")
except ImportError:
    pass

try:
    from hypernodes.executors.modal import ModalExecutor
    __all__.append("ModalExecutor")
except ImportError:
    pass
