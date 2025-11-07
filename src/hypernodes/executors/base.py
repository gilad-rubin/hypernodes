"""Base classes for pipeline executors.

This module defines the abstract base class for all pipeline executors.
Executors are responsible for executing pipelines with different strategies:
- LocalExecutor: Node-by-node execution with various parallelism strategies
- FrameworkExecutor: Holistic executors (like Daft) that handle entire pipeline graph
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from hypernodes.pipeline import Pipeline
    from hypernodes.callbacks import CallbackContext


class Executor(ABC):
    """Abstract base class for pipeline executors.

    Executors implement strategies for executing pipelines. There are two main types:

    1. Node-level executors (like LocalExecutor):
       - Execute pipelines node-by-node
       - Handle different parallelism strategies (sequential/async/threaded/parallel)
       - Fire callbacks per node
       - Manage caching per node

    2. Framework executors (like DaftExecutor, ModalExecutor):
       - Execute entire pipeline holistically
       - Convert pipeline graph to framework-native representation
       - Handle nested pipelines and maps within framework semantics

    Both types implement the same interface for consistency.
    """

    @abstractmethod
    def run(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: Optional["CallbackContext"] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute a pipeline with the given inputs.

        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values for root arguments
            ctx: Optional callback context for lifecycle hooks
            output_name: Optional output name(s) to compute. Only specified
                outputs will be returned and only required nodes executed.

        Returns:
            Dictionary containing the requested pipeline outputs
        """
        pass

    @abstractmethod
    def map(
        self,
        pipeline: "Pipeline",
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        ctx: Optional["CallbackContext"] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a pipeline over multiple items.

        Args:
            pipeline: The pipeline to execute
            items: List of input dictionaries (one per item)
            inputs: Shared inputs for all items
            ctx: Optional callback context for lifecycle hooks
            output_name: Optional output name(s) to compute. Only specified
                outputs will be returned and only required nodes executed.

        Returns:
            List of output dictionaries (one per item)
        """
        pass
