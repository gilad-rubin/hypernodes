"""Base classes for pipeline engines.

This module defines the abstract base class for all pipeline engines.
Engines are orchestrators responsible for executing pipelines with different strategies:
- HyperNodesEngine: Node-by-node execution with various parallelism strategies
- DaftEngine: Framework-level execution using Daft DataFrames
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from hypernodes.pipeline import Pipeline
    from hypernodes.callbacks import CallbackContext


class Engine(ABC):
    """Abstract base class for pipeline engines.

    Engines are orchestrators that implement strategies for executing pipelines.
    There are two main types:

    1. Node-level engines (like HyperNodesEngine):
       - Execute pipelines node-by-node in topological order
       - Handle different parallelism strategies (sequential/async/threaded/parallel)
       - Fire callbacks per node
       - Manage caching per node

    2. Framework engines (like DaftEngine):
       - Execute entire pipeline holistically
       - Convert pipeline graph to framework-native representation (e.g., DataFrames)
       - Handle nested pipelines and maps within framework semantics

    Both types implement the same interface for consistency.
    """

    @abstractmethod
    def run(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        output_name: Union[str, List[str], None] = None,
        _ctx: Optional["CallbackContext"] = None,
    ) -> Dict[str, Any]:
        """Execute a pipeline with the given inputs.

        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values for root arguments
            output_name: Optional output name(s) to compute. Only specified
                outputs will be returned and only required nodes executed.
            _ctx: Internal callback context (not for public use)

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
        output_name: Union[str, List[str], None] = None,
        _ctx: Optional["CallbackContext"] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a pipeline over multiple items.

        Args:
            pipeline: The pipeline to execute
            items: List of input dictionaries (one per item)
            inputs: Shared inputs for all items
            output_name: Optional output name(s) to compute. Only specified
                outputs will be returned and only required nodes executed.
            _ctx: Internal callback context (not for public use)

        Returns:
            List of output dictionaries (one per item)
        """
        pass
