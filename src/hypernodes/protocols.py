"""Protocol definitions for HyperNodes.

This module defines the core protocols (interfaces) used throughout the library.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Protocol, Union

if TYPE_CHECKING:
    from .pipeline import Pipeline


class Engine(Protocol):
    """Protocol for pipeline execution engines.

    Engines implement strategies for executing pipelines using different executors.
    """

    def run(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute a pipeline with the given inputs.

        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values for root arguments
            output_name: Optional output name(s) to compute

        Returns:
            Dictionary containing the requested pipeline outputs
        """
        ...

    def map(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        map_over: List[str],
        map_mode: str,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a pipeline over multiple items.

        Args:
            pipeline: The pipeline to execute
            inputs: Input dictionary containing both varying and fixed parameters
            map_over: List of parameter names to map over
            map_mode: "zip" or "product"
            output_name: Optional output name(s) to compute

        Returns:
            List of output dictionaries (one per item)
        """
        ...
