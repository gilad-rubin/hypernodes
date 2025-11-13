"""Protocol definitions for HyperNodes.

This module defines the core protocols (interfaces) used throughout the library.
"""

from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Protocol, Union

if TYPE_CHECKING:
    from .pipeline import Pipeline


class Executor(Protocol):
    """Protocol for executors (compatible with concurrent.futures.Executor).

    Custom executors must implement this interface:
    - submit(): Schedule a function for execution and return a Future
    - shutdown(): Cleanup resources

    This matches the concurrent.futures.Executor interface, so you can use:
    - ThreadPoolExecutor
    - ProcessPoolExecutor
    - AsyncExecutor (custom)
    - SequentialExecutor (custom)
    - Or any custom implementation
    """

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Schedule a callable to be executed and return a Future.

        Args:
            fn: Function to execute
            *args: Positional arguments for fn
            **kwargs: Keyword arguments for fn

        Returns:
            Future representing the execution
        """
        ...

    def shutdown(self, wait: bool = True) -> None:
        """Cleanup executor resources.

        Args:
            wait: If True, wait for pending tasks to complete
        """
        ...


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


class AsyncStrategy(Protocol):
    """Protocol for async execution strategies."""

    def should_use_async(
        self, pipeline: "Pipeline", node_executor: Any
    ) -> tuple[bool, str]:
        """Check if async execution should be used.

        Returns:
            Tuple of (should_use_async, runner_strategy)
        """
        ...
