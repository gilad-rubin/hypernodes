"""Execution engine for HyperNodes pipelines.

This module provides the Engine abstraction and HypernodesEngine implementation
which orchestrates pipeline execution using executors and orchestrators.

Key responsibilities:
- Resolve executor specifications (strings → instances)
- Create orchestrators for node and map execution
- Manage map operations (prepare items, transpose results)
- Handle depth tracking for nested maps (future feature)
"""

import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from .executor_adapters import SequentialExecutor, AsyncExecutor, DEFAULT_WORKERS
from .orchestrator import PipelineOrchestrator

if TYPE_CHECKING:
    from .pipeline import Pipeline
    from .callbacks import CallbackContext


class Engine(ABC):
    """Abstract base class for pipeline engines.

    Engines are orchestrators that implement strategies for executing pipelines.
    All engines must implement run() and map() methods.
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
            output_name: Optional output name(s) to compute
            _ctx: Internal callback context

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
            output_name: Optional output name(s) to compute
            _ctx: Internal callback context

        Returns:
            List of output dictionaries (one per item)
        """
        pass


class HypernodesEngine(Engine):
    """HyperNodes native execution engine with configurable executors.

    This engine executes pipelines node-by-node using orchestrators and
    supports different execution strategies via executors.

    Args:
        node_executor: Executor for running nodes within a pipeline.
            Can be:
            - "sequential": SequentialExecutor (default)
            - "async": AsyncExecutor
            - "threaded": ThreadPoolExecutor
            - "parallel": ProcessPoolExecutor
            - Or a custom executor instance
        map_executor: Executor for running map operations.
            Same options as node_executor. Defaults to "sequential".
        max_workers: Maximum workers for parallel executors.
            Defaults to CPU count.
    """

    def __init__(
        self,
        node_executor: Union[str, Any] = "sequential",
        map_executor: Union[str, Any] = "sequential",
        max_workers: Optional[int] = None,
    ):
        self.max_workers = max_workers or os.cpu_count() or 4

        # Store original specs to track ownership
        self._node_executor_spec = node_executor
        self._map_executor_spec = map_executor

        # Resolve executors
        self.node_executor = self._resolve_executor(node_executor)
        self.map_executor = self._resolve_executor(map_executor)

    def _resolve_executor(self, executor_spec: Union[str, Any]) -> Any:
        """Resolve an executor specification to an executor instance.

        Args:
            executor_spec: Either a string ("sequential", "async", "threaded",
                "parallel") or an executor instance

        Returns:
            Executor instance

        Raises:
            ValueError: If string spec is invalid
        """
        if isinstance(executor_spec, str):
            # Create executor from string spec
            if executor_spec == "sequential":
                return SequentialExecutor()
            elif executor_spec == "async":
                return AsyncExecutor(max_workers=DEFAULT_WORKERS["async"])
            elif executor_spec == "threaded":
                workers = self.max_workers if hasattr(self, 'max_workers') else DEFAULT_WORKERS["threaded"]
                return ThreadPoolExecutor(max_workers=workers)
            elif executor_spec == "parallel":
                workers = self.max_workers if hasattr(self, 'max_workers') else DEFAULT_WORKERS["parallel"]
                return ProcessPoolExecutor(max_workers=workers)
            else:
                raise ValueError(
                    f"Invalid executor spec: {executor_spec}. "
                    f"Must be 'sequential', 'async', 'threaded', or 'parallel'"
                )
        else:
            # User provided an executor instance
            return executor_spec

    def run(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        output_name: Union[str, List[str], None] = None,
        _ctx: Optional["CallbackContext"] = None,
    ) -> Dict[str, Any]:
        """Execute a pipeline with the configured node executor.

        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values for root arguments
            output_name: Optional output name(s) to compute
            _ctx: Internal callback context

        Returns:
            Dictionary containing the requested pipeline outputs
        """
        # Create orchestrator with node executor
        orchestrator = PipelineOrchestrator(self.node_executor)

        # Execute pipeline
        return orchestrator.execute(pipeline, inputs, _ctx, output_name)

    def map(
        self,
        pipeline: "Pipeline",
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        output_name: Union[str, List[str], None] = None,
        _ctx: Optional["CallbackContext"] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a pipeline over multiple items using map executor.

        Args:
            pipeline: The pipeline to execute
            items: List of input dictionaries (one per item)
            inputs: Shared inputs for all items
            output_name: Optional output name(s) to compute
            _ctx: Internal callback context

        Returns:
            List of output dictionaries (one per item)
        """
        # Create orchestrator with map executor
        orchestrator = PipelineOrchestrator(self.map_executor)

        # Execute each item
        results = []
        for item in items:
            # Merge shared inputs with item-specific inputs
            merged_inputs = {**inputs, **item}
            result = orchestrator.execute(pipeline, merged_inputs, _ctx, output_name)
            results.append(result)

        return results

    def shutdown(self, wait: bool = True):
        """Shutdown executors that we own.

        Only shuts down executors that were created by this engine
        (from string specs), not user-provided instances.

        Args:
            wait: If True, wait for pending tasks to complete
        """
        # Shutdown node executor if we created it
        if isinstance(self._node_executor_spec, str):
            if hasattr(self.node_executor, "shutdown"):
                self.node_executor.shutdown(wait=wait)

        # Shutdown map executor if we created it
        if isinstance(self._map_executor_spec, str):
            if hasattr(self.map_executor, "shutdown"):
                self.map_executor.shutdown(wait=wait)
