"""Execution engine for HyperNodes pipelines.

This module provides the HypernodesEngine - the default execution engine.

Responsibilities:
- Manage executor lifecycle (creation, reuse, shutdown)
- Delegate to orchestrator for actual execution
- Public API entry points (run, map)
"""

import os
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from .engine_orchestrator import MapOrchestrator, PipelineOrchestrator
from .engine_utils import managed_callback_context
from .executor_factory import create_executor
from .map_planner import MapPlanner
from .protocols import Executor

if TYPE_CHECKING:
    from .pipeline import Pipeline


class HypernodesEngine:
    """HyperNodes native execution engine with configurable executors.

    Args:
        node_executor: Executor for running nodes within a pipeline.
            Can be:
            - "sequential": SequentialExecutor (default)
            - "async": AsyncExecutor
            - "threaded": ThreadPoolExecutor
            - Or a custom Executor instance implementing the protocol
        map_executor: Executor for running map operations.
            Same options as node_executor, plus:
            - "parallel": ProcessPoolExecutor (loky if available)
            Defaults to "sequential".
        max_concurrency: Maximum concurrent tasks for parallel executors.
            Defaults to CPU count. Applies to async, threaded, and parallel executors.

    Example with custom executor:
        ```python
        from concurrent.futures import ThreadPoolExecutor

        # Use standard library executor
        engine = HypernodesEngine(
            node_executor=ThreadPoolExecutor(max_workers=8)
        )

        # Or create a custom executor implementing the Executor protocol
        class CustomExecutor:
            def submit(self, fn, *args, **kwargs):
                # Custom execution logic
                ...
                return future

            def shutdown(self, wait=True):
                # Cleanup logic
                ...

        engine = HypernodesEngine(node_executor=CustomExecutor())
        ```
    """

    def __init__(
        self,
        node_executor: Union[
            Literal["sequential", "async", "threaded"], Executor
        ] = "sequential",
        map_executor: Union[
            Literal["sequential", "async", "threaded", "parallel"], Executor
        ] = "sequential",
        max_concurrency: Optional[int] = None,
    ):
        self.max_concurrency = max_concurrency or os.cpu_count() or 4

        # Store executor specs (create fresh executors on each run)
        self._node_executor_spec = node_executor
        self._map_executor_spec = map_executor

        # Create orchestrators and planner (stateless, reusable)
        self._pipeline_orchestrator = PipelineOrchestrator()
        self._map_orchestrator = MapOrchestrator(self._pipeline_orchestrator)
        self._map_planner = MapPlanner()

    def run(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute a pipeline with the configured node executor.

        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values for root arguments
            output_name: Optional output name(s) to compute

        Returns:
            Dictionary containing the requested pipeline outputs
        """
        # Create fresh executor for this run
        executor_info = create_executor(
            self._node_executor_spec,
            self.max_concurrency,
            role="node",
            loky_timeout=1200,
        )

        with managed_callback_context() as ctx:
            result = self._pipeline_orchestrator.execute(
                pipeline, inputs, executor_info.executor, ctx, output_name
            )

        # Cleanup executor (skip reusable executors like loky)
        if not executor_info.is_reusable:
            executor_info.executor.shutdown(wait=True)

        return result

    def map(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        map_over: List[str],
        map_mode: str,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute pipeline across multiple items.

        Args:
            pipeline: The pipeline to execute
            inputs: Input dictionary containing both varying and fixed parameters
            map_over: List of parameter names to map over
            map_mode: "zip" or "product"
            output_name: Optional output name(s) to compute

        Returns:
            List of output dictionaries (one per item)
        """
        # Create fresh executor for this map operation
        executor_info = create_executor(
            self._map_executor_spec, self.max_concurrency, role="map", loky_timeout=1200
        )

        with managed_callback_context() as ctx:
            # Use MapPlanner to convert inputs into items
            items = self._map_planner.plan_execution(inputs, map_over, map_mode)

            # Separate fixed inputs (not in map_over)
            fixed_inputs = {k: v for k, v in inputs.items() if k not in map_over}

            # Execute map operation
            results = self._map_orchestrator.execute_map(
                pipeline, items, fixed_inputs, executor_info.executor, ctx, output_name
            )

        # Cleanup executor (skip reusable executors like loky)
        if not executor_info.is_reusable:
            executor_info.executor.shutdown(wait=True)

        return results
