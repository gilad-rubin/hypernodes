"""Execution engine for HyperNodes pipelines.

This module provides the Engine abstraction and HypernodesEngine implementation
which orchestrates pipeline execution using executors.

Key responsibilities:
- Resolve executor specifications (strings → instances)
- Orchestrate pipeline execution (node sequencing, dependency resolution)
- Manage map operations (prepare items, transpose results)
- Handle depth tracking for nested maps (future feature)
"""

import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from .executors import SequentialExecutor, AsyncExecutor, DEFAULT_WORKERS
from .callbacks import CallbackContext
from .node_execution import execute_single_node, _get_node_id

if TYPE_CHECKING:
    from .pipeline import Pipeline


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

    def _execute_pipeline(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        executor: Any,
        ctx: Optional[CallbackContext],
        output_name: Union[str, List[str], None],
    ) -> Dict[str, Any]:
        """Internal orchestration logic for executing a pipeline.

        This method implements the core orchestration:
        1. Setup phase: Initialize context, callbacks, determine nodes to execute
        2. Execution loop: Find ready nodes, submit to executor, accumulate results
        3. Cleanup phase: Filter outputs, trigger callbacks

        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values for root arguments
            executor: The executor to use for node execution
            ctx: Callback context (created if None)
            output_name: Optional output name(s) to compute

        Returns:
            Dictionary containing the requested outputs
        """
        # Create or reuse callback context
        if ctx is None:
            ctx = CallbackContext()

        # Determine callbacks to use (support inheritance via effective_callbacks)
        callbacks = (
            pipeline.effective_callbacks
            if hasattr(pipeline, "effective_callbacks")
            else (pipeline.callbacks or [])
        )

        # Get node IDs for metadata
        node_ids = [_get_node_id(n) for n in pipeline.execution_order]

        # Set pipeline metadata
        ctx.set_pipeline_metadata(
            pipeline.id,
            {
                "total_nodes": len(pipeline.execution_order),
                "node_ids": node_ids,
                "pipeline_name": pipeline.name or pipeline.id,
            },
        )

        # Push this pipeline onto hierarchy stack
        ctx.push_pipeline(pipeline.id)

        # Compute required nodes based on output_name
        required_nodes = pipeline._compute_required_nodes(output_name)
        nodes_to_execute = (
            required_nodes if required_nodes is not None else pipeline.execution_order
        )

        # Trigger pipeline start callbacks
        pipeline_start_time = time.time()
        for callback in callbacks:
            callback.on_pipeline_start(pipeline.id, inputs, ctx)

        try:
            # Start with provided inputs
            available_values = dict(inputs)

            # Track outputs separately (this is what we'll return)
            outputs = {}

            # Track node signatures for dependency hashing
            node_signatures = {}

            # For sequential executor, execute nodes one by one
            if hasattr(executor, '__class__') and \
               executor.__class__.__name__ == 'SequentialExecutor':
                # Sequential execution - simple loop
                for node in nodes_to_execute:
                    # Collect inputs for this node
                    node_inputs = {
                        param: available_values[param] for param in node.parameters
                    }

                    # Execute node
                    result, signature = execute_single_node(
                        node, node_inputs, pipeline, callbacks, ctx, node_signatures
                    )

                    # Store output and signature
                    if hasattr(node, "pipeline"):
                        # PipelineNode - result is dict of outputs
                        outputs.update(result)
                        available_values.update(result)
                        for output_name_key in result.keys():
                            node_signatures[output_name_key] = signature
                    else:
                        # Regular node - single output
                        outputs[node.output_name] = result
                        available_values[node.output_name] = result
                        node_signatures[node.output_name] = signature
            else:
                # Parallel execution - use executor with futures
                pending_nodes = set(nodes_to_execute)
                active_futures = {}  # Map future -> node

                while pending_nodes or active_futures:
                    # Find ready nodes (all dependencies satisfied)
                    ready_nodes = []
                    for node in list(pending_nodes):
                        # Check if all dependencies are available
                        deps_satisfied = all(
                            param in available_values for param in node.parameters
                        )
                        if deps_satisfied:
                            ready_nodes.append(node)

                    # Submit ready nodes to executor
                    for node in ready_nodes:
                        pending_nodes.remove(node)

                        # Collect inputs for this node
                        node_inputs = {
                            param: available_values[param] for param in node.parameters
                        }

                        # Submit to executor
                        future = executor.submit(
                            execute_single_node,
                            node,
                            node_inputs,
                            pipeline,
                            callbacks,
                            ctx,
                            node_signatures,
                        )
                        active_futures[future] = node

                    # Wait for at least one future to complete
                    if active_futures:
                        # Wait for the first future to complete
                        for future in as_completed(active_futures.keys()):
                            node = active_futures.pop(future)
                            result, signature = future.result()

                            # Store output and signature
                            if hasattr(node, "pipeline"):
                                # PipelineNode - result is dict of outputs
                                outputs.update(result)
                                available_values.update(result)
                                for output_name_key in result.keys():
                                    node_signatures[output_name_key] = signature
                            else:
                                # Regular node - single output
                                outputs[node.output_name] = result
                                available_values[node.output_name] = result
                                node_signatures[node.output_name] = signature

                            break  # Process one at a time to find new ready nodes

        finally:
            # Trigger pipeline end callbacks
            pipeline_duration = time.time() - pipeline_start_time
            for callback in callbacks:
                callback.on_pipeline_end(pipeline.id, outputs, pipeline_duration, ctx)

            # Pop pipeline from hierarchy stack
            ctx.pop_pipeline()

        # Filter outputs if output_name is specified
        if output_name is not None:
            # Normalize output_name to list
            if isinstance(output_name, str):
                requested_names = [output_name]
            else:
                requested_names = list(output_name)

            # Filter to only include requested outputs
            filtered_outputs = {name: outputs[name] for name in requested_names if name in outputs}
            return filtered_outputs

        return outputs

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
        return self._execute_pipeline(pipeline, inputs, self.node_executor, _ctx, output_name)

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
        # For sequential executor, process items one by one
        if hasattr(self.map_executor, '__class__') and \
           self.map_executor.__class__.__name__ == 'SequentialExecutor':
            results = []
            for item in items:
                merged_inputs = {**inputs, **item}
                result = self._execute_pipeline(pipeline, merged_inputs, self.node_executor, _ctx, output_name)
                results.append(result)
            return results

        # For parallel executors, submit all items and collect results
        futures = []
        for item in items:
            merged_inputs = {**inputs, **item}
            future = self.map_executor.submit(
                self._execute_pipeline,
                pipeline,
                merged_inputs,
                self.node_executor,
                _ctx,
                output_name
            )
            futures.append(future)

        # Collect results in order
        results = [future.result() for future in futures]
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
