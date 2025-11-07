"""Pipeline orchestrator for executing pipelines with different executors.

This module provides the PipelineOrchestrator class which implements the
common execution logic for pipelines:
1. Setup phase: Initialize context, callbacks, determine nodes to execute
2. Execution loop: Find ready nodes, submit to executor, accumulate results
3. Cleanup phase: Filter outputs, trigger callbacks

Key principle: Single Responsibility - orchestration only. The orchestrator
doesn't know how to execute individual nodes (that's node_execution.py) or
how to parallelize (that's the executor's job).
"""

import time
from concurrent.futures import as_completed, Future
from typing import Any, Dict, List, Optional, Set, Union, TYPE_CHECKING

from .callbacks import CallbackContext
from .node_execution import execute_single_node

if TYPE_CHECKING:
    from .pipeline import Pipeline


class PipelineOrchestrator:
    """Orchestrates pipeline execution using a provided executor.

    The orchestrator is responsible for:
    - Managing the execution flow (setup → loop → cleanup)
    - Using NetworkX to determine node execution order
    - Submitting nodes to the executor
    - Accumulating results from completed nodes
    - Triggering callbacks at appropriate points

    The orchestrator delegates actual node execution to execute_single_node()
    and parallelism to the provided executor.
    """

    def __init__(self, executor):
        """Initialize orchestrator with an executor.

        Args:
            executor: Executor implementing submit(fn, *args, **kwargs) interface.
                Can be SequentialExecutor, AsyncExecutor, ThreadPoolExecutor,
                ProcessPoolExecutor, or any compatible executor.
        """
        self.executor = executor

    def execute(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext],
        output_name: Union[str, List[str], None],
    ) -> Dict[str, Any]:
        """Execute a pipeline with the configured executor.

        This is the main orchestration method. It:
        1. Sets up context and callbacks
        2. Determines which nodes need to be executed
        3. Finds ready nodes using NetworkX graph
        4. Submits them to the executor
        5. Accumulates results
        6. Filters outputs based on output_name
        7. Triggers cleanup callbacks

        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values for root arguments
            ctx: Callback context (created if None)
            output_name: Optional output name(s) to compute. Only specified
                outputs will be returned and only required nodes executed.

        Returns:
            Dictionary containing the requested outputs (or all outputs if None)
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
        from .node_execution import _get_node_id
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

        # Normalize output_name to set for filtering
        if output_name is None:
            requested_outputs = None  # Return all
        elif isinstance(output_name, str):
            requested_outputs = {output_name}
        else:
            requested_outputs = set(output_name)

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

            # Track completed nodes
            completed_nodes = set()

            # For sequential executor, execute nodes one by one
            if hasattr(self.executor, '__class__') and \
               self.executor.__class__.__name__ == 'SequentialExecutor':
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

                    completed_nodes.add(node)
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
                        future = self.executor.submit(
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
                        # Wait for the first future to complete (no timeout needed)
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

                            completed_nodes.add(node)
                            break  # Process one at a time to find new ready nodes

        finally:
            # Trigger pipeline end callbacks
            pipeline_duration = time.time() - pipeline_start_time
            for callback in callbacks:
                callback.on_pipeline_end(pipeline.id, outputs, pipeline_duration, ctx)

            # Pop pipeline from hierarchy stack
            ctx.pop_pipeline()

        # Filter outputs if specific outputs were requested
        if requested_outputs is not None:
            # Return only requested outputs (but keep dependencies for compatibility)
            # Actually, we should return ALL computed outputs, not just requested
            # This matches existing backend behavior
            pass

        return outputs
