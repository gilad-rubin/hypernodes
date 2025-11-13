"""Pure orchestration logic for pipeline execution.

This module contains the core orchestration responsibilities:
- Graph traversal and dependency resolution
- Node execution coordination (delegates to node_execution)
- Result collection and output filtering
- Map operation coordination

Key principle: The orchestrator doesn't know about executor TYPES,
it only uses the executor INTERFACE (submit() and result()).
"""

import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

from .callbacks import CallbackContext
from .executors import SequentialExecutor
from .node_execution import _get_node_id, execute_single_node, execute_single_node_async

if TYPE_CHECKING:
    from .pipeline import Pipeline


def _compute_topological_generations(
    nodes: List[Any], dependencies: Dict[Any, Set[Any]]
) -> List[List[Any]]:
    """Compute topological generations for parallel execution.

    Groups nodes into "generations" where each generation contains nodes that:
    - Have no dependencies on each other (can run in parallel)
    - Only depend on nodes from previous generations

    This is an independent implementation of NetworkX's topological_generations
    algorithm, using an in-degree based approach.

    Algorithm:
        1. Calculate in-degree (number of dependencies) for each node
        2. Nodes with in-degree 0 form generation 0 (no dependencies)
        3. Remove generation 0 nodes, decrement in-degrees of their dependents
        4. New nodes with in-degree 0 form generation 1
        5. Repeat until all nodes are assigned to generations

    Args:
        nodes: List of nodes to organize into generations
        dependencies: Dict mapping each node to its set of dependencies
                     (i.e., dependencies[consumer] = {producer1, producer2, ...})

    Returns:
        List of generations, where each generation is a list of nodes
        Example: [[a, b], [c], [d, e]]
                 Level 0: a,b can run in parallel
                 Level 1: c depends on a or b
                 Level 2: d,e depend on c, can run in parallel

    Raises:
        ValueError: If a cycle is detected in the dependencies
    """
    # Handle empty input
    if not nodes:
        return []

    # Calculate in-degree (number of dependencies) for each node
    in_degree: Dict[Any, int] = {node: len(dependencies.get(node, set())) for node in nodes}

    # Build reverse mapping: node -> nodes that depend on it
    dependents: Dict[Any, List[Any]] = defaultdict(list)
    for node, deps in dependencies.items():
        for dep in deps:
            dependents[dep].append(node)

    # Find nodes with no dependencies (in-degree = 0)
    queue: deque = deque([node for node in nodes if in_degree[node] == 0])

    # Generate levels
    generations: List[List[Any]] = []
    processed_count = 0

    while queue:
        # All nodes currently in queue form one generation (can run in parallel)
        current_generation = list(queue)
        generations.append(current_generation)
        processed_count += len(current_generation)

        # Clear queue for next generation
        queue.clear()

        # Process each node in current generation
        for node in current_generation:
            # Decrement in-degree for all nodes that depend on this node
            for dependent in dependents[node]:
                in_degree[dependent] -= 1

                # If dependent now has no remaining dependencies, add to next generation
                if in_degree[dependent] == 0:
                    queue.append(dependent)

    # Verify all nodes were processed (detect cycles)
    if processed_count != len(nodes):
        raise ValueError(
            f"Cycle detected in dependencies. Processed {processed_count} nodes "
            f"but expected {len(nodes)}. This indicates a circular dependency."
        )

    return generations


class PipelineOrchestrator:
    """Orchestrates pipeline execution with pluggable executors.

    This class is responsible for:
    1. Determining which nodes need to execute
    2. Deciding execution strategy (sequential vs parallel)
    3. Coordinating execution via the provided executor
    4. Collecting and filtering results

    It does NOT:
    - Create or manage executors (engine's job)
    - Execute nodes directly (node_execution's job)
    - Know about specific executor types (depends on interface only)
    """

    def execute(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        executor: Any,
        ctx: CallbackContext,
        output_name: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Execute a pipeline and return results.

        This is the main entry point for pipeline execution.

        Args:
            pipeline: Pipeline to execute
            inputs: Input values for the pipeline
            executor: Executor instance (SequentialExecutor, AsyncExecutor, etc.)
            ctx: Callback context for event tracking
            output_name: Optional filter for specific outputs

        Returns:
            Dictionary of output_name -> value
        """
        # Get callbacks from pipeline
        callbacks = pipeline.callbacks or []

        # Get node IDs for metadata
        node_ids = [_get_node_id(n) for n in pipeline.graph.execution_order]

        # Set pipeline metadata
        ctx.set_pipeline_metadata(
            pipeline.id,
            {
                "total_nodes": len(pipeline.graph.execution_order),
                "node_ids": node_ids,
                "pipeline_name": pipeline.name or pipeline.id,
            },
        )

        # Push this pipeline onto hierarchy stack
        ctx.push_pipeline(pipeline.id)

        # Compute required nodes based on output_name
        required_nodes = pipeline.graph.get_required_nodes(output_name)
        nodes_to_execute = (
            required_nodes if required_nodes is not None else pipeline.graph.execution_order
        )

        # Trigger pipeline start callbacks
        pipeline_start_time = time.time()
        for callback in callbacks:
            callback.on_pipeline_start(pipeline.id, inputs, ctx)

        # Initialize outputs dict
        outputs: Dict[str, Any] = {}

        try:
            # Choose execution strategy
            if self._is_sequential_executor(executor):
                outputs = self._execute_sequential(
                    pipeline, nodes_to_execute, inputs, callbacks, ctx
                )
            else:
                outputs = self._execute_with_parallelism(
                    pipeline, nodes_to_execute, inputs, executor, callbacks, ctx
                )
        finally:
            # Trigger pipeline end callbacks
            pipeline_duration = time.time() - pipeline_start_time
            for callback in callbacks:
                callback.on_pipeline_end(pipeline.id, outputs, pipeline_duration, ctx)

            # Pop pipeline from hierarchy stack
            ctx.pop_pipeline()

        # Filter outputs if output_name is specified
        return self._filter_outputs(outputs, output_name)

    async def execute_async(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: CallbackContext,
        output_name: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Async version of execute() for native async pipelines.

        Used when all nodes are async functions and we want true async execution.
        """
        callbacks = pipeline.callbacks or []
        node_ids = [_get_node_id(n) for n in pipeline.graph.execution_order]

        ctx.set_pipeline_metadata(
            pipeline.id,
            {
                "total_nodes": len(pipeline.graph.execution_order),
                "node_ids": node_ids,
                "pipeline_name": pipeline.name or pipeline.id,
            },
        )
        ctx.push_pipeline(pipeline.id)

        required_nodes = pipeline.graph.get_required_nodes(output_name)
        nodes_to_execute = (
            required_nodes if required_nodes is not None else pipeline.graph.execution_order
        )

        pipeline_start_time = time.time()
        for callback in callbacks:
            callback.on_pipeline_start(pipeline.id, inputs, ctx)

        # Initialize outputs dict
        outputs: Dict[str, Any] = {}

        try:
            outputs = await self._execute_sequential_async(
                pipeline, nodes_to_execute, inputs, callbacks, ctx
            )
        finally:
            pipeline_duration = time.time() - pipeline_start_time
            for callback in callbacks:
                callback.on_pipeline_end(pipeline.id, outputs, pipeline_duration, ctx)
            ctx.pop_pipeline()

        return self._filter_outputs(outputs, output_name)

    # ============================================
    # Node Computation & Filtering
    # ============================================

    def _filter_outputs(
        self,
        outputs: Dict[str, Any],
        output_name: Optional[Union[str, List[str]]],
    ) -> Dict[str, Any]:
        """Filter results to only include requested outputs.

        Args:
            outputs: All execution results
            output_name: Requested output(s) or None for all

        Returns:
            Filtered results dictionary
        """
        if output_name is None:
            return outputs

        # Normalize to list
        if isinstance(output_name, str):
            requested_names = [output_name]
        else:
            requested_names = list(output_name)

        # Filter to only requested outputs
        return {name: outputs[name] for name in requested_names if name in outputs}

    # ============================================
    # Sequential Execution
    # ============================================

    def _execute_sequential(
        self,
        pipeline: "Pipeline",
        nodes_to_execute: List,
        inputs: Dict[str, Any],
        callbacks: List,
        ctx: CallbackContext,
    ) -> Dict[str, Any]:
        """Execute nodes sequentially in order.

        Simple loop - no parallelism, no futures, no complexity.

        Returns:
            Dictionary of all outputs produced
        """
        available_values = dict(inputs)
        outputs = {}
        node_signatures = {}

        for node in nodes_to_execute:
            # Gather inputs for this node
            node_inputs = {
                param: available_values[param] for param in node.root_args
            }

            # Execute node (delegates to node_execution module)
            result, signature = execute_single_node(
                node, node_inputs, pipeline, callbacks, ctx, node_signatures
            )

            # Store results and signature
            self._store_node_outputs(node, result, outputs, available_values, node_signatures, signature)

        return outputs

    async def _execute_sequential_async(
        self,
        pipeline: "Pipeline",
        nodes_to_execute: List,
        inputs: Dict[str, Any],
        callbacks: List,
        ctx: CallbackContext,
    ) -> Dict[str, Any]:
        """Async version of sequential execution."""
        available_values = dict(inputs)
        outputs = {}
        node_signatures = {}

        for node in nodes_to_execute:
            node_inputs = {
                param: available_values[param] for param in node.root_args
            }

            result, signature = await execute_single_node_async(
                node, node_inputs, pipeline, callbacks, ctx, node_signatures, offload_sync=True
            )

            self._store_node_outputs(node, result, outputs, available_values, node_signatures, signature)

        return outputs

    # ============================================
    # Parallel Execution (Dependency Levels)
    # ============================================

    def _execute_with_parallelism(
        self,
        pipeline: "Pipeline",
        nodes_to_execute: List,
        inputs: Dict[str, Any],
        executor: Any,
        callbacks: List,
        ctx: CallbackContext,
    ) -> Dict[str, Any]:
        """Execute nodes with level-based parallelism.

        Uses dependency levels to execute independent nodes concurrently.
        """
        available_values = dict(inputs)
        outputs = {}
        node_signatures = {}

        # Determine execution strategy based on executor type
        executor_class_name = executor.__class__.__name__
        is_async = executor_class_name == "AsyncExecutor"
        is_threaded = isinstance(executor, ThreadPoolExecutor)

        # Use node-level parallelism for async and threaded executors
        use_node_parallelism = (is_async or is_threaded) and len(nodes_to_execute) > 1

        if use_node_parallelism:
            # Compute dependency levels (nodes that can run in parallel)
            levels = self._compute_dependency_levels(pipeline, nodes_to_execute)

            # Execute nodes level by level
            for level in levels:
                if len(level) == 1:
                    # Single node in this level - execute directly (no parallelism benefit)
                    node = level[0]
                    node_inputs = {
                        param: available_values[param] for param in node.root_args
                    }
                    result, signature = execute_single_node(
                        node, node_inputs, pipeline, callbacks, ctx, node_signatures
                    )
                    self._store_node_outputs(node, result, outputs, available_values, node_signatures, signature)
                else:
                    # Multiple independent nodes - submit in parallel
                    futures = {}
                    for node in level:
                        node_inputs = {
                            param: available_values[param] for param in node.root_args
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
                        futures[future] = node

                    # Wait for all nodes in this level to complete
                    for future in as_completed(futures.keys()):
                        node = futures[future]
                        result, signature = future.result()
                        self._store_node_outputs(node, result, outputs, available_values, node_signatures, signature)
        else:
            # Other executors - use greedy execution
            pending_nodes = set(nodes_to_execute)
            active_futures = {}  # Map future -> node

            while pending_nodes or active_futures:
                # Find ready nodes (all dependencies satisfied)
                ready_nodes = []
                for node in list(pending_nodes):
                    deps_satisfied = all(
                        param in available_values for param in node.root_args
                    )
                    if deps_satisfied:
                        ready_nodes.append(node)

                # Submit ready nodes to executor
                for node in ready_nodes:
                    pending_nodes.remove(node)
                    node_inputs = {
                        param: available_values[param] for param in node.root_args
                    }

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
                    for future in as_completed(active_futures.keys()):
                        node = active_futures.pop(future)
                        result, signature = future.result()
                        self._store_node_outputs(node, result, outputs, available_values, node_signatures, signature)
                        break  # Process one at a time to find new ready nodes

        return outputs

    def _compute_dependency_levels(
        self,
        pipeline: "Pipeline",
        nodes_to_execute: List,
    ) -> List[List]:
        """Compute dependency levels for parallel execution.

        Uses custom topological generations algorithm to find nodes
        that can run in parallel.

        Returns:
            List of levels, where each level is a list of nodes
            Example: [[a, b], [c], [d, e]]
        """
        # Build dependency mapping: node -> set of nodes it depends on
        dependencies: Dict[Any, Set[Any]] = {}

        for consumer in nodes_to_execute:
            deps = set()
            for param in consumer.root_args:
                # If this param is produced by another node in the same execution set
                producer = pipeline.graph.output_to_node.get(param)
                if producer is not None and producer in nodes_to_execute:
                    deps.add(producer)
            dependencies[consumer] = deps

        # Compute topological generations using custom implementation
        return _compute_topological_generations(nodes_to_execute, dependencies)

    # ============================================
    # Helper Methods
    # ============================================

    def _store_node_outputs(
        self,
        node,
        result: Any,
        outputs: Dict[str, Any],
        available_values: Dict[str, Any],
        node_signatures: Dict[str, str],
        signature: str,
    ) -> None:
        """Store node outputs in results dicts.

        Handles both single outputs and multiple outputs (PipelineNodes).
        """
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

    def _is_sequential_executor(self, executor: Any) -> bool:
        """Check if executor is sequential (no parallelism)."""
        return isinstance(executor, SequentialExecutor)


class MapOrchestrator:
    """Orchestrates map operations across multiple items.

    Responsibilities:
    - Prepare map items and transpose inputs
    - Execute pipeline for each item using map_executor
    - Transpose results back to lists
    - Handle map-specific callbacks
    """

    def __init__(self, pipeline_orchestrator: PipelineOrchestrator):
        """Initialize with a pipeline orchestrator for single-item execution."""
        self.pipeline_orchestrator = pipeline_orchestrator

    def execute_map(
        self,
        pipeline: "Pipeline",
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        executor: Any,
        ctx: CallbackContext,
        output_name: Optional[Union[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute pipeline across multiple items.

        Args:
            pipeline: Pipeline to execute
            items: List of input dicts (one per item)
            inputs: Shared inputs for all items
            executor: Executor for map items (map_executor)
            ctx: Callback context
            output_name: Optional output filter

        Returns:
            List of output dictionaries (one per item)
        """
        # Determine execution strategy
        is_sequential = isinstance(executor, SequentialExecutor)
        is_thread_pool = isinstance(executor, ThreadPoolExecutor)

        if is_sequential:
            # Sequential map: run items one by one with map context
            results = []
            for idx, item in enumerate(items):
                # Mark this execution as part of a map operation
                ctx.set("_in_map", True)
                ctx.set("_map_item_index", idx)
                ctx.set("_map_item_start_time", time.time())

                merged_inputs = {**inputs, **item}
                result = self.pipeline_orchestrator.execute(
                    pipeline, merged_inputs, SequentialExecutor(), ctx, output_name
                )
                results.append(result)

                # Clear per-item markers
                ctx.set("_map_item_start_time", None)

            # Unset the in-map flag
            ctx.set("_in_map", False)
            ctx.set("_map_item_index", None)
            return results

        # Parallel execution using executors
        if is_thread_pool:
            # Thread pools - simple parallel execution
            def run_item(item_inputs: Dict[str, Any]) -> Dict[str, Any]:
                merged_inputs = {**inputs, **item_inputs}
                item_ctx = CallbackContext()  # Fresh context per item
                return self.pipeline_orchestrator.execute(
                    pipeline, merged_inputs, SequentialExecutor(), item_ctx, output_name
                )

            futures = [executor.submit(run_item, item) for item in items]
            return [future.result() for future in futures]

        # For process-based or other executors
        # Submit items directly (pipeline is already pickled without engine)
        futures = []
        for item in items:
            merged_inputs = {**inputs, **item}
            future = executor.submit(
                _execute_pipeline_for_map_item,
                pipeline,
                merged_inputs,
                output_name,
            )
            futures.append(future)
        return [future.result() for future in futures]


# ============================================
# Module-level helper for parallel map
# ============================================

def _execute_pipeline_for_map_item(
    pipeline: "Pipeline",
    inputs: Dict[str, Any],
    output_name: Optional[Union[str, List[str]]],
) -> Dict[str, Any]:
    """Picklable function for parallel map execution.

    This must be at module level to be picklable for ProcessPoolExecutor.
    """
    orchestrator = PipelineOrchestrator()
    ctx = CallbackContext()

    # Execute pipeline for this item with sequential executor
    return orchestrator.execute(
        pipeline=pipeline,
        inputs=inputs,
        executor=SequentialExecutor(),
        ctx=ctx,
        output_name=output_name,
    )


def _pipeline_supports_async_native(pipeline: "Pipeline") -> bool:
    """Check if all nodes in the pipeline are native async functions.

    Used by engine to determine if async execution path should be used.
    """
    import inspect

    for node in pipeline.graph.execution_order:
        if hasattr(node, "pipeline"):
            # Nested pipelines introduce sync boundaries for now
            return False
        if not hasattr(node, "func"):
            return False
        if not inspect.iscoroutinefunction(node.func):
            return False
    return len(pipeline.graph.execution_order) > 0
