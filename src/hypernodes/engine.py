"""Execution engine for HyperNodes pipelines.

This module provides the Engine abstraction and HypernodesEngine implementation
which orchestrates pipeline execution using executors.

Key responsibilities:
- Resolve executor specifications (strings → instances)
- Orchestrate pipeline execution (node sequencing, dependency resolution)
- Manage map operations (prepare items, transpose results)
- Handle depth tracking for nested maps (future feature)
- Enable node-level parallelism for async and threaded executors

Design note on "parallel" executors:
- Node-level "parallel" (process-based) execution is intentionally disabled.
  Use node_executor="threaded" for intra-pipeline concurrency, and
  map_executor="parallel" to parallelize across map items. This avoids
  pickling/locking complexity with nested pipelines while still enabling
  robust parallel throughput for data-parallel workloads.
"""

import inspect
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import networkx as nx

from .callbacks import CallbackContext
from .executors import DEFAULT_WORKERS, AsyncExecutor, SequentialExecutor
from .node_execution import (
    _get_node_id,
    execute_single_node,
    execute_single_node_async,
)

# Import loky for robust parallel execution with cloudpickle support
try:
    from loky import get_reusable_executor
    _LOKY_AVAILABLE = True
except ImportError:
    _LOKY_AVAILABLE = False
    get_reusable_executor = None  # type: ignore

if TYPE_CHECKING:
    from .pipeline import Pipeline


def _compute_dependency_levels(pipeline: "Pipeline", nodes_to_execute: List) -> List[List]:
    """Compute levels of nodes that can execute in parallel using NetworkX.
    
    Uses NetworkX's topological_generations to identify groups of nodes that
    have no dependencies on each other and can run concurrently.
    
    Args:
        pipeline: The pipeline containing the dependency graph
        nodes_to_execute: List of nodes to execute (may be subset of all nodes)
    
    Returns:
        List of lists where each inner list contains nodes that can run in parallel.
        Example: [[node_a, node_b], [node_c], [node_d, node_e]]
                 Level 0: a,b have no deps between them (can run parallel)
                 Level 1: c depends on a or b (must wait for level 0)
                 Level 2: d,e depend on c (can run parallel after level 1)
    """
    # Build a subgraph where vertices are actual node objects (not output names)
    # This handles nodes that produce multiple outputs by keeping a single vertex
    # and connecting dependencies based on pipeline.output_to_node mapping.
    subgraph = nx.DiGraph()

    # Add all nodes as vertices
    for node in nodes_to_execute:
        subgraph.add_node(node)

    # Build dependency edges: producer_node -> consumer_node
    for consumer in nodes_to_execute:
        for param in consumer.parameters:
            # If this param is produced by another node in the same execution set
            producer = pipeline.output_to_node.get(param)
            if producer is not None and producer in nodes_to_execute:
                subgraph.add_edge(producer, consumer)

    # Compute topological generations (levels)
    levels = []
    for generation in nx.topological_generations(subgraph):
        # generation already contains node objects
        levels.append(list(generation))

    return levels


def _pipeline_supports_async_native(pipeline: "Pipeline") -> bool:
    """Check if all nodes in the pipeline are native async functions."""
    for node in pipeline.execution_order:
        if hasattr(node, "pipeline"):
            # Nested pipelines introduce sync boundaries for now
            return False
        if not hasattr(node, "func"):
            return False
        if not inspect.iscoroutinefunction(node.func):
            return False
    return len(pipeline.execution_order) > 0


def _execute_pipeline_for_map_item(
    pipeline: "Pipeline",
    inputs: Dict[str, Any],
    output_name: Union[str, List[str], None],
) -> Dict[str, Any]:
    """Execute a pipeline for a single map item (picklable standalone function).

    This function is designed to be pickled and sent to worker processes.
    It strips the backend from the pipeline before execution to avoid pickling
    locks and other unpicklable objects, then executes with a sequential executor.

    Args:
        pipeline: The pipeline to execute (will be copied without backend)
        inputs: Dictionary of input values
        output_name: Optional output name(s) to compute

    Returns:
        Dictionary containing the requested outputs
    """
    # Use sequential execution within each map item
    # (parallelization happens across map items, not within them)
    executor = SequentialExecutor()
    ctx = None  # Don't pass context across processes

    return _execute_pipeline_impl(pipeline, inputs, executor, ctx, output_name)


async def _execute_pipeline_for_map_item_async(
    pipeline: "Pipeline",
    inputs: Dict[str, Any],
    output_name: Union[str, List[str], None],
) -> Dict[str, Any]:
    """Async variant used when map executor natively handles coroutines."""
    ctx = None
    return await _execute_pipeline_impl_async(pipeline, inputs, ctx, output_name)


def _execute_pipeline_impl(
    pipeline: "Pipeline",
    inputs: Dict[str, Any],
    executor: Any,
    ctx: Optional[CallbackContext],
    output_name: Union[str, List[str], None],
) -> Dict[str, Any]:
    """Standalone implementation of pipeline execution.

    This is a module-level function (not a method) so it can be pickled
    for use with parallel executors like ProcessPoolExecutor/loky.

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

        # Determine execution strategy based on executor type
        executor_class_name = executor.__class__.__name__
        is_sequential = executor_class_name == 'SequentialExecutor'
        is_async = executor_class_name == 'AsyncExecutor'
        is_threaded = isinstance(executor, ThreadPoolExecutor)
        
        # Use node-level parallelism for async and threaded executors
        use_node_parallelism = (is_async or is_threaded) and len(nodes_to_execute) > 1
        
        if is_sequential:
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
                    
        elif use_node_parallelism:
            # Node-level parallelism using dependency levels
            # Compute dependency levels (nodes that can run in parallel)
            levels = _compute_dependency_levels(pipeline, nodes_to_execute)
            
            # Execute nodes level by level
            for level in levels:
                if len(level) == 1:
                    # Single node in this level - execute directly (no parallelism benefit)
                    node = level[0]
                    node_inputs = {
                        param: available_values[param] for param in node.parameters
                    }
                    result, signature = execute_single_node(
                        node, node_inputs, pipeline, callbacks, ctx, node_signatures
                    )
                    
                    # Store output and signature
                    if hasattr(node, "pipeline"):
                        outputs.update(result)
                        available_values.update(result)
                        for output_name_key in result.keys():
                            node_signatures[output_name_key] = signature
                    else:
                        outputs[node.output_name] = result
                        available_values[node.output_name] = result
                        node_signatures[node.output_name] = signature
                        
                else:
                    # Multiple independent nodes - submit in parallel
                    futures = {}
                    for node in level:
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
                        futures[future] = node
                    
                    # Wait for all nodes in this level to complete
                    for future in as_completed(futures.keys()):
                        node = futures[future]
                        result, signature = future.result()
                        
                        # Store output and signature
                        if hasattr(node, "pipeline"):
                            outputs.update(result)
                            available_values.update(result)
                            for output_name_key in result.keys():
                                node_signatures[output_name_key] = signature
                        else:
                            outputs[node.output_name] = result
                            available_values[node.output_name] = result
                            node_signatures[node.output_name] = signature
        else:
            # Other executors (e.g., ProcessPoolExecutor for node execution)
            # Use greedy execution - submit ready nodes as soon as possible
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


async def _execute_pipeline_impl_async(
    pipeline: "Pipeline",
    inputs: Dict[str, Any],
    ctx: Optional[CallbackContext],
    output_name: Union[str, List[str], None],
) -> Dict[str, Any]:
    """Async implementation used for native async pipelines."""
    if ctx is None:
        ctx = CallbackContext()

    callbacks = (
        pipeline.effective_callbacks
        if hasattr(pipeline, "effective_callbacks")
        else (pipeline.callbacks or [])
    )

    node_ids = [_get_node_id(n) for n in pipeline.execution_order]
    ctx.set_pipeline_metadata(
        pipeline.id,
        {
            "total_nodes": len(pipeline.execution_order),
            "node_ids": node_ids,
            "pipeline_name": pipeline.name or pipeline.id,
        },
    )
    ctx.push_pipeline(pipeline.id)

    required_nodes = pipeline._compute_required_nodes(output_name)
    nodes_to_execute = (
        required_nodes if required_nodes is not None else pipeline.execution_order
    )

    pipeline_start_time = time.time()
    for callback in callbacks:
        callback.on_pipeline_start(pipeline.id, inputs, ctx)

    try:
        available_values = dict(inputs)
        outputs: Dict[str, Any] = {}
        node_signatures: Dict[str, str] = {}

        for node in nodes_to_execute:
            node_inputs = {param: available_values[param] for param in node.parameters}
            result, signature = await execute_single_node_async(
                node,
                node_inputs,
                pipeline,
                callbacks,
                ctx,
                node_signatures,
                offload_sync=True,
            )

            if hasattr(node, "pipeline"):
                outputs.update(result)
                available_values.update(result)
                for output_name_key in result.keys():
                    node_signatures[output_name_key] = signature
            else:
                outputs[node.output_name] = result
                available_values[node.output_name] = result
                node_signatures[node.output_name] = signature

    finally:
        pipeline_duration = time.time() - pipeline_start_time
        for callback in callbacks:
            callback.on_pipeline_end(pipeline.id, outputs, pipeline_duration, ctx)
        ctx.pop_pipeline()

    if output_name is not None:
        if isinstance(output_name, str):
            requested_names = [output_name]
        else:
            requested_names = list(output_name)
        return {name: outputs[name] for name in requested_names if name in outputs}

    return outputs


class Engine(ABC):
    """Abstract base class for pipeline engines.

    Engines implement strategies for executing pipelines using different executors.
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

    This engine executes pipelines node-by-node and supports different
    execution strategies via configurable executors.

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
        async_strategy: How to await async nodes when called from sync contexts.
            - "per_call": status quo (new event loop per await)
            - "thread_local": reuse thread-local loop
            - "async_native": prefer async pipelines end-to-end
            - "auto": hybrid detection (thread_local fallback)
    """

    def __init__(
        self,
        node_executor: Union[str, Any] = "sequential",
        map_executor: Union[str, Any] = "sequential",
        max_workers: Optional[int] = None,
        async_strategy: str = "auto",
        loky_timeout: Optional[int] = 1200,
    ):
        self.max_workers = max_workers or os.cpu_count() or 4
        if async_strategy not in ("per_call", "thread_local", "async_native", "auto"):
            raise ValueError(
                "async_strategy must be one of "
                "('per_call', 'thread_local', 'async_native', 'auto')"
            )
        self.async_strategy = async_strategy
        self.loky_timeout = loky_timeout

        # Store original specs to track ownership
        self._node_executor_spec = node_executor
        self._map_executor_spec = map_executor
        
        # Track if we're using reusable executors (shouldn't be shut down)
        self._node_executor_is_reusable = False
        self._map_executor_is_reusable = False

        # Resolve executors
        self.node_executor, self._node_executor_is_reusable = self._resolve_executor(node_executor, role="node")
        self.map_executor, self._map_executor_is_reusable = self._resolve_executor(map_executor, role="map")

    def _resolve_executor(self, executor_spec: Union[str, Any], role: str) -> tuple[Any, bool]:
        """Resolve an executor specification to an executor instance.

        Args:
            executor_spec: Either a string ("sequential", "async", "threaded",
                "parallel") or an executor instance
            role: "node" when resolving node_executor, "map" for map_executor

        Returns:
            Tuple of (executor instance, is_reusable)
            - is_reusable: True if executor is from loky's reusable pool (shouldn't be shut down)

        Raises:
            ValueError: If string spec is invalid
        """
        if isinstance(executor_spec, str):
            # Create executor from string spec
            if executor_spec == "sequential":
                return SequentialExecutor(), False
            elif executor_spec == "async":
                return AsyncExecutor(max_workers=DEFAULT_WORKERS["async"]), False
            elif executor_spec == "threaded":
                workers = self.max_workers if hasattr(self, 'max_workers') else DEFAULT_WORKERS["threaded"]
                return ThreadPoolExecutor(max_workers=workers), False
            elif executor_spec == "parallel":
                if role == "node":
                    # Node-level parallel is intentionally disabled: it introduces
                    # pickling and resource-management complexity with nested pipelines.
                    # Prefer node_executor="threaded" for intra-pipeline concurrency and
                    # map_executor="parallel" for cross-item parallelism.
                    raise ValueError(
                        "Node-level 'parallel' is disabled. Use node_executor='threaded' "
                        "for node concurrency or map_executor='parallel' for parallel map."
                    )
                workers = self.max_workers if hasattr(self, 'max_workers') else DEFAULT_WORKERS["parallel"]
                # Use loky for robust parallel execution with automatic cloudpickle support
                if _LOKY_AVAILABLE:
                    try:
                        return get_reusable_executor(max_workers=workers, timeout=self.loky_timeout), True  # type: ignore
                    except TypeError:
                        return get_reusable_executor(max_workers=workers), True  # type: ignore
                else:
                    return ProcessPoolExecutor(max_workers=workers), False
            else:
                raise ValueError(
                    f"Invalid executor spec: {executor_spec}. "
                    f"Must be 'sequential', 'async', 'threaded', or 'parallel'"
                )
        else:
            # User provided an executor instance - we don't own it, don't shut it down
            return executor_spec, False

    def _execute_pipeline(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        executor: Any,
        ctx: Optional[CallbackContext],
        output_name: Union[str, List[str], None],
    ) -> Dict[str, Any]:
        """Internal orchestration logic for executing a pipeline.

        Delegates to the module-level _execute_pipeline_impl function.

        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values for root arguments
            executor: The executor to use for node execution
            ctx: Callback context (created if None)
            output_name: Optional output name(s) to compute

        Returns:
            Dictionary containing the requested outputs
        """
        return _execute_pipeline_impl(pipeline, inputs, executor, ctx, output_name)

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
        is_sequential = isinstance(self.map_executor, SequentialExecutor)
        is_thread_pool = isinstance(self.map_executor, ThreadPoolExecutor)

        if is_sequential:
            # Sequential map: run items one by one while annotating context
            # so per-item callbacks (start/end/cached) can fire correctly.
            results = []
            # Create a context if none provided (should be provided by Pipeline.map)
            ctx = _ctx if _ctx is not None else CallbackContext()
            for idx, item in enumerate(items):
                # Mark this execution as part of a map operation
                ctx.set("_in_map", True)
                ctx.set("_map_item_index", idx)
                ctx.set("_map_item_start_time", time.time())

                merged_inputs = {**inputs, **item}
                result = self._execute_pipeline(
                    pipeline, merged_inputs, self.node_executor, ctx, output_name
                )
                results.append(result)

                # Clear per-item markers to avoid leaking state
                ctx.set("_map_item_start_time", None)
            # Unset the in-map flag after processing all items
            ctx.set("_in_map", False)
            ctx.set("_map_item_index", None)
            return results

        use_async_native = (
            isinstance(self.map_executor, AsyncExecutor)
            and self.async_strategy in ("async_native", "auto")
            and _pipeline_supports_async_native(pipeline)
        )

        if use_async_native:
            futures = []

            async def run_item_async(item_inputs: Dict[str, Any]) -> Dict[str, Any]:
                merged_inputs = {**inputs, **item_inputs}
                return await _execute_pipeline_for_map_item_async(
                    pipeline,
                    merged_inputs,
                    output_name,
                )

            for item in items:
                futures.append(self.map_executor.submit(run_item_async, item))

            return [future.result() for future in futures]

        # Thread pools can execute map items without detaching pipeline metadata
        if is_thread_pool:
            def run_item(item_inputs: Dict[str, Any]) -> Dict[str, Any]:
                merged_inputs = {**inputs, **item_inputs}
                return _execute_pipeline_impl(
                    pipeline,
                    merged_inputs,
                    SequentialExecutor(),
                    None,
                    output_name,
                )

            futures = [self.map_executor.submit(run_item, item) for item in items]
            return [future.result() for future in futures]

        # For process-based executors (or unknown ones that likely need pickling),
        # submit all items and collect results via the standalone helper.
        original_backend = getattr(pipeline, "backend", None)
        original_parent = getattr(pipeline, "_parent", None)
        effective_cache = getattr(pipeline, "effective_cache", None)
        effective_callbacks = getattr(pipeline, "effective_callbacks", [])

        pipeline.backend = None  # type: ignore[assignment]
        pipeline._parent = None  # type: ignore[attr-defined]
        if effective_cache is not None:
            pipeline.cache = effective_cache
        if effective_callbacks is not None:
            pipeline.callbacks = list(effective_callbacks)

        try:
            futures = []
            for item in items:
                merged_inputs = {**inputs, **item}
                future = self.map_executor.submit(
                    _execute_pipeline_for_map_item,
                    pipeline,
                    merged_inputs,
                    output_name,
                )
                futures.append(future)
            return [future.result() for future in futures]
        finally:
            pipeline.backend = original_backend
            pipeline._parent = original_parent  # type: ignore[attr-defined]

    def shutdown(self, wait: bool = True):
        """Shutdown executors that we own.

        Only shuts down executors that were created by this engine
        (from string specs), not user-provided instances or reusable executors.
        
        Note: Loky's get_reusable_executor() returns a singleton that should NOT
        be shut down as it's meant to be reused across multiple operations.

        Args:
            wait: If True, wait for pending tasks to complete
        """
        # Shutdown node executor if we created it and it's not reusable
        if isinstance(self._node_executor_spec, str) and not self._node_executor_is_reusable:
            if hasattr(self.node_executor, "shutdown"):
                self.node_executor.shutdown(wait=wait)

        # Shutdown map executor if we created it and it's not reusable
        if isinstance(self._map_executor_spec, str) and not self._map_executor_is_reusable:
            if hasattr(self.map_executor, "shutdown"):
                self.map_executor.shutdown(wait=wait)
