"""Execution backends for running pipelines."""

import asyncio
import hashlib
import io
import os
import pickle
import time
from abc import ABC, abstractmethod
from concurrent.futures import (
    Executor,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set, Union

from .cache import compute_signature, hash_code, hash_inputs
from .callbacks import CallbackContext

if TYPE_CHECKING:
    from .pipeline import Pipeline


def _get_node_id(node) -> str:
    """Get a consistent node ID for callbacks and logging.

    Handles regular nodes, PipelineNodes, and nested pipelines.

    Args:
        node: Node object (Node, PipelineNode, or Pipeline)

    Returns:
        String identifier for the node
    """
    # PipelineNode with explicit name
    if hasattr(node, "name") and node.name:
        return node.name

    # Regular node with function name
    if hasattr(node, "func") and hasattr(node.func, "__name__"):
        return node.func.__name__

    # Pipeline or object with id
    if hasattr(node, "id"):
        return node.id

    # Object with __name__
    if hasattr(node, "__name__"):
        return node.__name__

    # Fallback to string representation
    return str(node)


def _get_node_dependencies(node, available_values: Set[str]) -> Set[str]:
    """Get the dependencies of a node based on its parameters.

    Args:
        node: Node object (Node, PipelineNode, or Pipeline)
        available_values: Set of currently available value names

    Returns:
        Set of parameter names that this node depends on
    """
    if hasattr(node, "parameters"):
        # Regular node or PipelineNode
        return set(node.parameters) & available_values
    return set()


class Backend(ABC):
    """Abstract base class for pipeline execution backends.

    All backends must implement the run() and map() methods to execute
    pipelines and map operations respectively.
    """

    @abstractmethod
    def run(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute a pipeline with the given inputs.

        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values for root arguments
            ctx: Optional callback context
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
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a pipeline over multiple items.

        Args:
            pipeline: The pipeline to execute
            items: List of input dictionaries (one per item)
            inputs: Shared inputs for all items
            ctx: Optional callback context
            output_name: Optional output name(s) to compute. Only specified
                outputs will be returned and only required nodes executed.

        Returns:
            List of output dictionaries (one per item)
        """
        pass


class PipelineExecutionEngine:
    """Reusable execution engine for pipelines.

    This engine encapsulates execution strategy (sequential/async/threaded/parallel)
    and mapping strategy, but delegates actual logic to LocalBackend to avoid code
    duplication. Backends (local or remote) can reuse this engine to run pipelines
    with identical semantics.
    """

    def __init__(
        self,
        node_execution: Literal[
            "sequential", "async", "threaded", "parallel"
        ] = "sequential",
        map_execution: Literal[
            "sequential", "async", "threaded", "parallel"
        ] = "sequential",
        max_workers: Optional[int] = None,
        executor: Optional[Union[Executor, Any]] = None,
    ):
        self.node_execution = node_execution
        self.map_execution = map_execution
        self.max_workers = max_workers
        self.executor = executor

        # Lazy constructed LocalBackend used to perform actual work
        self._local_backend: Optional["LocalBackend"] = None

    def _ensure_local(self) -> "LocalBackend":  # type: ignore[name-defined]
        if self._local_backend is None:
            # Import here to avoid circular references during module import
            # and to support environments where only the engine is serialized.
            self._local_backend = LocalBackend(
                node_execution=self.node_execution,
                map_execution=self.map_execution,
                max_workers=self.max_workers,
                executor=self.executor,
            )
        return self._local_backend

    def run(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        return self._ensure_local().run(pipeline, inputs, ctx, output_name=output_name)

    def map(
        self,
        pipeline: "Pipeline",
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        return self._ensure_local().map(pipeline, items, inputs, ctx, output_name=output_name)


class LocalBackend(Backend):
    """Local execution backend with multiple execution strategies.

    Supports four execution modes:
    - Sequential: Nodes execute one at a time (default)
    - Async: Independent nodes execute concurrently using asyncio
    - Threaded: Independent nodes execute in parallel using ThreadPoolExecutor
    - Parallel: Independent nodes execute in true parallel using ProcessPoolExecutor

    Also supports threaded/parallel/async execution for map operations.

    Args:
        node_execution: How nodes execute within a single pipeline.run()
            - "sequential": One node at a time (default)
            - "async": Concurrent execution for I/O-bound work
            - "threaded": Thread-based parallelism (good for I/O + some CPU)
            - "parallel": Process-based parallelism (true multi-core for CPU-bound)
        map_execution: How items are processed in pipeline.map()
            - "sequential": One item at a time (default)
            - "async": Concurrent items using asyncio
            - "threaded": Parallel items using threads
            - "parallel": Parallel items using processes
        max_workers: Maximum number of workers for parallel/threaded execution.
            Defaults to CPU count. Intelligently managed for nested maps.
        executor: Optional custom executor (ThreadPoolExecutor, ProcessPoolExecutor,
            or any Executor-compatible object). If provided, this overrides the
            node_execution/map_execution settings for parallel operations.
    """

    def __init__(
        self,
        node_execution: Literal[
            "sequential", "async", "threaded", "parallel"
        ] = "sequential",
        map_execution: Literal[
            "sequential", "async", "threaded", "parallel"
        ] = "sequential",
        max_workers: Optional[int] = None,
        executor: Optional[Union[Executor, Any]] = None,
    ):
        self.node_execution = node_execution
        self.map_execution = map_execution
        self.max_workers = max_workers or os.cpu_count() or 4
        self.executor = executor

    def run(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute a pipeline with the configured execution strategy.

        Dispatches to the appropriate execution method based on node_execution setting:
        - sequential: _run_sequential (one node at a time)
        - async: _run_async (concurrent I/O-bound)
        - parallel: _run_parallel (true parallelism)

        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values for root arguments
            ctx: Callback context (created if None)
            output_name: Optional output name(s) to compute. Only specified
                outputs will be returned and only required nodes executed.

        Returns:
            Dictionary containing only the requested outputs (or all if None)
        """
        # Dispatch based on node_execution mode
        if self.node_execution == "sequential":
            return self._run_sequential(pipeline, inputs, ctx, output_name=output_name)
        elif self.node_execution == "async":
            return asyncio.run(self._run_async(pipeline, inputs, ctx, output_name=output_name))
        elif self.node_execution == "threaded":
            return self._run_threaded(pipeline, inputs, ctx, output_name=output_name)
        elif self.node_execution == "parallel":
            return self._run_parallel(pipeline, inputs, ctx, output_name=output_name)
        else:
            raise ValueError(f"Invalid node_execution mode: {self.node_execution}")

    def _run_sequential(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute a pipeline sequentially with caching and callbacks support.

        Executes nodes in topological order, collecting outputs as they
        are produced. Supports nested pipelines by delegating to their
        own backends. Integrates with cache backend and callbacks if configured.

        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values for root arguments
            ctx: Callback context (created if None)
            output_name: Optional output name(s) to compute. Only specified
                outputs will be returned and only required nodes executed.

        Returns:
            Dictionary containing only the requested outputs (or all if None)
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

        # Set pipeline metadata
        # Get node IDs using consistent helper
        node_ids = [_get_node_id(n) for n in pipeline.execution_order]

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

            # Execute nodes in topological order (only required nodes)
            for node in nodes_to_execute:
                # Handle PipelineNode (wrapped pipelines from auto-wrapping)
                if hasattr(node, "pipeline"):  # PipelineNode
                    # PipelineNode has a __call__ method that handles mapping and map_over
                    # Just treat it like a regular node
                    node_inputs = {
                        param: available_values[param] for param in node.parameters
                    }

                    # Mark this as a PipelineNode in context so progress bar knows not to create node bar
                    node_id = _get_node_id(node)
                    ctx.set(f"_is_pipeline_node:{node_id}", True)

                    # Compute signature for PipelineNode caching
                    # Hash the inner pipeline structure (all node functions)
                    inner_pipeline = node.pipeline
                    inner_code_hashes = []
                    for inner_node in inner_pipeline.execution_order:
                        if hasattr(inner_node, "pipeline"):
                            # Nested PipelineNode - use pipeline ID to represent its structure
                            inner_code_hashes.append(inner_node.pipeline.id)
                        elif hasattr(inner_node, "func"):
                            # Regular node - hash its function
                            inner_code_hashes.append(hash_code(inner_node.func))
                    code_hash = hashlib.sha256(
                        "::".join(inner_code_hashes).encode()
                    ).hexdigest()

                    inputs_hash = hash_inputs(node_inputs)

                    # Compute dependencies hash from upstream node signatures
                    deps_signatures = []
                    for param in node.parameters:
                        if param in node_signatures:
                            deps_signatures.append(node_signatures[param])
                    deps_hash = ":".join(sorted(deps_signatures))

                    signature = compute_signature(
                        code_hash=code_hash,
                        inputs_hash=inputs_hash,
                        deps_hash=deps_hash,
                    )

                    # Check cache if enabled
                    effective_cache = (
                        pipeline.effective_cache
                        if hasattr(pipeline, "effective_cache")
                        else pipeline.cache
                    )
                    cache_enabled = effective_cache is not None and node.cache
                    result = None

                    if cache_enabled:
                        result = effective_cache.get(signature)

                        if result is not None:
                            # Cache hit - trigger callbacks
                            for callback in callbacks:
                                callback.on_node_cached(node_id, signature, ctx)

                    if result is None:
                        # Cache miss or caching disabled - execute PipelineNode
                        # Trigger node start callbacks (for progress bar updates)
                        node_start_time = time.time()
                        for callback in callbacks:
                            callback.on_node_start(node_id, node_inputs, ctx)

                        # Trigger nested pipeline start callbacks
                        nested_start_time = time.time()
                        for callback in callbacks:
                            callback.on_nested_pipeline_start(
                                pipeline.id, inner_pipeline.id, ctx
                            )

                        # Pass context to PipelineNode so it can share with nested pipeline
                        # Store temporarily as attribute (PipelineNode will use it if available)
                        node._exec_ctx = ctx

                        # Temporarily set parent so nested pipeline inherits callbacks/cache/backend
                        old_parent = inner_pipeline._parent
                        inner_pipeline._parent = pipeline

                        # Call the PipelineNode (it handles all mapping internally)
                        result = node(**node_inputs)

                        # Restore original parent and clean up
                        inner_pipeline._parent = old_parent
                        node._exec_ctx = None

                        # Trigger nested pipeline end callbacks
                        nested_duration = time.time() - nested_start_time
                        for callback in callbacks:
                            callback.on_nested_pipeline_end(
                                pipeline.id, inner_pipeline.id, nested_duration, ctx
                            )

                        # Trigger node end callbacks (for progress bar updates)
                        node_duration = time.time() - node_start_time
                        for callback in callbacks:
                            callback.on_node_end(
                                _get_node_id(node),
                                result,
                                node_duration,
                                ctx,
                            )

                        # Store in cache if enabled
                        if cache_enabled:
                            effective_cache.put(signature, result)

                    # Result is a dict of outputs
                    outputs.update(result)
                    available_values.update(result)
                    # Store signature for each output
                    for output_name in result.keys():
                        node_signatures[output_name] = signature
                else:
                    # Regular node execution with caching and callbacks
                    node_inputs = {
                        param: available_values[param] for param in node.parameters
                    }

                    # Compute node signature for caching
                    code_hash = hash_code(node.func)
                    inputs_hash = hash_inputs(node_inputs)

                    # Compute dependencies hash from upstream node signatures
                    deps_signatures = []
                    for param in node.parameters:
                        if param in node_signatures:
                            deps_signatures.append(node_signatures[param])
                    deps_hash = ":".join(sorted(deps_signatures))

                    signature = compute_signature(
                        code_hash=code_hash,
                        inputs_hash=inputs_hash,
                        deps_hash=deps_hash,
                    )

                    # Check cache if enabled and node allows caching
                    # Use effective_cache to support inheritance
                    effective_cache = (
                        pipeline.effective_cache
                        if hasattr(pipeline, "effective_cache")
                        else pipeline.cache
                    )
                    cache_enabled = effective_cache is not None and node.cache
                    result = None

                    if cache_enabled:
                        result = effective_cache.get(signature)

                        if result is not None:
                            # Cache hit - trigger callbacks
                            for callback in callbacks:
                                callback.on_node_cached(
                                    _get_node_id(node), signature, ctx
                                )

                            # If in a map operation, also trigger map item cached callback
                            if ctx.get("_in_map"):
                                item_index = ctx.get("_map_item_index")
                                for callback in callbacks:
                                    callback.on_map_item_cached(
                                        item_index, signature, ctx
                                    )

                    if result is None:
                        # Cache miss or caching disabled - execute node
                        # Trigger node start callbacks
                        node_start_time = time.time()
                        for callback in callbacks:
                            callback.on_node_start(_get_node_id(node), node_inputs, ctx)

                        # If in a map operation, also trigger map item start callback
                        if ctx.get("_in_map"):
                            item_index = ctx.get("_map_item_index")
                            for callback in callbacks:
                                callback.on_map_item_start(item_index, ctx)

                        try:
                            result = node(**node_inputs)

                            # Trigger node end callbacks
                            node_duration = time.time() - node_start_time
                            for callback in callbacks:
                                callback.on_node_end(
                                    _get_node_id(node),
                                    {node.output_name: result},
                                    node_duration,
                                    ctx,
                                )

                            # If in a map operation, also trigger map item end callback
                            if ctx.get("_in_map"):
                                item_index = ctx.get("_map_item_index")
                                map_item_start_time = ctx.get("_map_item_start_time")
                                map_item_duration = (
                                    time.time() - map_item_start_time
                                    if map_item_start_time
                                    else 0
                                )
                                for callback in callbacks:
                                    callback.on_map_item_end(
                                        item_index, map_item_duration, ctx
                                    )

                        except Exception as e:
                            # Trigger error callbacks
                            for callback in callbacks:
                                callback.on_error(_get_node_id(node), e, ctx)
                            raise

                        # Store in cache if enabled
                        if cache_enabled:
                            effective_cache.put(signature, result)

                    # Store output and signature
                    # Handle PipelineNode which returns dict of outputs
                    if isinstance(result, dict) and hasattr(node, "output_mapping"):
                        # PipelineNode - result is already a dict
                        outputs.update(result)
                        available_values.update(result)
                        # Store signature for each output
                        for output_name in result.keys():
                            node_signatures[output_name] = signature
                    else:
                        # Regular node - single output
                        outputs[node.output_name] = result
                        available_values[node.output_name] = result
                        node_signatures[node.output_name] = signature

            # Filter outputs if specific outputs were requested
            if requested_outputs is not None:
                outputs = {k: v for k, v in outputs.items() if k in requested_outputs}

            # Trigger pipeline end callbacks
            pipeline_duration = time.time() - pipeline_start_time
            for callback in callbacks:
                callback.on_pipeline_end(pipeline.id, outputs, pipeline_duration, ctx)

            return outputs

        finally:
            # Pop pipeline from hierarchy stack
            ctx.pop_pipeline()

    async def _run_async(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute a pipeline with async execution for independent nodes.

        Nodes that don't depend on each other execute concurrently using asyncio.
        Best for I/O-bound pipelines (API calls, file I/O, database queries).

        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values
            ctx: Callback context (created if None)
            output_name: Optional output name(s) to compute

        Returns:
            Dictionary containing only the requested outputs (or all if None)
        """
        # Create or reuse callback context
        if ctx is None:
            ctx = CallbackContext()

        # Determine callbacks to use
        callbacks = (
            pipeline.effective_callbacks
            if hasattr(pipeline, "effective_callbacks")
            else (pipeline.callbacks or [])
        )

        # Set pipeline metadata
        node_ids = [_get_node_id(n) for n in pipeline.execution_order]
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
            requested_outputs = None
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
            outputs = {}
            node_signatures = {}

            # Track pending nodes (only required ones)
            pending_nodes = set(nodes_to_execute)

            while pending_nodes:
                # Find nodes ready to execute (all deps satisfied)
                ready_nodes = []
                for node in list(pending_nodes):
                    # Check if all required parameters are available
                    if hasattr(node, "parameters"):
                        required_params = set(node.parameters)
                        if required_params.issubset(available_values.keys()):
                            ready_nodes.append(node)
                    else:
                        # No parameters required
                        ready_nodes.append(node)

                if not ready_nodes:
                    break  # No more nodes can execute (circular dependency or error)

                # Start tasks for ready nodes in this batch
                batch_tasks = []
                for node in ready_nodes:
                    # Pass a copy of available_values to avoid race conditions
                    task = asyncio.create_task(
                        self._execute_node_async(
                            node,
                            dict(available_values),
                            pipeline,
                            callbacks,
                            ctx,
                            node_signatures,
                        )
                    )
                    batch_tasks.append((task, node))
                    pending_nodes.remove(node)

                # Wait for ALL tasks in this batch to complete before starting next batch
                for task, node in batch_tasks:
                    result, signature = await task

                    # Store output and signature
                    if isinstance(result, dict) and hasattr(node, "output_mapping"):
                        # PipelineNode - result is already a dict
                        outputs.update(result)
                        available_values.update(result)
                        for output_name in result.keys():
                            node_signatures[output_name] = signature
                    else:
                        # Regular node - single output
                        outputs[node.output_name] = result
                        available_values[node.output_name] = result
                        node_signatures[node.output_name] = signature

            # Filter outputs if specific outputs were requested
            if requested_outputs is not None:
                outputs = {k: v for k, v in outputs.items() if k in requested_outputs}

            # Trigger pipeline end callbacks
            pipeline_duration = time.time() - pipeline_start_time
            for callback in callbacks:
                callback.on_pipeline_end(pipeline.id, outputs, pipeline_duration, ctx)

            return outputs

        finally:
            # Pop pipeline from hierarchy stack
            ctx.pop_pipeline()

    async def _execute_node_async(
        self,
        node,
        available_values: Dict[str, Any],
        pipeline: "Pipeline",
        callbacks: List,
        ctx: CallbackContext,
        node_signatures: Dict[str, str],
    ):
        """Execute a single node asynchronously.

        Handles both regular nodes and PipelineNodes with caching support.

        Returns:
            Tuple of (result, signature)
        """
        # Handle PipelineNode
        if hasattr(node, "pipeline"):
            node_inputs = {param: available_values[param] for param in node.parameters}

            # Mark this as a PipelineNode in context
            node_id = _get_node_id(node)
            ctx.set(f"_is_pipeline_node:{node_id}", True)

            # Compute signature for PipelineNode caching
            inner_pipeline = node.pipeline
            inner_code_hashes = []
            for inner_node in inner_pipeline.execution_order:
                if hasattr(inner_node, "pipeline"):
                    inner_code_hashes.append(inner_node.pipeline.id)
                elif hasattr(inner_node, "func"):
                    inner_code_hashes.append(hash_code(inner_node.func))
            code_hash = hashlib.sha256(
                "::".join(inner_code_hashes).encode()
            ).hexdigest()

            inputs_hash = hash_inputs(node_inputs)

            deps_signatures = []
            for param in node.parameters:
                if param in node_signatures:
                    deps_signatures.append(node_signatures[param])
            deps_hash = ":".join(sorted(deps_signatures))

            signature = compute_signature(
                code_hash=code_hash,
                inputs_hash=inputs_hash,
                deps_hash=deps_hash,
            )

            # Check cache
            effective_cache = (
                pipeline.effective_cache
                if hasattr(pipeline, "effective_cache")
                else pipeline.cache
            )
            cache_enabled = effective_cache is not None and node.cache
            result = None

            if cache_enabled:
                result = effective_cache.get(signature)
                if result is not None:
                    for callback in callbacks:
                        callback.on_node_cached(node_id, signature, ctx)

            if result is None:
                # Execute PipelineNode
                node_start_time = time.time()
                for callback in callbacks:
                    callback.on_node_start(node_id, node_inputs, ctx)

                nested_start_time = time.time()
                for callback in callbacks:
                    callback.on_nested_pipeline_start(
                        pipeline.id, inner_pipeline.id, ctx
                    )

                node._exec_ctx = ctx
                old_parent = inner_pipeline._parent
                inner_pipeline._parent = pipeline

                # Run synchronously (nested pipeline may have its own async execution)
                result = node(**node_inputs)

                inner_pipeline._parent = old_parent
                node._exec_ctx = None

                nested_duration = time.time() - nested_start_time
                for callback in callbacks:
                    callback.on_nested_pipeline_end(
                        pipeline.id, inner_pipeline.id, nested_duration, ctx
                    )

                node_duration = time.time() - node_start_time
                for callback in callbacks:
                    callback.on_node_end(
                        _get_node_id(node),
                        result,
                        node_duration,
                        ctx,
                    )

                if cache_enabled:
                    effective_cache.put(signature, result)

            return result, signature

        else:
            # Regular node execution
            node_inputs = {param: available_values[param] for param in node.parameters}

            # Compute node signature
            code_hash = hash_code(node.func)
            inputs_hash = hash_inputs(node_inputs)

            deps_signatures = []
            for param in node.parameters:
                if param in node_signatures:
                    deps_signatures.append(node_signatures[param])
            deps_hash = ":".join(sorted(deps_signatures))

            signature = compute_signature(
                code_hash=code_hash,
                inputs_hash=inputs_hash,
                deps_hash=deps_hash,
            )

            # Check cache
            effective_cache = (
                pipeline.effective_cache
                if hasattr(pipeline, "effective_cache")
                else pipeline.cache
            )
            cache_enabled = effective_cache is not None and node.cache
            result = None

            if cache_enabled:
                result = effective_cache.get(signature)
                if result is not None:
                    for callback in callbacks:
                        callback.on_node_cached(_get_node_id(node), signature, ctx)

            if result is None:
                # Execute node
                node_start_time = time.time()
                for callback in callbacks:
                    callback.on_node_start(_get_node_id(node), node_inputs, ctx)

                try:
                    # Run synchronously - asyncio handles concurrency at node level
                    result = node(**node_inputs)

                    node_duration = time.time() - node_start_time
                    for callback in callbacks:
                        callback.on_node_end(
                            _get_node_id(node),
                            {node.output_name: result},
                            node_duration,
                            ctx,
                        )

                except Exception as e:
                    for callback in callbacks:
                        callback.on_error(_get_node_id(node), e, ctx)
                    raise

                if cache_enabled:
                    effective_cache.put(signature, result)

            return result, signature

    def _run_threaded(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute a pipeline with threaded execution for independent nodes.

        Nodes that don't depend on each other execute in parallel using threads.
        Best for I/O-bound work with some CPU processing.

        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values
            ctx: Callback context (created if None)
            output_name: Optional output name(s) to compute

        Returns:
            Dictionary containing only the requested outputs (or all if None)
        """
        # Create or reuse callback context
        if ctx is None:
            ctx = CallbackContext()

        # Determine callbacks to use
        callbacks = (
            pipeline.effective_callbacks
            if hasattr(pipeline, "effective_callbacks")
            else (pipeline.callbacks or [])
        )

        # Set pipeline metadata
        node_ids = [_get_node_id(n) for n in pipeline.execution_order]
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
            requested_outputs = None
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
            outputs = {}
            node_signatures = {}

            # Track pending nodes (only required ones)
            pending_nodes = set(nodes_to_execute)

            # Use custom executor or create ThreadPoolExecutor
            executor = self.executor
            should_close = executor is None

            if executor is None:
                executor = ThreadPoolExecutor(max_workers=self.max_workers)

            try:
                while pending_nodes:
                    # Find nodes ready to execute (all deps satisfied)
                    ready_nodes = []
                    for node in list(pending_nodes):
                        # Check if all required parameters are available
                        if hasattr(node, "parameters"):
                            required_params = set(node.parameters)
                            if required_params.issubset(available_values.keys()):
                                ready_nodes.append(node)
                        else:
                            # No parameters required
                            ready_nodes.append(node)

                    if not ready_nodes:
                        break  # No more nodes can execute (circular dependency or error)

                    # Submit ready nodes to executor
                    future_to_node = {}
                    for node in ready_nodes:
                        # Pass a copy of available_values to avoid race conditions
                        future = executor.submit(
                            self._execute_node_sync,
                            node,
                            dict(available_values),
                            pipeline,
                            callbacks,
                            ctx,
                            node_signatures,
                        )
                        future_to_node[future] = node
                        pending_nodes.remove(node)

                    # Wait for all futures in this batch to complete
                    for future in as_completed(future_to_node):
                        node = future_to_node[future]
                        result, signature = future.result()

                        # Store output and signature
                        if isinstance(result, dict) and hasattr(node, "output_mapping"):
                            # PipelineNode - result is already a dict
                            outputs.update(result)
                            available_values.update(result)
                            for output_name in result.keys():
                                node_signatures[output_name] = signature
                        else:
                            # Regular node - single output
                            outputs[node.output_name] = result
                            available_values[node.output_name] = result
                            node_signatures[node.output_name] = signature

                # Filter outputs if specific outputs were requested
                if requested_outputs is not None:
                    outputs = {k: v for k, v in outputs.items() if k in requested_outputs}

                # Trigger pipeline end callbacks
                pipeline_duration = time.time() - pipeline_start_time
                for callback in callbacks:
                    callback.on_pipeline_end(
                        pipeline.id, outputs, pipeline_duration, ctx
                    )

                return outputs

            finally:
                # Close executor if we created it
                if should_close and hasattr(executor, "shutdown"):
                    executor.shutdown(wait=True)

        finally:
            # Pop pipeline from hierarchy stack
            ctx.pop_pipeline()

    def _run_parallel(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute a pipeline with true parallel execution for independent nodes.

        Nodes that don't depend on each other execute in parallel using processes.
        Best for CPU-bound work that can benefit from multiple cores.

        Note: Uses ProcessPoolExecutor which requires picklable functions and data.

        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values
            ctx: Callback context (created if None)
            output_name: Optional output name(s) to compute

        Returns:
            Dictionary containing only the requested outputs (or all if None)
        """
        # Create or reuse callback context
        if ctx is None:
            ctx = CallbackContext()

        # Determine callbacks to use
        callbacks = (
            pipeline.effective_callbacks
            if hasattr(pipeline, "effective_callbacks")
            else (pipeline.callbacks or [])
        )

        # Set pipeline metadata
        node_ids = [_get_node_id(n) for n in pipeline.execution_order]
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
            requested_outputs = None
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
            outputs = {}
            node_signatures = {}

            # Track pending nodes (only required ones)
            pending_nodes = set(nodes_to_execute)

            # Use custom executor or create ProcessPoolExecutor
            executor = self.executor
            should_close = executor is None

            if executor is None:
                executor = ProcessPoolExecutor(max_workers=self.max_workers)

            try:
                while pending_nodes:
                    # Find nodes ready to execute (all deps satisfied)
                    ready_nodes = []
                    for node in list(pending_nodes):
                        # Check if all required parameters are available
                        if hasattr(node, "parameters"):
                            required_params = set(node.parameters)
                            if required_params.issubset(available_values.keys()):
                                ready_nodes.append(node)
                        else:
                            # No parameters required
                            ready_nodes.append(node)

                    if not ready_nodes:
                        break  # No more nodes can execute (circular dependency or error)

                    # Submit ready nodes to executor
                    future_to_node = {}
                    for node in ready_nodes:
                        # Pass a copy of available_values to avoid race conditions
                        future = executor.submit(
                            self._execute_node_sync,
                            node,
                            dict(available_values),
                            pipeline,
                            callbacks,
                            ctx,
                            node_signatures,
                        )
                        future_to_node[future] = node
                        pending_nodes.remove(node)

                    # Wait for all futures in this batch to complete
                    for future in as_completed(future_to_node):
                        node = future_to_node[future]
                        result, signature = future.result()

                        # Store output and signature
                        if isinstance(result, dict) and hasattr(node, "output_mapping"):
                            # PipelineNode - result is already a dict
                            outputs.update(result)
                            available_values.update(result)
                            for output_name in result.keys():
                                node_signatures[output_name] = signature
                        else:
                            # Regular node - single output
                            outputs[node.output_name] = result
                            available_values[node.output_name] = result
                            node_signatures[node.output_name] = signature

                # Filter outputs if specific outputs were requested
                if requested_outputs is not None:
                    outputs = {k: v for k, v in outputs.items() if k in requested_outputs}

                # Trigger pipeline end callbacks
                pipeline_duration = time.time() - pipeline_start_time
                for callback in callbacks:
                    callback.on_pipeline_end(
                        pipeline.id, outputs, pipeline_duration, ctx
                    )

                return outputs

            finally:
                # Close executor if we created it
                if should_close and hasattr(executor, "shutdown"):
                    executor.shutdown(wait=True)

        finally:
            # Pop pipeline from hierarchy stack
            ctx.pop_pipeline()

    def _execute_node_sync(
        self,
        node,
        available_values: Dict[str, Any],
        pipeline: "Pipeline",
        callbacks: List,
        ctx: CallbackContext,
        node_signatures: Dict[str, str],
    ):
        """Execute a single node synchronously (for parallel execution).

        This is very similar to _execute_node_async but runs synchronously.
        Used by the parallel executor.

        Returns:
            Tuple of (result, signature)
        """
        # The implementation is the same as async but without async/await
        # This is used by ThreadPoolExecutor

        # Handle PipelineNode
        if hasattr(node, "pipeline"):
            node_inputs = {param: available_values[param] for param in node.parameters}

            node_id = _get_node_id(node)
            ctx.set(f"_is_pipeline_node:{node_id}", True)

            # Compute signature
            inner_pipeline = node.pipeline
            inner_code_hashes = []
            for inner_node in inner_pipeline.execution_order:
                if hasattr(inner_node, "pipeline"):
                    inner_code_hashes.append(inner_node.pipeline.id)
                elif hasattr(inner_node, "func"):
                    inner_code_hashes.append(hash_code(inner_node.func))
            code_hash = hashlib.sha256(
                "::".join(inner_code_hashes).encode()
            ).hexdigest()

            inputs_hash = hash_inputs(node_inputs)

            deps_signatures = []
            for param in node.parameters:
                if param in node_signatures:
                    deps_signatures.append(node_signatures[param])
            deps_hash = ":".join(sorted(deps_signatures))

            signature = compute_signature(
                code_hash=code_hash,
                inputs_hash=inputs_hash,
                deps_hash=deps_hash,
            )

            # Check cache
            effective_cache = (
                pipeline.effective_cache
                if hasattr(pipeline, "effective_cache")
                else pipeline.cache
            )
            cache_enabled = effective_cache is not None and node.cache
            result = None

            if cache_enabled:
                result = effective_cache.get(signature)
                if result is not None:
                    for callback in callbacks:
                        callback.on_node_cached(node_id, signature, ctx)

            if result is None:
                node_start_time = time.time()
                for callback in callbacks:
                    callback.on_node_start(node_id, node_inputs, ctx)

                nested_start_time = time.time()
                for callback in callbacks:
                    callback.on_nested_pipeline_start(
                        pipeline.id, inner_pipeline.id, ctx
                    )

                node._exec_ctx = ctx
                old_parent = inner_pipeline._parent
                inner_pipeline._parent = pipeline

                result = node(**node_inputs)

                inner_pipeline._parent = old_parent
                node._exec_ctx = None

                nested_duration = time.time() - nested_start_time
                for callback in callbacks:
                    callback.on_nested_pipeline_end(
                        pipeline.id, inner_pipeline.id, nested_duration, ctx
                    )

                node_duration = time.time() - node_start_time
                for callback in callbacks:
                    callback.on_node_end(
                        _get_node_id(node),
                        result,
                        node_duration,
                        ctx,
                    )

                if cache_enabled:
                    effective_cache.put(signature, result)

            return result, signature

        else:
            # Regular node execution
            node_inputs = {param: available_values[param] for param in node.parameters}

            code_hash = hash_code(node.func)
            inputs_hash = hash_inputs(node_inputs)

            deps_signatures = []
            for param in node.parameters:
                if param in node_signatures:
                    deps_signatures.append(node_signatures[param])
            deps_hash = ":".join(sorted(deps_signatures))

            signature = compute_signature(
                code_hash=code_hash,
                inputs_hash=inputs_hash,
                deps_hash=deps_hash,
            )

            effective_cache = (
                pipeline.effective_cache
                if hasattr(pipeline, "effective_cache")
                else pipeline.cache
            )
            cache_enabled = effective_cache is not None and node.cache
            result = None

            if cache_enabled:
                result = effective_cache.get(signature)
                if result is not None:
                    for callback in callbacks:
                        callback.on_node_cached(_get_node_id(node), signature, ctx)

            if result is None:
                node_start_time = time.time()
                for callback in callbacks:
                    callback.on_node_start(_get_node_id(node), node_inputs, ctx)

                try:
                    result = node(**node_inputs)

                    node_duration = time.time() - node_start_time
                    for callback in callbacks:
                        callback.on_node_end(
                            _get_node_id(node),
                            {node.output_name: result},
                            node_duration,
                            ctx,
                        )

                except Exception as e:
                    for callback in callbacks:
                        callback.on_error(_get_node_id(node), e, ctx)
                    raise

                if cache_enabled:
                    effective_cache.put(signature, result)

            return result, signature

    def map(
        self,
        pipeline: "Pipeline",
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute pipeline.map() with the configured map_execution strategy.

        Intelligently manages resources for nested maps to avoid overwhelming
        the system. For example, if processing 100 items each with a nested
        map of 50 items, limits concurrency to avoid 5000 parallel operations.

        Args:
            pipeline: The pipeline to execute
            items: List of input dictionaries (one per item)
            inputs: Shared inputs for all items
            ctx: Callback context (created if None)
            output_name: Optional output name(s) to compute

        Returns:
            List of output dictionaries (one per item, only requested outputs)
        """
        # Create or reuse callback context
        if ctx is None:
            ctx = CallbackContext()

        # Calculate intelligent worker limit for nested maps
        # If we're already in a map operation, reduce parallelism
        map_depth = ctx.get("_map_depth", 0)
        effective_workers = self._calculate_effective_workers(len(items), map_depth)

        # Store map depth for nested operations
        ctx.set("_map_depth", map_depth + 1)

        try:
            # Dispatch based on map_execution mode
            if self.map_execution == "sequential":
                return self._map_sequential(pipeline, items, inputs, ctx, output_name)
            elif self.map_execution == "async":
                return asyncio.run(
                    self._map_async(pipeline, items, inputs, ctx, effective_workers, output_name)
                )
            elif self.map_execution == "threaded":
                return self._map_threaded(
                    pipeline, items, inputs, ctx, effective_workers, output_name
                )
            elif self.map_execution == "parallel":
                return self._map_parallel(
                    pipeline, items, inputs, ctx, effective_workers, output_name
                )
            else:
                raise ValueError(f"Invalid map_execution mode: {self.map_execution}")
        finally:
            # Restore map depth
            ctx.set("_map_depth", map_depth)

    def _calculate_effective_workers(self, num_items: int, map_depth: int) -> int:
        """Calculate effective worker count for nested map operations.

        Intelligently reduces parallelism for nested maps to avoid
        exponential explosion of concurrent operations.

        Args:
            num_items: Number of items to process
            map_depth: Current nesting depth (0 = top level)

        Returns:
            Effective number of workers to use
        """
        if map_depth == 0:
            # Top-level map: use full worker count
            return min(self.max_workers, num_items)
        elif map_depth == 1:
            # First nested level: reduce to sqrt of max_workers
            return min(int(self.max_workers**0.5) or 1, num_items)
        else:
            # Deeper nesting: use sequential to avoid explosion
            return 1

    def _map_sequential(
        self,
        pipeline: "Pipeline",
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        ctx: CallbackContext,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute map sequentially - one item at a time.

        Simple loop processing. Predictable and easy to debug.
        """
        results = []
        for i, item in enumerate(items):
            # Mark that we're in a map operation
            ctx.set("_in_map", True)
            ctx.set("_map_item_index", i)
            ctx.set("_map_item_start_time", time.time())

            # Merge item inputs with shared inputs
            item_inputs = {**inputs, **item}

            # Execute pipeline for this item with output_name
            result = self.run(pipeline, item_inputs, ctx, output_name=output_name)
            results.append(result)

            # Clear map markers
            ctx.set("_in_map", False)

        return results

    async def _map_async(
        self,
        pipeline: "Pipeline",
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        ctx: CallbackContext,
        max_concurrent: int,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute map with async concurrency.

        Processes multiple items concurrently using asyncio.
        Limits concurrency based on effective_workers calculation.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_item(i: int, item: Dict[str, Any]):
            async with semaphore:
                # Create a new context for this item to avoid conflicts
                item_ctx = CallbackContext()
                # Copy relevant context data
                for key, value in ctx.data.items():
                    if not key.startswith("_"):
                        item_ctx.data[key] = value

                # Mark that we're in a map operation
                item_ctx.set("_in_map", True)
                item_ctx.set("_map_item_index", i)
                item_ctx.set("_map_item_start_time", time.time())
                item_ctx.set("_map_depth", ctx.get("_map_depth", 0))

                # Merge item inputs with shared inputs
                item_inputs = {**inputs, **item}

                # Execute pipeline for this item
                # Note: run() might call asyncio.run() if node_execution="async"
                # To avoid nested event loops, we run in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: self.run(pipeline, item_inputs, item_ctx, output_name=output_name)
                )

                return result

        # Create tasks for all items
        tasks = [process_item(i, item) for i, item in enumerate(items)]

        # Execute all tasks concurrently (with semaphore limiting concurrency)
        results = await asyncio.gather(*tasks)

        return list(results)

    def _map_threaded(
        self,
        pipeline: "Pipeline",
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        ctx: CallbackContext,
        max_concurrent: int,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute map with threaded execution using threads.

        Processes multiple items in parallel using ThreadPoolExecutor.
        Limits concurrency based on effective_workers calculation.
        """

        def process_item(i: int, item: Dict[str, Any]):
            # Create a new context for this item to avoid conflicts
            item_ctx = CallbackContext()
            # Copy relevant context data
            for key, value in ctx.data.items():
                if not key.startswith("_"):
                    item_ctx.data[key] = value

            # Mark that we're in a map operation
            item_ctx.set("_in_map", True)
            item_ctx.set("_map_item_index", i)
            item_ctx.set("_map_item_start_time", time.time())
            item_ctx.set("_map_depth", ctx.get("_map_depth", 0))

            # Merge item inputs with shared inputs
            item_inputs = {**inputs, **item}

            # Execute pipeline for this item
            result = self.run(pipeline, item_inputs, item_ctx, output_name=output_name)

            return (i, result)

        # Use custom executor or create ThreadPoolExecutor
        executor = self.executor
        should_close = executor is None

        if executor is None:
            executor = ThreadPoolExecutor(max_workers=max_concurrent)

        try:
            # Submit all items
            futures = [
                executor.submit(process_item, i, item) for i, item in enumerate(items)
            ]

            # Collect results in order
            results = [None] * len(items)
            for future in as_completed(futures):
                i, result = future.result()
                results[i] = result

            return results
        finally:
            # Close executor if we created it
            if should_close and hasattr(executor, "shutdown"):
                executor.shutdown(wait=True)

    def _map_parallel(
        self,
        pipeline: "Pipeline",
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        ctx: CallbackContext,
        max_concurrent: int,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute map with true parallel execution using processes.

        Processes multiple items in parallel using ProcessPoolExecutor.
        Best for CPU-bound per-item work. Requires picklable functions and data.
        Limits concurrency based on effective_workers calculation.
        """

        def process_item(i: int, item: Dict[str, Any]):
            # Create a new context for this item to avoid conflicts
            item_ctx = CallbackContext()
            # Copy relevant context data
            for key, value in ctx.data.items():
                if not key.startswith("_"):
                    item_ctx.data[key] = value

            # Mark that we're in a map operation
            item_ctx.set("_in_map", True)
            item_ctx.set("_map_item_index", i)
            item_ctx.set("_map_item_start_time", time.time())
            item_ctx.set("_map_depth", ctx.get("_map_depth", 0))

            # Merge item inputs with shared inputs
            item_inputs = {**inputs, **item}

            # Execute pipeline for this item
            result = self.run(pipeline, item_inputs, item_ctx, output_name=output_name)

            return (i, result)

        # Use custom executor or create ProcessPoolExecutor
        executor = self.executor
        should_close = executor is None

        if executor is None:
            executor = ProcessPoolExecutor(max_workers=max_concurrent)

        try:
            # Submit all items
            futures = [
                executor.submit(process_item, i, item) for i, item in enumerate(items)
            ]

            # Collect results in order
            results = [None] * len(items)
            for future in as_completed(futures):
                i, result = future.result()
                results[i] = result

            return results
        finally:
            # Close executor if we created it
            if should_close and hasattr(executor, "shutdown"):
                executor.shutdown(wait=True)


# Modal Backend Implementation
# =============================


def _import_modal():
    """Lazy import modal with clear error message if not installed."""
    try:
        import modal

        return modal
    except ImportError as e:
        raise ImportError(
            "Modal is not installed. Install it with: uv pip install 'hypernodes[modal]' "
            "or: pip install modal cloudpickle"
        ) from e


def _import_cloudpickle():
    """Lazy import cloudpickle with clear error message if not installed."""
    try:
        import cloudpickle

        return cloudpickle
    except ImportError as e:
        raise ImportError(
            "cloudpickle is not installed. Install it with: uv pip install 'hypernodes[modal]' "
            "or: pip install cloudpickle"
        ) from e


class ModalBackend(Backend):
    """Remote execution backend on Modal Labs serverless infrastructure.

    Executes pipelines on Modal's cloud infrastructure with configurable
    resources (GPU, memory, CPU). Each pipeline execution runs in an isolated
    container with your specified configuration.

    This backend requires modal to be installed:
        uv pip install 'hypernodes[modal]'

    Phase 1 Implementation:
    - One pipeline per container (simple & predictable)
    - Basic batching for .map() using Modal's native distribution
    - Direct serialization approach

    Args:
        image: Modal image with dependencies (required)
        gpu: GPU type ("A100", "A10G", "T4", "any", None)
        memory: Memory limit ("32GB", "256GB", etc.)
        cpu: CPU cores (1.0, 2.0, 8.0, etc.)
        timeout: Max execution time in seconds (default: 3600)
        volumes: Volume mounts {"/path": modal.Volume}
        secrets: Modal secrets for API keys, etc.
        max_concurrent: Max parallel containers for .map() (default: 100)

    Example:
        >>> import modal
        >>> from hypernodes import Pipeline, ModalBackend
        >>>
        >>> # Build image with dependencies
        >>> image = (
        ...     modal.Image.debian_slim(python_version="3.12")
        ...     .pip_install("numpy", "pandas")
        ... )
        >>>
        >>> # Configure backend
        >>> backend = ModalBackend(
        ...     image=image,
        ...     gpu="A100",
        ...     memory="32GB"
        ... )
        >>>
        >>> pipeline = Pipeline(nodes=[...], backend=backend)
        >>> result = pipeline.run(inputs={...})
    """

    def __init__(
        self,
        image: Any,  # modal.Image - can't type hint due to optional dependency
        gpu: Optional[str] = None,
        memory: Optional[str] = None,
        cpu: Optional[float] = None,
        timeout: int = 3600,
        volumes: Optional[Dict[str, Any]] = None,  # Dict[str, modal.Volume]
        secrets: Optional[list] = None,
        max_concurrent: int = 100,
        # Execution engine configuration (reused locally and remotely)
        node_execution: Literal[
            "sequential", "async", "threaded", "parallel"
        ] = "sequential",
        map_execution: Literal[
            "sequential", "async", "threaded", "parallel"
        ] = "sequential",
        max_workers: Optional[int] = None,
    ):
        # Lazy import modal
        self.modal = _import_modal()
        self.cloudpickle = _import_cloudpickle()

        # Validate image
        if not isinstance(image, self.modal.Image):
            raise TypeError(f"image must be a modal.Image, got {type(image)}")

        self.image = image
        self.gpu = gpu
        self.memory = memory
        self.cpu = cpu
        self.timeout = timeout
        self.volumes = volumes or {}
        self.secrets = secrets or []
        self.max_concurrent = max_concurrent

        # Engine configuration to be reused remotely
        self._engine_config = {
            "node_execution": node_execution,
            "map_execution": map_execution,
            "max_workers": max_workers,
        }

        # Create Modal app for this backend
        self._app = self.modal.App(name="hypernodes-pipeline")

        # Create the remote function
        self._create_remote_function()

    def _create_remote_function(self):
        """Create Modal function with specified resources."""
        # Build function kwargs
        function_kwargs = {
            "image": self.image,
            "timeout": self.timeout,
            "serialized": True,  # Required for functions not in global scope
        }

        if self.gpu:
            function_kwargs["gpu"] = self.gpu
        if self.memory:
            function_kwargs["memory"] = self.memory
        if self.cpu:
            function_kwargs["cpu"] = self.cpu
        if self.volumes:
            function_kwargs["volumes"] = self.volumes
        if self.secrets:
            function_kwargs["secrets"] = self.secrets

        # Create the remote execution function
        @self._app.function(**function_kwargs)
        def _remote_execute(serialized_payload: bytes) -> bytes:
            """Executes pipeline on Modal infrastructure."""
            # Import cloudpickle in the remote environment
            import cloudpickle

            # Deserialize pipeline, inputs, and output_name
            pipeline, inputs, ctx_data, engine_config, output_name = cloudpickle.loads(
                serialized_payload
            )

            # Reconstruct callback context
            from hypernodes.callbacks import CallbackContext

            ctx = CallbackContext()
            if ctx_data:
                # Restore context state
                for key, value in ctx_data.items():
                    ctx.data[key] = value

            # Reuse LocalBackend semantics via PipelineExecutionEngine
            # Ensure the pipeline uses the SAME LocalBackend instance as the engine
            from hypernodes.backend import PipelineExecutionEngine

            engine = PipelineExecutionEngine(
                node_execution=engine_config.get("node_execution", "sequential"),
                map_execution=engine_config.get("map_execution", "sequential"),
                max_workers=engine_config.get("max_workers"),
            )
            # Assign the engine's local backend to the pipeline for inheritance
            pipeline.backend = engine._ensure_local()  # type: ignore[attr-defined]
            results = engine.run(pipeline, inputs, ctx, output_name=output_name)

            # Serialize and return results
            return cloudpickle.dumps(results)

        self._remote_execute = _remote_execute

    def _serialize_payload(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> bytes:
        """Serialize pipeline, inputs, context, and output_name for remote execution."""
        # Extract context data (only serializable parts)
        ctx_data = {}
        if ctx:
            # Copy relevant context fields, excluding unpicklable objects
            # Skip progress bars, spans, and other callback-internal state
            excluded_prefixes = (
                "progress_bar:",
                "span:",
                "map_span",
                "map_item_span:",
                "current_span",
                "map_node_bars",
            )
            ctx_data = {
                k: v
                for k, v in ctx.data.items()
                if not any(k.startswith(prefix) for prefix in excluded_prefixes)
                and isinstance(v, (str, int, float, bool, list, dict, type(None)))
            }

        # Replace backend with None to avoid serializing ModalBackend recursion
        original_backend = pipeline.backend
        pipeline.backend = None
        
        # Also temporarily remove callbacks - they may contain unpicklable objects
        # (e.g., ProgressCallback with rich/tqdm internals)
        original_callbacks = pipeline.callbacks
        pipeline.callbacks = []

        try:
            # Serialize everything together, including engine config and output_name
            payload = (pipeline, inputs, ctx_data, self._engine_config, output_name)
            result = self.cloudpickle.dumps(payload)
        finally:
            # Restore original backend and callbacks
            pipeline.backend = original_backend
            pipeline.callbacks = original_callbacks

        return result

    def _deserialize_results(self, result_bytes: bytes) -> Dict[str, Any]:
        """Safely deserialize results, handling PyTorch tensors from GPU.
        
        If results contain PyTorch tensors that were saved on CUDA but we're
        running on CPU, this will map them to CPU during deserialization.
        
        Args:
            result_bytes: Serialized results from remote execution
            
        Returns:
            Deserialized results dictionary
        """
        # Check if torch is available and if we need to handle CUDA->CPU mapping
        try:
            import torch
            
            # If torch is available but CUDA is not, we need to map tensors to CPU
            if not torch.cuda.is_available():
                # Use a custom unpickler that maps CUDA tensors to CPU
                class CPUUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module == 'torch.storage' and name == '_load_from_bytes':
                            # Override torch's _load_from_bytes to force CPU mapping
                            def _load_from_bytes_cpu(b):
                                return torch.load(
                                    io.BytesIO(b),
                                    map_location=torch.device('cpu'),
                                    weights_only=False
                                )
                            return _load_from_bytes_cpu
                        return super().find_class(module, name)
                
                # Deserialize with CPU mapping
                return CPUUnpickler(io.BytesIO(result_bytes)).load()
        except ImportError:
            # torch not available, use standard deserialization
            pass
        
        # Standard deserialization (no torch or CUDA is available)
        return self.cloudpickle.loads(result_bytes)

    def run(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute a single pipeline run on Modal.

        Serializes the pipeline and inputs, submits to Modal, executes remotely,
        and returns the deserialized results.

        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values
            ctx: Callback context (for telemetry/progress tracking)
            output_name: Optional output name(s) to compute

        Returns:
            Dictionary of requested pipeline outputs
        """
        # Create context if needed
        if ctx is None:
            ctx = CallbackContext()

        # Serialize payload with output_name
        serialized_payload = self._serialize_payload(pipeline, inputs, ctx, output_name)

        # Submit to Modal and wait for result
        with self._app.run():
            result_bytes = self._remote_execute.remote(serialized_payload)

        # Deserialize results (with CUDA->CPU mapping if needed)
        results = self._deserialize_results(result_bytes)

        return results

    def map(
        self,
        pipeline: "Pipeline",
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None,
        output_name: Union[str, List[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Execute multiple runs on Modal sequentially.

        Note: In typical usage, mapping happens inside a single remote run
        because inner pipelines use LocalBackend remotely. This method exists to
        satisfy the Backend ABC and provide a simple fallback.
        """
        results: List[Dict[str, Any]] = []
        for item in items:
            merged = {**inputs, **item}
            results.append(self.run(pipeline, merged, ctx, output_name=output_name))
        return results
