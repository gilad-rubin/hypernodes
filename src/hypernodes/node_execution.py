"""Node execution logic for running individual nodes with caching and callbacks.

This module provides functions for executing single nodes (both regular nodes
and PipelineNodes) with full support for:
- Signature computation for caching
- Cache get/put operations
- Callback lifecycle events
- Error handling

Key principle: Single Responsibility - this module only knows how to execute
one node at a time. It doesn't know about orchestration, parallelism, or graph
traversal.
"""

import asyncio
import inspect
import time
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from .async_utils import run_coroutine_sync
from .cache import compute_signature, hash_inputs
from .callbacks import CallbackContext, PipelineCallback

if TYPE_CHECKING:
    from .node import Node
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


def _get_sync_runner_strategy(pipeline: "Pipeline") -> str:
    """Determine how to run coroutines for this pipeline's engine."""
    engine = pipeline.engine
    if engine is None:
        return "per_call"
    strategy = getattr(engine, "async_strategy", "per_call")
    if strategy in ("thread_local", "auto", "async_native"):
        return "thread_local"
    return "per_call"


def compute_node_signature(
    node: "Node", inputs: Dict[str, Any], node_signatures: Dict[str, str]
) -> str:
    """Compute signature for a regular node.

    Signature = hash(code_hash + inputs_hash + deps_hash)

    Args:
        node: The node to compute signature for
        inputs: Input values for the node
        node_signatures: Signatures of upstream nodes (for dependency tracking)

    Returns:
        64-character hex string (SHA256)
    """
    # Use cached code hash from Node (avoids expensive inspect.getsource() on every call)
    code_hash = node.code_hash

    # Hash the input values
    inputs_hash = hash_inputs(inputs)

    # Compute dependencies hash from upstream node signatures
    deps_signatures = []
    for param in node.root_args:
        if param in node_signatures:
            deps_signatures.append(node_signatures[param])
    deps_hash = ":".join(sorted(deps_signatures))

    return compute_signature(
        code_hash=code_hash,
        inputs_hash=inputs_hash,
        deps_hash=deps_hash,
    )


def compute_pipeline_node_signature(
    pipeline_node, inputs: Dict[str, Any], node_signatures: Dict[str, str]
) -> str:
    """Compute signature for a PipelineNode.

    Signature = hash(inner_pipeline_structure + inputs + dependencies)

    Args:
        pipeline_node: The PipelineNode to compute signature for
        inputs: Input values for the pipeline node
        node_signatures: Signatures of upstream nodes (for dependency tracking)

    Returns:
        64-character hex string (SHA256)
    """
    # Use cached code hash from pipeline_node (computed at initialization)
    code_hash = pipeline_node.code_hash

    # Hash inputs
    inputs_hash = hash_inputs(inputs)

    # Compute dependencies hash from upstream node signatures
    deps_signatures = []
    for param in pipeline_node.root_args:
        if param in node_signatures:
            deps_signatures.append(node_signatures[param])
    deps_hash = ":".join(sorted(deps_signatures))

    return compute_signature(
        code_hash=code_hash,
        inputs_hash=inputs_hash,
        deps_hash=deps_hash,
    )


def _execute_pipeline_node(
    pipeline_node,
    inputs: Dict[str, Any],
    pipeline: "Pipeline",
    callbacks: List[PipelineCallback],
    ctx: CallbackContext,
) -> Dict[str, Any]:
    """Execute a PipelineNode and return its outputs.
    
    IMPORTANT: The inner pipeline inherits the parent's configuration:
    - cache: Parent's cache instance is used for inner nodes
    - callbacks: Parent's callbacks receive events from inner nodes
    - engine: Parent's engine is used to execute the inner pipeline
    
    This ensures unified behavior across nested pipelines and prevents
    dual caching/callback systems.

    Args:
        pipeline_node: The PipelineNode to execute
        inputs: Input values for the pipeline node
        pipeline: The parent pipeline (provides config to inherit)
        callbacks: List of callbacks to trigger
        ctx: Callback context

    Returns:
        Dictionary of outputs from the nested pipeline
    """
    node_id = _get_node_id(pipeline_node)
    inner_pipeline = pipeline_node.pipeline

    # Mark this as a PipelineNode in context
    ctx.set(f"_is_pipeline_node:{node_id}", True)

    # Save inner pipeline's original configuration
    original_cache = inner_pipeline.cache
    original_callbacks = inner_pipeline.callbacks
    original_engine = inner_pipeline.engine

    # INHERIT PARENT CONFIGURATION
    # This ensures:
    # - Inner nodes cache to parent's cache
    # - Parent's callbacks see all nested node events
    # - Parent's engine controls execution strategy
    inner_pipeline.cache = pipeline.cache
    inner_pipeline.callbacks = pipeline.callbacks
    inner_pipeline.engine = pipeline.engine

    try:
        # Trigger nested pipeline start callbacks
        nested_start_time = time.time()
        for callback in callbacks:
            callback.on_nested_pipeline_start(pipeline.id, inner_pipeline.id, ctx)

        # Call the PipelineNode (it handles all mapping internally)
        # Context is already available through contextvar
        result = pipeline_node(**inputs)

        # Trigger nested pipeline end callbacks
        nested_duration = time.time() - nested_start_time
        for callback in callbacks:
            callback.on_nested_pipeline_end(
                pipeline.id, inner_pipeline.id, nested_duration, ctx
            )

        return result

    finally:
        # Restore inner pipeline's original configuration
        # This is important if the pipeline is reused elsewhere
        inner_pipeline.cache = original_cache
        inner_pipeline.callbacks = original_callbacks
        inner_pipeline.engine = original_engine


def execute_single_node(
    node,
    inputs: Dict[str, Any],
    pipeline: "Pipeline",
    callbacks: List[PipelineCallback],
    ctx: CallbackContext,
    node_signatures: Dict[str, str],
) -> Tuple[Any, str]:
    """Synchronous wrapper that delegates to the async implementation."""
    runner_strategy = _get_sync_runner_strategy(pipeline)
    return run_coroutine_sync(
        execute_single_node_async(
            node,
            inputs,
            pipeline,
            callbacks,
            ctx,
            node_signatures,
            offload_sync=False,
        ),
        strategy=runner_strategy,
    )


async def execute_single_node_async(
    node,
    inputs: Dict[str, Any],
    pipeline: "Pipeline",
    callbacks: List[PipelineCallback],
    ctx: CallbackContext,
    node_signatures: Dict[str, str],
    offload_sync: bool = False,
) -> Tuple[Any, str]:
    """Execute a single node with caching, supporting async contexts."""
    node_id = _get_node_id(node)

    # Get cache from pipeline
    cache = pipeline.cache
    cache_enabled = cache is not None and node.cache

    # Only compute signature if caching is enabled (optimization)
    signature = None
    result = None

    if cache_enabled:
        # Compute signature based on node type
        if hasattr(node, "pipeline"):
            # PipelineNode
            signature = compute_pipeline_node_signature(node, inputs, node_signatures)
        else:
            # Regular node
            signature = compute_node_signature(node, inputs, node_signatures)

        # Check cache
        result = cache.get(signature)

        if result is not None:
            # Cache hit - trigger callbacks
            for callback in callbacks:
                callback.on_node_cached(node_id, signature, ctx)

            # If in a map operation, also trigger map item cached callback
            if ctx.get("_in_map"):
                item_index = ctx.get("_map_item_index")
                for callback in callbacks:
                    callback.on_map_item_cached(item_index, signature, ctx)

            # For regular nodes, return just the value (not wrapped in dict)
            if not hasattr(node, "pipeline"):
                return result, signature
            # For PipelineNodes, result is already a dict
            return result, signature

    # Cache miss or caching disabled - execute node
    node_start_time = time.time()

    # Trigger node start callbacks
    for callback in callbacks:
        callback.on_node_start(node_id, inputs, ctx)

    # If in a map operation, also trigger map item start callback
    if ctx.get("_in_map"):
        item_index = ctx.get("_map_item_index")
        for callback in callbacks:
            callback.on_map_item_start(item_index, ctx)

    try:
        # Execute based on node type
        if hasattr(node, "pipeline"):
            # PipelineNode - delegate to specialized function
            if offload_sync:
                result = await asyncio.to_thread(
                    _execute_pipeline_node, node, inputs, pipeline, callbacks, ctx
                )
            else:
                result = _execute_pipeline_node(node, inputs, pipeline, callbacks, ctx)
        else:
            # Regular node - call directly
            call_async = hasattr(node, "func") and inspect.iscoroutinefunction(
                node.func
            )
            if offload_sync and not call_async:
                result = await asyncio.to_thread(node, **inputs)
            else:
                result = node(**inputs)
            # If result is a coroutine, await it
            if inspect.iscoroutine(result):
                result = await result

        # Trigger node end callbacks
        node_duration = time.time() - node_start_time
        output_dict = result if isinstance(result, dict) else {node.output_name: result}
        for callback in callbacks:
            callback.on_node_end(node_id, output_dict, node_duration, ctx)

        # If in a map operation, also trigger map item end callback
        if ctx.get("_in_map"):
            item_index = ctx.get("_map_item_index")
            map_item_start_time = ctx.get("_map_item_start_time")
            map_item_duration = (
                time.time() - map_item_start_time
                if map_item_start_time
                else node_duration
            )
            for callback in callbacks:
                callback.on_map_item_end(item_index, map_item_duration, ctx)

    except Exception as e:
        # Trigger error callbacks
        for callback in callbacks:
            callback.on_error(node_id, e, ctx)
        raise

    # Compute signature only if needed (for caching or dependency tracking)
    # If caching is disabled globally and node doesn't need signature for deps, skip computation
    if signature is None:
        # Check if any downstream nodes might need this signature for dependency tracking
        # For now, always compute it to maintain correctness
        # TODO: Optimize by tracking which nodes actually need signatures
        if hasattr(node, "pipeline"):
            signature = compute_pipeline_node_signature(node, inputs, node_signatures)
        else:
            signature = compute_node_signature(node, inputs, node_signatures)

    # Store in cache if enabled
    if cache_enabled and cache is not None:
        cache.put(signature, result)

    return result, signature
