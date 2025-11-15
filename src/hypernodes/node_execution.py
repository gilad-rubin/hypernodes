"""Node execution logic for running individual nodes with caching and callbacks.

This module provides functions for executing single nodes (both regular nodes
and PipelineNodes) with full support for:
- Signature computation for caching
- Cache get/put operations
- Callback lifecycle events
- Error handling
- Async function auto-detection and execution

Key principle: Single Responsibility - this module only knows how to execute
one node at a time. It doesn't know about orchestration, parallelism, or graph
traversal.
"""

import asyncio
import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Awaitable, Dict, List, Optional, Tuple, TypeVar

from .cache import compute_signature, hash_inputs
from .callbacks import CallbackContext, PipelineCallback

if TYPE_CHECKING:
    from .node import Node
    from .pipeline import Pipeline

T = TypeVar("T")

# Shared thread pool for running async in Jupyter/nested event loop contexts
_ASYNC_EXECUTOR: Optional[ThreadPoolExecutor] = None


def _get_async_executor() -> ThreadPoolExecutor:
    """Lazily create a shared thread pool for running coroutines in Jupyter."""
    global _ASYNC_EXECUTOR
    if _ASYNC_EXECUTOR is None:
        _ASYNC_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hypernodes-async")
    return _ASYNC_EXECUTOR


def _run_coroutine_sync(coro: Awaitable[T]) -> T:
    """Run a coroutine to completion, handling Jupyter/nested event loops.

    Automatically detects if we're in a running event loop (e.g., Jupyter)
    and offloads to a background thread if needed.

    Args:
        coro: Coroutine to execute

    Returns:
        Result of the coroutine
    """
    try:
        # Check if there's a running event loop (Jupyter, async framework, etc.)
        asyncio.get_running_loop()
        # Running loop detected - offload to background thread
        executor = _get_async_executor()
        future = executor.submit(asyncio.run, coro)
        return future.result()
    except RuntimeError:
        # No running loop - safe to use asyncio.run() directly
        return asyncio.run(coro)


def _get_node_id(node) -> str:
    """Get a consistent node ID for callbacks and logging.

    Handles regular nodes, DualNodes, PipelineNodes, and nested pipelines.

    Args:
        node: Node object (Node, DualNode, PipelineNode, or Pipeline)

    Returns:
        String identifier for the node
    """
    # DualNode - use name property
    if hasattr(node, "is_dual_node") and node.is_dual_node:
        return node.name

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

        # Look up required outputs from parent pipeline's graph
        # This is the parent-level optimization information
        required_outputs = pipeline.graph.required_outputs.get(pipeline_node)

        # Call the PipelineNode with required outputs
        # Context is already available through contextvar
        result = pipeline_node(required_outputs=required_outputs, **inputs)

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
    """Execute a single node with caching and callbacks.

    Args:
        node: Node or PipelineNode to execute
        inputs: Input values for the node
        pipeline: Parent pipeline (for config inheritance)
        callbacks: List of callbacks to trigger
        ctx: Callback context for state sharing
        node_signatures: Signatures of upstream nodes (for dependency tracking)

    Returns:
        Tuple of (result, signature)
    """
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
        elif hasattr(node, "is_dual_node") and node.is_dual_node:
            # DualNode - use node's code_hash which combines both implementations
            signature = compute_node_signature(node, inputs, node_signatures)
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
                signature_str = signature if signature else ""
                for callback in callbacks:
                    callback.on_map_item_cached(item_index, signature_str, ctx)

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

    try:
        # Execute based on node type
        if hasattr(node, "pipeline"):
            # PipelineNode - delegate to specialized function
            result = _execute_pipeline_node(node, inputs, pipeline, callbacks, ctx)
        elif hasattr(node, "is_dual_node") and node.is_dual_node:
            # DualNode - use singular function (SequentialEngine executes one at a time)
            result = node.singular(**inputs)

            # Handle async functions - auto-detect and run with asyncio
            if inspect.iscoroutine(result):
                result = _run_coroutine_sync(result)
        else:
            # Regular node - call directly
            result = node(**inputs)

            # Handle async functions - auto-detect and run with asyncio
            if inspect.iscoroutine(result):
                result = _run_coroutine_sync(result)

        # Trigger node end callbacks
        node_duration = time.time() - node_start_time
        output_dict = result if isinstance(result, dict) else {node.output_name: result}
        for callback in callbacks:
            callback.on_node_end(node_id, output_dict, node_duration, ctx)

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
        elif hasattr(node, "is_dual_node") and node.is_dual_node:
            signature = compute_node_signature(node, inputs, node_signatures)
        else:
            signature = compute_node_signature(node, inputs, node_signatures)

    # Store in cache if enabled
    if cache_enabled and cache is not None:
        cache.put(signature, result)

    return result, signature
