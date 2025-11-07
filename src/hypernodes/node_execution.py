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

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .cache import compute_signature, hash_code, hash_inputs
from .callbacks import CallbackContext, PipelineCallback

if TYPE_CHECKING:
    from .pipeline import Pipeline
    from .node import Node


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


def compute_node_signature(
    node: "Node",
    inputs: Dict[str, Any],
    node_signatures: Dict[str, str]
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
    # Hash the node's function code
    code_hash = hash_code(node.func)

    # Hash the input values
    inputs_hash = hash_inputs(inputs)

    # Compute dependencies hash from upstream node signatures
    deps_signatures = []
    for param in node.parameters:
        if param in node_signatures:
            deps_signatures.append(node_signatures[param])
    deps_hash = ":".join(sorted(deps_signatures))

    return compute_signature(
        code_hash=code_hash,
        inputs_hash=inputs_hash,
        deps_hash=deps_hash,
    )


def compute_pipeline_node_signature(
    pipeline_node,
    inputs: Dict[str, Any],
    node_signatures: Dict[str, str]
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
    inner_pipeline = pipeline_node.pipeline

    # Hash the inner pipeline structure (all node functions)
    inner_code_hashes = []
    for inner_node in inner_pipeline.execution_order:
        if hasattr(inner_node, "pipeline"):
            # Nested PipelineNode - use its pipeline ID
            inner_code_hashes.append(inner_node.pipeline.id)
        elif hasattr(inner_node, "func"):
            # Regular node - hash its function
            inner_code_hashes.append(hash_code(inner_node.func))

    code_hash = hashlib.sha256("::".join(inner_code_hashes).encode()).hexdigest()

    # Hash inputs
    inputs_hash = hash_inputs(inputs)

    # Compute dependencies hash from upstream node signatures
    deps_signatures = []
    for param in pipeline_node.parameters:
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
    ctx: CallbackContext
) -> Dict[str, Any]:
    """Execute a PipelineNode and return its outputs.

    Args:
        pipeline_node: The PipelineNode to execute
        inputs: Input values for the pipeline node
        pipeline: The parent pipeline (for context)
        callbacks: List of callbacks to trigger
        ctx: Callback context

    Returns:
        Dictionary of outputs from the nested pipeline
    """
    node_id = _get_node_id(pipeline_node)
    inner_pipeline = pipeline_node.pipeline

    # Mark this as a PipelineNode in context
    ctx.set(f"_is_pipeline_node:{node_id}", True)

    # Trigger nested pipeline start callbacks
    nested_start_time = time.time()
    for callback in callbacks:
        callback.on_nested_pipeline_start(pipeline.id, inner_pipeline.id, ctx)

    # Pass context to PipelineNode so it can share with nested pipeline
    pipeline_node._exec_ctx = ctx

    # Temporarily set parent so nested pipeline inherits callbacks/cache/backend
    old_parent = inner_pipeline._parent
    inner_pipeline._parent = pipeline

    try:
        # Call the PipelineNode (it handles all mapping internally)
        result = pipeline_node(**inputs)
    finally:
        # Restore original parent and clean up
        inner_pipeline._parent = old_parent
        pipeline_node._exec_ctx = None

    # Trigger nested pipeline end callbacks
    nested_duration = time.time() - nested_start_time
    for callback in callbacks:
        callback.on_nested_pipeline_end(
            pipeline.id, inner_pipeline.id, nested_duration, ctx
        )

    return result


def execute_single_node(
    node,
    inputs: Dict[str, Any],
    pipeline: "Pipeline",
    callbacks: List[PipelineCallback],
    ctx: CallbackContext,
    node_signatures: Dict[str, str]
) -> Tuple[Any, str]:
    """Execute a single node with caching and callbacks.

    This function handles both regular nodes and PipelineNodes. It:
    1. Computes the node signature
    2. Checks cache (if enabled)
    3. Executes node if cache miss
    4. Triggers all appropriate callbacks
    5. Stores result in cache (if enabled)
    6. Returns (result, signature)

    Args:
        node: The node to execute (Node or PipelineNode)
        inputs: Input values for the node (already extracted from available_values)
        pipeline: The pipeline this node belongs to
        callbacks: List of callbacks to trigger
        ctx: Callback context
        node_signatures: Signatures of upstream nodes (for dependency tracking)

    Returns:
        Tuple of (result, signature) where:
        - result: The node's output value (or dict of outputs for PipelineNode)
        - signature: The computed signature for this execution
    """
    node_id = _get_node_id(node)

    # Compute signature based on node type
    if hasattr(node, "pipeline"):
        # PipelineNode
        signature = compute_pipeline_node_signature(node, inputs, node_signatures)
    else:
        # Regular node
        signature = compute_node_signature(node, inputs, node_signatures)

    # Get effective cache (support inheritance)
    effective_cache = (
        pipeline.effective_cache
        if hasattr(pipeline, "effective_cache")
        else pipeline.cache
    )
    cache_enabled = effective_cache is not None and node.cache
    result = None

    # Check cache if enabled
    if cache_enabled:
        result = effective_cache.get(signature)

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
            result = _execute_pipeline_node(node, inputs, pipeline, callbacks, ctx)
        else:
            # Regular node - call directly
            result = node(**inputs)

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

    # Store in cache if enabled
    if cache_enabled:
        effective_cache.put(signature, result)

    return result, signature
