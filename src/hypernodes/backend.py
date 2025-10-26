"""Execution backends for running pipelines."""
import time
from typing import Dict, Any, Optional, TYPE_CHECKING

from .cache import compute_signature, hash_code, hash_inputs
from .callbacks import CallbackContext

if TYPE_CHECKING:
    from .pipeline import Pipeline


class LocalBackend:
    """Simple sequential execution backend.
    
    This backend executes nodes one at a time in topological order.
    It's the default backend and is ideal for:
    - Development and debugging
    - Single-machine execution
    - Understanding execution flow
    
    Future backends will support parallel and remote execution.
    """
    
    def run(
        self, 
        pipeline: 'Pipeline', 
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None
    ) -> Dict[str, Any]:
        """Execute a pipeline sequentially with caching and callbacks support.
        
        Executes nodes in topological order, collecting outputs as they
        are produced. Supports nested pipelines by delegating to their
        own backends. Integrates with cache backend and callbacks if configured.
        
        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values for root arguments
            ctx: Callback context (created if None)
            
        Returns:
            Dictionary containing only the outputs from nodes (not inputs)
        """
        # Create or reuse callback context
        if ctx is None:
            ctx = CallbackContext()
        
        # Determine callbacks to use (support inheritance via effective_callbacks)
        callbacks = pipeline.effective_callbacks if hasattr(pipeline, 'effective_callbacks') else (pipeline.callbacks or [])
        
        # Set pipeline metadata
        # Get node IDs safely for different node types
        node_ids = []
        for n in pipeline.execution_order:
            if hasattr(n, 'func') and hasattr(n.func, '__name__'):
                node_ids.append(n.func.__name__)
            elif hasattr(n, 'id'):
                node_ids.append(n.id)
            elif hasattr(n, '__name__'):
                node_ids.append(n.__name__)
            else:
                node_ids.append(str(n))
        
        ctx.set_pipeline_metadata(pipeline.id, {
            "total_nodes": len(pipeline.execution_order),
            "node_ids": node_ids
        })
        
        # Push this pipeline onto hierarchy stack
        ctx.push_pipeline(pipeline.id)
        
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
            
            # Execute nodes in topological order
            for node in pipeline.execution_order:
                # Check if it's a PipelineNode (wrapped pipeline with mapping)
                from .pipeline import PipelineNode
                if isinstance(node, PipelineNode):
                    # PipelineNode - execute like a regular node but returns dict
                    node_inputs = {
                        param: available_values[param]
                        for param in node.parameters
                    }
                    
                    # Execute the PipelineNode (it handles mapping internally)
                    result = node(**node_inputs)
                    
                    # Result is a dict of outputs
                    outputs.update(result)
                    available_values.update(result)
                    
                # Handle nested pipelines (raw Pipeline objects)
                elif hasattr(node, 'run'):  # Duck typing for Pipeline
                    # Note: Don't override nested pipeline's callbacks if they're set
                    # The nested pipeline will use effective_callbacks which handles inheritance
                    
                    # Trigger nested pipeline start callbacks (use parent's callbacks for this)
                    nested_start_time = time.time()
                    for callback in callbacks:
                        callback.on_nested_pipeline_start(pipeline.id, node.id, ctx)
                    
                    # Nested pipeline - delegate to its effective backend
                    nested_inputs = {
                        k: available_values[k] 
                        for k in node.root_args
                    }
                    # Use effective_backend for configuration inheritance
                    nested_backend = node.effective_backend if hasattr(node, 'effective_backend') else (node.backend or LocalBackend())
                    nested_results = nested_backend.run(node, nested_inputs, ctx)
                    
                    # Trigger nested pipeline end callbacks
                    nested_duration = time.time() - nested_start_time
                    for callback in callbacks:
                        callback.on_nested_pipeline_end(pipeline.id, node.id, nested_duration, ctx)
                    
                    # Nested pipeline results are outputs
                    outputs.update(nested_results)
                    available_values.update(nested_results)
                else:
                    # Regular node execution with caching and callbacks
                    node_inputs = {
                        param: available_values[param]
                        for param in node.parameters
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
                        deps_hash=deps_hash
                    )
                    
                    # Check cache if enabled and node allows caching
                    # Use effective_cache to support inheritance
                    effective_cache = pipeline.effective_cache if hasattr(pipeline, 'effective_cache') else pipeline.cache
                    cache_enabled = effective_cache is not None and node.cache
                    result = None
                    
                    if cache_enabled:
                        result = effective_cache.get(signature)
                        
                        if result is not None:
                            # Cache hit - trigger callbacks
                            for callback in callbacks:
                                callback.on_node_cached(node.func.__name__, signature, ctx)
                            
                            # If in a map operation, also trigger map item cached callback
                            if ctx.get("_in_map"):
                                item_index = ctx.get("_map_item_index")
                                for callback in callbacks:
                                    callback.on_map_item_cached(item_index, signature, ctx)
                    
                    if result is None:
                        # Cache miss or caching disabled - execute node
                        # Trigger node start callbacks
                        node_start_time = time.time()
                        for callback in callbacks:
                            callback.on_node_start(node.func.__name__, node_inputs, ctx)
                        
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
                                callback.on_node_end(node.func.__name__, {node.output_name: result}, node_duration, ctx)
                            
                            # If in a map operation, also trigger map item end callback
                            if ctx.get("_in_map"):
                                item_index = ctx.get("_map_item_index")
                                map_item_start_time = ctx.get("_map_item_start_time")
                                map_item_duration = time.time() - map_item_start_time if map_item_start_time else 0
                                for callback in callbacks:
                                    callback.on_map_item_end(item_index, map_item_duration, ctx)
                            
                        except Exception as e:
                            # Trigger error callbacks
                            for callback in callbacks:
                                callback.on_error(node.func.__name__, e, ctx)
                            raise
                        
                        # Store in cache if enabled
                        if cache_enabled:
                            effective_cache.put(signature, result)
                    
                    # Store output and signature
                    # Handle PipelineNode which returns dict of outputs
                    if isinstance(result, dict) and hasattr(node, 'output_mapping'):
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
            
            # Trigger pipeline end callbacks
            pipeline_duration = time.time() - pipeline_start_time
            for callback in callbacks:
                callback.on_pipeline_end(pipeline.id, outputs, pipeline_duration, ctx)
            
            return outputs
            
        finally:
            # Pop pipeline from hierarchy stack
            ctx.pop_pipeline()
