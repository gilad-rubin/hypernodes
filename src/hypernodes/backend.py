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


class ModalBackend:
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
            
            # Deserialize pipeline and inputs
            pipeline, inputs, ctx_data = cloudpickle.loads(serialized_payload)
            
            # Reconstruct callback context
            from hypernodes.callbacks import CallbackContext
            ctx = CallbackContext()
            if ctx_data:
                # Restore context state
                for key, value in ctx_data.items():
                    ctx.data[key] = value
            
            # Execute pipeline using LocalBackend
            # IMPORTANT: We ALWAYS use LocalBackend in the remote environment
            # to avoid recursion (ModalBackend calling ModalBackend)
            from hypernodes.backend import LocalBackend
            
            backend = LocalBackend()
            results = backend.run(pipeline, inputs, ctx)
            
            # Serialize and return results
            return cloudpickle.dumps(results)
        
        self._remote_execute = _remote_execute
    
    def _serialize_payload(
        self, 
        pipeline: 'Pipeline', 
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None
    ) -> bytes:
        """Serialize pipeline, inputs, and context for remote execution."""
        # Extract context data (only serializable parts)
        ctx_data = {}
        if ctx:
            # Copy relevant context fields
            ctx_data = {
                k: v for k, v in ctx.data.items()
                if isinstance(v, (str, int, float, bool, list, dict, type(None)))
            }
        
        # CRITICAL: Replace ModalBackend with LocalBackend to avoid recursion
        # We need to temporarily swap the backend before serialization
        original_backend = pipeline.backend
        pipeline.backend = LocalBackend()
        
        try:
            # Serialize everything together
            payload = (pipeline, inputs, ctx_data)
            result = self.cloudpickle.dumps(payload)
        finally:
            # Restore original backend
            pipeline.backend = original_backend
        
        return result
    
    def run(
        self,
        pipeline: 'Pipeline',
        inputs: Dict[str, Any],
        ctx: Optional[CallbackContext] = None
    ) -> Dict[str, Any]:
        """Execute a single pipeline run on Modal.
        
        Serializes the pipeline and inputs, submits to Modal, executes remotely,
        and returns the deserialized results.
        
        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values
            ctx: Callback context (for telemetry/progress tracking)
            
        Returns:
            Dictionary of pipeline outputs
        """
        # Create context if needed
        if ctx is None:
            ctx = CallbackContext()
        
        # Serialize payload
        serialized_payload = self._serialize_payload(pipeline, inputs, ctx)
        
        # Submit to Modal and wait for result
        with self._app.run():
            result_bytes = self._remote_execute.remote(serialized_payload)
        
        # Deserialize results
        results = self.cloudpickle.loads(result_bytes)
        
        return results
