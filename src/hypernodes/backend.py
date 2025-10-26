"""Execution backends for running pipelines."""
from typing import Dict, Any, TYPE_CHECKING

from .cache import compute_signature, hash_code, hash_inputs

if TYPE_CHECKING:
    from .pipeline import Pipeline
    from .node import Node


class LocalBackend:
    """Simple sequential execution backend.
    
    This backend executes nodes one at a time in topological order.
    It's the default backend and is ideal for:
    - Development and debugging
    - Single-machine execution
    - Understanding execution flow
    
    Future backends will support parallel and remote execution.
    """
    
    def run(self, pipeline: 'Pipeline', inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a pipeline sequentially with caching support.
        
        Executes nodes in topological order, collecting outputs as they
        are produced. Supports nested pipelines by delegating to their
        own backends. Integrates with cache backend if configured.
        
        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values for root arguments
            
        Returns:
            Dictionary containing only the outputs from nodes (not inputs)
        """
        # Start with provided inputs
        available_values = dict(inputs)
        
        # Track outputs separately (this is what we'll return)
        outputs = {}
        
        # Track node signatures for dependency hashing
        node_signatures = {}
        
        # Execute nodes in topological order
        for node in pipeline.execution_order:
            # Handle nested pipelines
            if hasattr(node, 'run'):  # Duck typing for Pipeline
                # Nested pipeline - delegate to its backend
                nested_inputs = {
                    k: available_values[k] 
                    for k in node.root_args
                }
                nested_results = node.backend.run(node, nested_inputs)
                
                # Nested pipeline results are outputs
                outputs.update(nested_results)
                available_values.update(nested_results)
            else:
                # Regular node execution with caching
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
                cache_enabled = pipeline.cache is not None and node.cache
                result = None
                
                if cache_enabled:
                    result = pipeline.cache.get(signature)
                
                if result is None:
                    # Cache miss or caching disabled - execute node
                    result = node(**node_inputs)
                    
                    # Store in cache if enabled
                    if cache_enabled:
                        pipeline.cache.put(signature, result)
                
                # Store output and signature
                outputs[node.output_name] = result
                available_values[node.output_name] = result
                node_signatures[node.output_name] = signature
        
        # Return only outputs, not inputs
        return outputs
