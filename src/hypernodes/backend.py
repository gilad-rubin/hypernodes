"""Execution backends for running pipelines."""
from typing import Dict, Any, TYPE_CHECKING

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
        """Execute a pipeline sequentially.
        
        Executes nodes in topological order, collecting outputs as they
        are produced. Supports nested pipelines by delegating to their
        own backends.
        
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
                # Regular node execution
                node_inputs = {
                    param: available_values[param]
                    for param in node.parameters
                }
                result = node(**node_inputs)
                
                # Store output
                outputs[node.output_name] = result
                available_values[node.output_name] = result
        
        # Return only outputs, not inputs
        return outputs
