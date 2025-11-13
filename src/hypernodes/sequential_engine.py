"""Sequential execution engine - minimal implementation.

This engine executes nodes one by one in topological order.
For nested pipelines, it inherits the parent's cache, callbacks, and engine.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .callbacks import CallbackContext
from .map_planner import MapPlanner
from .node_execution import _get_node_id, execute_single_node

if TYPE_CHECKING:
    from .pipeline import Pipeline


class SequentialEngine:
    """Minimal sequential execution engine.
    
    Executes nodes one by one in topological order.
    No parallelism, no async - just simple, predictable execution.
    
    For nested pipelines (PipelineNode), the parent pipeline's configuration
    (cache, callbacks, engine) is inherited by the inner pipeline during execution.
    
    Example:
        >>> engine = SequentialEngine()
        >>> pipeline = Pipeline(nodes=[...], engine=engine)
        >>> result = pipeline.run(inputs={"x": 5})
    """
    
    def run(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        output_name: Optional[Union[str, List[str]]] = None,
        _ctx: Optional[CallbackContext] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute pipeline sequentially.
        
        Args:
            pipeline: Pipeline to execute
            inputs: Input values
            output_name: Optional output name(s) to filter results
            _ctx: Internal callback context (for nested pipelines)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Dictionary of output values
        """
        ctx = _ctx or CallbackContext()
        callbacks = pipeline.callbacks or []
        
        # Trigger pipeline start
        for callback in callbacks:
            callback.on_pipeline_start(pipeline.id, ctx)
        
        # Execute nodes in topological order
        available_values = dict(inputs)
        outputs = {}
        node_signatures = {}
        
        for node in pipeline.graph.execution_order:
            node_id = _get_node_id(node)
            
            # Gather inputs for this node
            node_inputs = {param: available_values[param] for param in node.root_args}
            
            # Execute the node (handles caching, callbacks, PipelineNodes)
            result, signature = execute_single_node(
                node, node_inputs, pipeline, callbacks, ctx, node_signatures
            )
            
            # Store outputs in available values and outputs dict
            self._store_node_outputs(node, result, outputs, available_values, node_signatures, signature)
        
        # Trigger pipeline end
        for callback in callbacks:
            callback.on_pipeline_end(pipeline.id, ctx)
        
        # Filter outputs if requested
        if output_name:
            names = [output_name] if isinstance(output_name, str) else output_name
            return {k: outputs[k] for k in names}
        
        return outputs
    
    def map(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        map_over: Union[str, List[str]],
        map_mode: str = "zip",
        output_name: Optional[Union[str, List[str]]] = None,
        _ctx: Optional[CallbackContext] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Execute pipeline for each item sequentially.
        
        Args:
            pipeline: Pipeline to execute
            inputs: Input values (lists for map_over params, scalars for fixed params)
            map_over: Parameter name(s) to map over
            map_mode: "zip" (parallel iteration) or "product" (all combinations)
            output_name: Optional output name(s) to filter results
            _ctx: Internal callback context
            **kwargs: Additional arguments (ignored)
            
        Returns:
            List of output dictionaries, one per item
        """
        ctx = _ctx or CallbackContext()
        callbacks = pipeline.callbacks or []
        
        # Normalize map_over to list
        map_over_list = [map_over] if isinstance(map_over, str) else map_over
        
        # Use MapPlanner to create items
        planner = MapPlanner()
        items = planner.plan_execution(inputs, map_over_list, map_mode)
        
        # Trigger map start
        for callback in callbacks:
            callback.on_map_start(pipeline.id, len(items), ctx)
        
        # Execute each item sequentially
        results = []
        for idx, item_inputs in enumerate(items):
            for callback in callbacks:
                callback.on_map_item_start(pipeline.id, idx, len(items), ctx)
            
            # Run pipeline for this item
            result = self.run(pipeline, item_inputs, output_name, _ctx=ctx)
            results.append(result)
            
            for callback in callbacks:
                callback.on_map_item_end(pipeline.id, idx, len(items), ctx)
        
        # Trigger map end
        for callback in callbacks:
            callback.on_map_end(pipeline.id, len(items), ctx)
        
        return results
    
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
        
        Handles both regular nodes and PipelineNodes (which return dicts).
        """
        if hasattr(node, "pipeline"):
            # PipelineNode - result is dict of outputs
            outputs.update(result)
            available_values.update(result)
            for output_name_key in result.keys():
                node_signatures[output_name_key] = signature
        else:
            # Regular node
            if isinstance(node.output_name, tuple):
                # Multiple outputs
                for name, val in zip(node.output_name, result):
                    outputs[name] = val
                    available_values[name] = val
                    node_signatures[name] = signature
            else:
                # Single output
                outputs[node.output_name] = result
                available_values[node.output_name] = result
                node_signatures[node.output_name] = signature

