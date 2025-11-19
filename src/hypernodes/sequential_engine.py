"""Sequential execution engine - minimal implementation.

This engine executes nodes one by one in topological order.
No parallelism, no async - just simple, predictable execution.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .callbacks import CallbackContext, PipelineCallback
from .map_planner import MapPlanner
from .node_execution import execute_single_node
from .orchestrator import ExecutionOrchestrator

if TYPE_CHECKING:
    from .cache import Cache
    from .pipeline import Pipeline


class SeqEngine:
    """Sequential execution engine.

    Executes nodes one by one in topological order.
    """

    def __init__(
        self,
        cache: Optional["Cache"] = None,
        callbacks: Optional[List[PipelineCallback]] = None,
    ):
        self.cache = cache
        self.callbacks = callbacks or []

    def run(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        output_name: Optional[Union[str, List[str]]] = None,
        _ctx: Optional[CallbackContext] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute pipeline sequentially."""
        # Use orchestrator for lifecycle management
        with ExecutionOrchestrator(pipeline, self.callbacks, _ctx) as orchestrator:
            orchestrator.validate_callbacks("SeqEngine")
            orchestrator.notify_start(inputs)

            # Plan
            execution_nodes = self._determine_execution_nodes(pipeline, output_name)

            # Execute
            outputs = self._execute_nodes(
                execution_nodes, inputs, pipeline, orchestrator.ctx
            )

            orchestrator.notify_end(outputs)
            return self._filter_outputs(outputs, output_name)

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
        """Execute pipeline for each item sequentially."""
        with ExecutionOrchestrator(pipeline, self.callbacks, _ctx) as orchestrator:
            orchestrator.validate_callbacks("SeqEngine")

            # Plan
            items, stateful_cache = self._plan_map(inputs, map_over, map_mode, pipeline)

            # Execute
            orchestrator.notify_map_start(len(items))

            results = self._execute_map_loop(
                items, stateful_cache, pipeline, output_name, orchestrator
            )

            orchestrator.notify_map_end()
            return results

    def _determine_execution_nodes(
        self, pipeline: "Pipeline", output_name: Optional[Union[str, List[str]]]
    ) -> List[Any]:
        if output_name is None:
            return pipeline.graph.execution_order
        nodes = pipeline.graph.get_required_nodes(output_name)
        return nodes if nodes is not None else pipeline.graph.execution_order

    def _execute_nodes(
        self,
        nodes: List[Any],
        inputs: Dict[str, Any],
        pipeline: "Pipeline",
        ctx: CallbackContext,
    ) -> Dict[str, Any]:
        available_values = dict(inputs)
        outputs = {}
        node_signatures = {}

        for node in nodes:
            node_inputs = {param: available_values[param] for param in node.root_args}
            result, signature = execute_single_node(
                node,
                node_inputs,
                pipeline,  # Still needed for now, but cache/callbacks are explicit
                self.cache,
                self.callbacks,
                ctx,
                node_signatures,
            )
            self._store_node_result(
                node, result, signature, available_values, outputs, node_signatures
            )

        return outputs

    def _store_node_result(
        self,
        node: Any,
        result: Any,
        signature: str,
        available_values: Dict[str, Any],
        outputs: Dict[str, Any],
        node_signatures: Dict[str, str],
    ) -> None:
        if hasattr(node, "pipeline"):
            # PipelineNode - result is dict of outputs
            for key, value in result.items():
                available_values[key] = value
                outputs[key] = value
                node_signatures[key] = signature
        elif isinstance(node.output_name, tuple):
            # Multi-output node
            for name, val in zip(node.output_name, result):
                available_values[name] = val
                outputs[name] = val
                node_signatures[name] = signature
        else:
            # Single-output node
            available_values[node.output_name] = result
            outputs[node.output_name] = result
            node_signatures[node.output_name] = signature

    def _filter_outputs(
        self, outputs: Dict[str, Any], output_name: Optional[Union[str, List[str]]]
    ) -> Dict[str, Any]:
        if not output_name:
            return outputs
        names = [output_name] if isinstance(output_name, str) else output_name
        return {k: outputs[k] for k in names}

    def _plan_map(
        self,
        inputs: Dict[str, Any],
        map_over: Union[str, List[str]],
        map_mode: str,
        pipeline: "Pipeline",
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        stateful_cache = {
            k: v for k, v in inputs.items() if self._is_stateful_object(v)
        }

        map_over_list = [map_over] if isinstance(map_over, str) else map_over
        planner = MapPlanner()
        items = planner.plan_execution(inputs, map_over_list, map_mode)

        return items, stateful_cache

    def _execute_map_loop(
        self,
        items: List[Dict[str, Any]],
        stateful_cache: Dict[str, Any],
        pipeline: "Pipeline",
        output_name: Optional[Union[str, List[str]]],
        orchestrator: ExecutionOrchestrator,
    ) -> List[Dict[str, Any]]:
        ctx = orchestrator.ctx
        ctx.set("_in_map", True)
        ctx.set("_map_total_items", len(items))

        results = []
        for idx, item_inputs in enumerate(items):
            item_start_time = __import__("time").time()
            ctx.set("_map_item_index", idx)
            ctx.set("_map_item_start_time", item_start_time)

            orchestrator.notify_map_item_start(idx)

            merged_inputs = {**item_inputs, **stateful_cache}
            # Recursively call run, passing the existing context
            result = self.run(pipeline, merged_inputs, output_name, _ctx=ctx)
            results.append(result)

            duration = __import__("time").time() - item_start_time
            orchestrator.notify_map_item_end(idx, duration)

        ctx.set("_in_map", False)
        return results

    def _is_stateful_object(self, obj: Any) -> bool:
        return (
            hasattr(obj, "__class__")
            and hasattr(obj.__class__, "__hypernode_stateful__")
            and obj.__class__.__hypernode_stateful__ is True
        )
