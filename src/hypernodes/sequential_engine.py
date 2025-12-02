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

        # Track which gate signals are active (for branch nodes)
        satisfied_gates: set = set()
        gate_dependencies = pipeline.graph.gate_dependencies
        branch_gates = pipeline.graph.branch_gates

        for node in nodes:
            # Check if this node should be skipped due to unsatisfied gate dependencies
            if self._should_skip_node(node, satisfied_gates, gate_dependencies):
                # Notify callbacks that node was skipped
                for callback in self.callbacks:
                    if hasattr(callback, "on_node_skipped"):
                        callback.on_node_skipped(
                            node.name,
                            f"Gate dependency not satisfied",
                            ctx,
                        )
                continue

            # Check if all required inputs are available
            # (some might be missing if upstream nodes were skipped due to branching)
            missing_inputs = [
                param for param in node.root_args if param not in available_values
            ]
            if missing_inputs:
                # Skip this node - its upstream was in a different branch
                for callback in self.callbacks:
                    if hasattr(callback, "on_node_skipped"):
                        callback.on_node_skipped(
                            node.name,
                            f"Input(s) not available: {missing_inputs}",
                            ctx,
                        )
                continue

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

            # Handle branch node results specially
            if hasattr(node, "_is_branch") and node._is_branch:
                # result is a boolean - add the winning gate to satisfied_gates
                if result:
                    satisfied_gates.add(node.true_gate)
                    # Notify callbacks of branch decision
                    for callback in self.callbacks:
                        if hasattr(callback, "on_branch_decision"):
                            callback.on_branch_decision(node.name, True, ctx)
                else:
                    satisfied_gates.add(node.false_gate)
                    for callback in self.callbacks:
                        if hasattr(callback, "on_branch_decision"):
                            callback.on_branch_decision(node.name, False, ctx)
                # Store gate signals in available_values (for consistency)
                self._store_branch_result(
                    node, result, signature, available_values, outputs, node_signatures
                )
            else:
                self._store_node_result(
                    node, result, signature, available_values, outputs, node_signatures
                )

        # Filter out gate signals from outputs (they're internal)
        return {k: v for k, v in outputs.items() if k not in branch_gates}

    def _should_skip_node(
        self,
        node: Any,
        satisfied_gates: set,
        gate_dependencies: Dict[Any, List[str]],
    ) -> bool:
        """Check if a node should be skipped due to unsatisfied gate dependencies.

        A node is skipped if it has gate dependencies and none of them are satisfied.

        Args:
            node: The node to check
            satisfied_gates: Set of gate signals that have been activated
            gate_dependencies: Mapping from node to list of gate signals it depends on

        Returns:
            True if the node should be skipped
        """
        if node not in gate_dependencies:
            return False

        # Get the gates this node depends on
        required_gates = gate_dependencies[node]
        if not required_gates:
            return False

        # Check if ANY of the required gates are satisfied
        # (a node might have multiple gates if it's in nested branches)
        for gate in required_gates:
            if gate in satisfied_gates:
                return False

        # None of the required gates are satisfied - skip this node
        return True

    def _store_branch_result(
        self,
        node: Any,
        result: bool,
        signature: str,
        available_values: Dict[str, Any],
        outputs: Dict[str, Any],
        node_signatures: Dict[str, str],
    ) -> None:
        """Store the result of a branch node execution.

        Branch nodes produce gate signals, not data outputs.
        Only the winning gate is stored.

        Args:
            node: The BranchNode
            result: Boolean result of the branch condition
            signature: Computed signature for caching
            available_values: Dict of available values to update
            outputs: Dict of outputs to update
            node_signatures: Dict of signatures to update
        """
        if result:
            gate_name = node.true_gate
        else:
            gate_name = node.false_gate

        # Store the gate signal (value is True to indicate it's active)
        available_values[gate_name] = True
        outputs[gate_name] = True
        node_signatures[gate_name] = signature

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

        # Optimization: Check if pipeline contains ONLY a single DualNode
        # If so, we can execute it as a batch operation instead of looping
        execution_nodes = self._determine_execution_nodes(pipeline, output_name)
        if (
            len(execution_nodes) == 1
            and hasattr(execution_nodes[0], "is_dual_node")
            and execution_nodes[0].is_dual_node
        ):
            return self._execute_dual_node_batch(
                execution_nodes[0],
                items,
                stateful_cache,
                pipeline,
                output_name,
                orchestrator,
            )

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

    def _execute_dual_node_batch(
        self,
        node: Any,
        items: List[Dict[str, Any]],
        stateful_cache: Dict[str, Any],
        pipeline: "Pipeline",
        output_name: Optional[Union[str, List[str]]],
        orchestrator: ExecutionOrchestrator,
    ) -> List[Dict[str, Any]]:
        """Execute a single DualNode in batch mode using PyArrow."""
        try:
            import pyarrow as pa
        except ImportError:
            # Fallback to loop if pyarrow is missing (or raise error if enforced)
            # Since DualNode explicitly checks for pyarrow, we can assume it's available
            # if we are here, OR we should let the import error bubble up.
            raise ImportError(
                "DualNode batch execution requires 'pyarrow'. "
                "Please install it with: uv add pyarrow"
            )

        # 1. Collect inputs
        # We need to transpose list of dicts to dict of lists
        batch_inputs = {}
        for param in node.root_args:
            if param in stateful_cache:
                # Stateful param - pass as single value (broadcast/constant)
                # But wait, the batch function expects array inputs usually.
                # However, for stateful params (like models), we usually pass the object itself.
                batch_inputs[param] = stateful_cache[param]
            else:
                # Map param - collect list
                batch_inputs[param] = [item.get(param) for item in items]

        # 2. Convert map inputs to PyArrow Arrays
        # Strict Input Contract: All mapped inputs must be convertible to pa.Array
        arrow_inputs = {}
        for k, v in batch_inputs.items():
            if k in stateful_cache:
                # Stateful/constant params: pass as-is (scalars)
                arrow_inputs[k] = v
            else:
                # Mapped params: convert to pa.Array
                try:
                    arrow_inputs[k] = pa.array(v)
                except Exception as e:
                    raise TypeError(
                        f"Failed to convert input '{k}' to PyArrow Array: {e}\n"
                        f"DualNode batch functions require inputs that can be converted to pa.Array.\n"
                        f"For complex types (dataclasses, custom objects), use regular nodes with .map() instead.\n"
                        f"Batch optimization is intended for vectorized operations on primitive types."
                    )

        # 3. Execute Batch Function
        ctx = orchestrator.ctx

        # Notify start (as a single batch operation)
        # We could notify per-item start/end, but that's fake.
        # Let's just notify map_start/end at the orchestrator level which is already done.

        try:
            batch_result = node.batch(**arrow_inputs)
        except Exception as e:
            raise RuntimeError(
                f"Error executing batch function for node '{node.output_name}': {e}"
            )

        # 4. Convert output back to list of dicts
        # We relax the contract here to allow list, np.ndarray, or pa.Array
        result_list = []

        if isinstance(batch_result, list):
            result_list = batch_result
        elif hasattr(batch_result, "tolist"):  # numpy, daft series
            result_list = batch_result.tolist()
        elif hasattr(batch_result, "to_pylist"):  # pyarrow
            result_list = batch_result.to_pylist()
        else:
            # Try iterating as last resort
            try:
                result_list = list(batch_result)
            except Exception:
                raise TypeError(
                    f"Batch function for '{node.output_name}' returned {type(batch_result)}. "
                    "Expected pyarrow.Array, list, or numpy.ndarray."
                )

        if len(result_list) != len(items):
            raise RuntimeError(
                f"Batch function returned {len(result_list)} items, expected {len(items)}. "
                "Batch functions must preserve input length."
            )

        # Reconstruct results
        results = []
        for val in result_list:
            if isinstance(node.output_name, str):
                results.append({node.output_name: val})
            else:
                # Tuple output support would require struct array or similar,
                # simpler to assume single output for now or handle dict/struct return
                results.append({node.output_name[0]: val})  # Simplified

        return results

    def _is_stateful_object(self, obj: Any) -> bool:
        return (
            hasattr(obj, "__class__")
            and hasattr(obj.__class__, "__hypernode_stateful__")
            and obj.__class__.__hypernode_stateful__ is True
        )
