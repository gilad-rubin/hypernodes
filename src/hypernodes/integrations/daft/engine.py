"""Clean DaftEngine implementation using modular operations.

Key Design:
- Engine acts as a facade/orchestrator.
- Operations (Simple, Batch, Pipeline) handle execution and code generation.
- CodeGenContext tracks state for code generation.
"""

import inspect
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

try:
    import daft

    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False

from hypernodes.callbacks import CallbackContext
from hypernodes.integrations.daft.codegen import CodeGenContext
from hypernodes.integrations.daft.operations import (
    BatchNodeOperation,
    DaftOperation,
    DualNodeOperation,
    ExecutionContext,
    FunctionNodeOperation,
    PipelineNodeOperation,
    SimplePipelineOperation,
)
from hypernodes.map_planner import MapPlanner
from hypernodes.protocols import Engine

if TYPE_CHECKING:
    from hypernodes.pipeline import Pipeline


class DaftEngine(Engine):
    """DaftEngine using modular operations for execution and code generation."""

    def __init__(
        self,
        use_batch_udf: bool = True,
        default_daft_config: Optional[Dict[str, Any]] = None,
        cache: Optional[Any] = None,
    ):
        if not DAFT_AVAILABLE:
            raise ImportError("daft is required. pip install getdaft")

        self.use_batch_udf = use_batch_udf
        self.default_daft_config = default_daft_config or {}
        self.cache = cache
        self._is_map_context = False
        self._stateful_inputs = {}

    def run(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        output_name: Optional[Union[str, List[str]]] = None,
        _ctx: Optional[CallbackContext] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute pipeline with 1-row DataFrame."""
        # Initialize context and dispatcher
        ctx = _ctx or CallbackContext()
        is_new_context = _ctx is None

        if is_new_context:
            ctx.push_pipeline(pipeline.id)

        try:
            callbacks = pipeline.callbacks or kwargs.get("callbacks") or []
            # Ensure callbacks is a list
            if not isinstance(callbacks, list):
                callbacks = [callbacks]

            from hypernodes.callbacks import CallbackDispatcher

            dispatcher = CallbackDispatcher(callbacks)

            # Set pipeline metadata for progress bar
            execution_nodes = pipeline.graph.execution_order
            from hypernodes.node_execution import _get_node_id

            ctx.set_pipeline_metadata(
                pipeline.id,
                {
                    "total_nodes": len(execution_nodes),
                    "pipeline_name": pipeline.name or pipeline.id,
                    "node_ids": [_get_node_id(node) for node in execution_nodes],
                },
            )

            # Notify pipeline start
            import time

            start_time = time.time()
            dispatcher.notify_pipeline_start(pipeline.id, inputs, ctx)

            df = self._build_dataframe(pipeline, inputs)

            # Execute
            context = ExecutionContext(self, self._stateful_inputs)
            node_signatures = {}
            nodes_to_cache = []

            try:
                available_columns = set(df.column_names)
                for node in pipeline.graph.execution_order:
                    # Compute signature
                    signature = self._compute_node_signature(
                        node, inputs, node_signatures
                    )
                    if signature:
                        node_signatures[node.output_name] = signature

                    # Check cache
                    cached_result = None
                    if self.cache and node.cache and signature:
                        cached_result = self.cache.get(signature)

                    if cached_result is not None:
                        # Cache HIT: Join cached data
                        import daft

                        # Handle different return types
                        if isinstance(cached_result, dict):
                            cached_data = {"__row_id__": [0]}
                            for k, v in cached_result.items():
                                cached_data[k] = [v]
                        else:
                            cached_data = {
                                "__row_id__": [0],
                                node.output_name: [cached_result],
                            }

                        cached_df = daft.from_pydict(cached_data)
                        df = df.join(cached_df, on="__row_id__")

                        # Update available columns
                        if hasattr(node, "output_name"):
                            out = node.output_name
                            if isinstance(out, (list, tuple)):
                                available_columns.update(out)
                            else:
                                available_columns.add(out)

                        # Notify cache hit
                        dispatcher.notify_node_cached(node.output_name, signature, ctx)
                        continue

                    # Cache MISS: Execute
                    op = self._create_operation(node)
                    df = op.execute(df, available_columns, context)

                    if hasattr(node, "output_name"):
                        out = node.output_name
                        if isinstance(out, (list, tuple)):
                            available_columns.update(out)
                        else:
                            available_columns.add(out)

                        # Mark for caching if enabled
                        if self.cache and node.cache and signature:
                            nodes_to_cache.append((node, signature))

                # Attach subscriber if callbacks are present
                subscriber_alias = None
                if DAFT_AVAILABLE and callbacks:
                    daft_ctx = get_context()
                    subscriber = HyperNodesDaftSubscriber(dispatcher, ctx, pipeline.id)
                    subscriber_alias = f"hypernodes_{pipeline.id}"
                    daft_ctx.attach_subscriber(subscriber_alias, subscriber)

                try:
                    result_df = df.collect()
                finally:
                    if DAFT_AVAILABLE and subscriber_alias:
                        get_context().detach_subscriber(subscriber_alias)

                # Store results in cache
                if nodes_to_cache:
                    pydict = result_df.to_pydict()
                    for node, signature in nodes_to_cache:
                        # Extract result
                        if isinstance(node.output_name, (list, tuple)):
                            result = {
                                name: pydict[name][0]
                                for name in node.output_name
                                if name in pydict
                            }
                            self.cache.put(signature, result)
                        elif node.output_name in pydict:
                            result = pydict[node.output_name][0]
                            self.cache.put(signature, result)

            finally:
                context.shutdown()

            outputs = self._extract_single_row_outputs(result_df, pipeline)

            duration = time.time() - start_time
            dispatcher.notify_pipeline_end(pipeline.id, outputs, duration, ctx)

            if output_name:
                names = [output_name] if isinstance(output_name, str) else output_name
                return {k: outputs[k] for k in names}
            return outputs

        finally:
            if is_new_context:
                ctx.pop_pipeline()

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
        """Execute pipeline over multiple inputs."""
        ctx = _ctx or CallbackContext()
        is_new_context = _ctx is None

        if is_new_context:
            ctx.push_pipeline(pipeline.id)

        try:
            callbacks = pipeline.callbacks or kwargs.get("callbacks") or []
            # Ensure callbacks is a list
            if not isinstance(callbacks, list):
                callbacks = [callbacks]

            from hypernodes.callbacks import CallbackDispatcher

            dispatcher = CallbackDispatcher(callbacks)

            map_over_list = [map_over] if isinstance(map_over, str) else map_over
            planner = MapPlanner()
            execution_plans = planner.plan_execution(inputs, map_over_list, map_mode)

            if not execution_plans:
                return []

            # Set pipeline metadata for progress bar
            execution_nodes = pipeline.graph.execution_order
            from hypernodes.node_execution import _get_node_id

            ctx.set_pipeline_metadata(
                pipeline.id,
                {
                    "total_nodes": len(execution_nodes),
                    "pipeline_name": pipeline.name or pipeline.id,
                    "node_ids": [_get_node_id(node) for node in execution_nodes],
                },
            )

            self._is_map_context = True

            # Notify map start
            import time

            start_time = time.time()
            dispatcher.notify_map_start(len(execution_plans), ctx)

            # Set map context variables
            ctx.set("_in_map", True)
            ctx.set("_map_total_items", len(execution_plans))
            ctx.set("map_start_time", start_time)

            try:
                # PRE-EXECUTION CACHE CHECK (per-item)
                cached_results = []  # List[Optional[Dict]]
                uncached_indices = []  # List[int]
                node_signatures_per_item = []  # List[Dict[str, str]]

                for idx, plan in enumerate(execution_plans):
                    item_node_sigs = {}
                    item_cached_results = {}
                    all_cached = True

                    for node in pipeline.graph.execution_order:
                        # Compute signature for THIS item
                        sig = self._compute_node_signature(node, plan, item_node_sigs)
                        if sig:
                            item_node_sigs[node.output_name] = sig

                        # Check cache
                        if self.cache and node.cache and sig:
                            cached = self.cache.get(sig)
                            if cached is not None:
                                # Handle multi-output nodes
                                if isinstance(node.output_name, (list, tuple)):
                                    for name in node.output_name:
                                        if isinstance(cached, dict) and name in cached:
                                            item_cached_results[name] = cached[name]
                                else:
                                    item_cached_results[node.output_name] = cached

                                # Notify cache hit for this item
                                dispatcher.notify_node_cached(
                                    node.output_name, sig, ctx
                                )
                            else:
                                all_cached = False
                                break
                        else:
                            all_cached = False
                            break

                    if all_cached:
                        cached_results.append(item_cached_results)
                    else:
                        cached_results.append(None)
                        uncached_indices.append(idx)

                    node_signatures_per_item.append(item_node_sigs)

                # If everything is cached, return immediately
                if not uncached_indices:
                    # Build result list from cache
                    final_results = [
                        cached_results[i] for i in range(len(execution_plans))
                    ]
                    duration = time.time() - start_time
                    dispatcher.notify_map_end(duration, ctx)

                    # Filter by output_name if specified
                    if output_name:
                        output_names = (
                            [output_name]
                            if isinstance(output_name, str)
                            else output_name
                        )
                        final_results = [
                            {k: v for k, v in result.items() if k in output_names}
                            for result in final_results
                        ]

                    return final_results

                # Build DataFrame ONLY for uncached items
                uncached_plans = [execution_plans[i] for i in uncached_indices]
                df = self._build_dataframe_from_plans(pipeline, uncached_plans)

                context = ExecutionContext(self, self._stateful_inputs)

                try:
                    available_columns = set(df.column_names)

                    # Execute all nodes for uncached items
                    for node in pipeline.graph.execution_order:
                        op = self._create_operation(node)
                        df = op.execute(df, available_columns, context)

                        if hasattr(node, "output_name"):
                            out = node.output_name
                            if isinstance(out, (list, tuple)):
                                available_columns.update(out)
                            else:
                                available_columns.add(out)

                    # Attach subscriber if callbacks are present
                    subscriber_alias = None
                    if DAFT_AVAILABLE and callbacks:
                        daft_ctx = get_context()
                        subscriber = HyperNodesDaftSubscriber(
                            dispatcher, ctx, pipeline.id
                        )
                        subscriber_alias = f"hypernodes_map_{pipeline.id}"
                        daft_ctx.attach_subscriber(subscriber_alias, subscriber)

                    try:
                        result_df = df.collect()
                    finally:
                        if DAFT_AVAILABLE and subscriber_alias:
                            get_context().detach_subscriber(subscriber_alias)

                    # Extract uncached results
                    uncached_results = self._extract_multi_row_outputs_as_list(
                        result_df, pipeline, None
                    )

                    # Cache newly computed results (per-item)
                    for uncached_idx, original_idx in enumerate(uncached_indices):
                        for node in pipeline.graph.execution_order:
                            if node.cache and self.cache:
                                sig = node_signatures_per_item[original_idx].get(
                                    node.output_name
                                )
                                if sig:
                                    # Extract result for this node
                                    if isinstance(node.output_name, (list, tuple)):
                                        result_data = {
                                            name: uncached_results[uncached_idx][name]
                                            for name in node.output_name
                                            if name in uncached_results[uncached_idx]
                                        }
                                        self.cache.put(sig, result_data)
                                    elif (
                                        node.output_name
                                        in uncached_results[uncached_idx]
                                    ):
                                        result_data = uncached_results[uncached_idx][
                                            node.output_name
                                        ]
                                        self.cache.put(sig, result_data)

                finally:
                    context.shutdown()

                # Merge cached + newly computed results
                final_results = []
                uncached_iter = iter(uncached_results)
                for i in range(len(execution_plans)):
                    if cached_results[i] is not None:
                        final_results.append(cached_results[i])
                    else:
                        final_results.append(next(uncached_iter))

            finally:
                self._is_map_context = False
                ctx.set("_in_map", False)

                duration = time.time() - start_time
                dispatcher.notify_map_end(duration, ctx)

            # Filter by output_name if specified
            if output_name:
                output_names = (
                    [output_name] if isinstance(output_name, str) else output_name
                )
                final_results = [
                    {k: v for k, v in result.items() if k in output_names}
                    for result in final_results
                ]

            return final_results

        finally:
            if is_new_context:
                ctx.pop_pipeline()

    def _compute_node_signature(
        self, node: Any, inputs: Dict[str, Any], node_signatures: Dict[str, str]
    ) -> Optional[str]:
        """Compute signature for a node."""
        from hypernodes.cache import compute_signature, hash_inputs

        # 1. Code Hash
        if hasattr(node, "code_hash"):
            code_hash = node.code_hash
        else:
            return None

        # 2. Inputs Hash
        # We need to extract inputs for this node from the global inputs
        # Only if they are root inputs.
        node_inputs = {}
        for arg in node.root_args:
            if arg in inputs:
                node_inputs[arg] = inputs[arg]

        inputs_hash = hash_inputs(node_inputs)

        # 3. Dependencies Hash (Upstream signatures)
        deps_signatures = []
        for arg in node.root_args:
            if arg in node_signatures:
                deps_signatures.append(node_signatures[arg])

        deps_hash = ":".join(sorted(deps_signatures))

        return compute_signature(code_hash, inputs_hash, deps_hash)

    def generate_code(
        self, pipeline: "Pipeline", inputs: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate Daft code for the pipeline."""
        context = CodeGenContext()

        # 1. Setup Inputs
        df_var = "df"
        available_columns = set(inputs.keys()) if inputs else set()

        operation_lines = []

        # Generate input loading code
        if inputs:
            # Handle scalar inputs by wrapping them in lists
            # This mirrors how we handle inputs in _build_dataframe
            processed_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, list):
                    processed_inputs[k] = v
                else:
                    processed_inputs[k] = [v]

            inputs_str = repr(processed_inputs)
            operation_lines.append("# Load inputs")
            operation_lines.append(f"{df_var} = daft.from_pydict({inputs_str})")
        else:
            operation_lines.append(f"{df_var} = daft.from_pydict({{}})")

        operation_lines.append("")

        # 2. Generate Operations
        for node in pipeline.graph.execution_order:
            op = self._create_operation(node)
            code = op.generate_code(df_var, available_columns, context)
            operation_lines.append(code)
            if hasattr(node, "output_name"):
                out = node.output_name
                if isinstance(out, (list, tuple)):
                    available_columns.update(out)
                else:
                    available_columns.add(out)

        return context.generate_full_code(operation_lines)

    def _create_operation(self, node: Any) -> DaftOperation:
        """Factory method to create the appropriate operation for a node."""
        # Check for PipelineNode
        if hasattr(node, "pipeline"):
            if hasattr(node, "map_over") and node.map_over:
                return PipelineNodeOperation(node)
            # Simple nested pipeline
            return SimplePipelineOperation(node)

        # Check for DualNode
        if hasattr(node, "is_dual_node") and node.is_dual_node:
            return DualNodeOperation(node, is_map_context=self._is_map_context)

        # Regular Node
        has_stateful_args = any(arg in self._stateful_inputs for arg in node.root_args)
        is_async = self._is_async_node(node)
        if (
            self._is_map_context
            and self.use_batch_udf
            and not has_stateful_args
            and not is_async
            and self._should_use_batch_udf(node)
        ):
            return BatchNodeOperation(node)

        return FunctionNodeOperation(node)

    def _should_use_batch_udf(self, node: Any) -> bool:
        """Determine if a node should use batch UDF."""
        import inspect
        from typing import get_origin

        # Same logic as before
        sig = inspect.signature(node.func)
        return_annotation = sig.return_annotation
        if return_annotation != inspect.Signature.empty:
            origin = get_origin(return_annotation)
            if origin in (list, dict):
                return False

        for param_name in node.root_args:
            param = sig.parameters.get(param_name)
            if param and param.annotation != inspect.Parameter.empty:
                origin = get_origin(param.annotation)
                if origin in (list, dict):
                    return False
        return True

    def _build_dataframe(
        self, pipeline: "Pipeline", inputs: Dict[str, Any]
    ) -> "daft.DataFrame":
        """Build 1-row Daft DataFrame."""
        df_inputs = {}
        stateful_inputs = {}

        for k, v in inputs.items():
            is_stateful = self._is_stateful_object(v)
            if is_stateful:
                stateful_inputs[k] = v
            else:
                df_inputs[k] = v

        self._stateful_inputs = stateful_inputs

        # Add __row_id__ for cache joining
        df_inputs["__row_id__"] = 0

        if df_inputs:
            return daft.from_pydict({k: [v] for k, v in df_inputs.items()})
        return daft.from_pydict({"__row_id__": [0]})

    def _build_dataframe_from_plans(
        self, pipeline: "Pipeline", execution_plans: List[Dict[str, Any]]
    ) -> "daft.DataFrame":
        """Build N-row Daft DataFrame."""
        input_data = {}
        stateful_inputs = {}

        for key in execution_plans[0].keys():
            values = [plan[key] for plan in execution_plans]
            if values:
                is_stateful = self._is_stateful_object(values[0])
                if is_stateful:
                    # Check constant
                    is_constant = all(v is values[0] for v in values[1:])
                    if is_constant:
                        stateful_inputs[key] = values[0]
                        continue
            input_data[key] = values

        self._stateful_inputs = stateful_inputs

        # Add __row_id__ column
        input_data["__row_id__"] = list(range(len(execution_plans)))

        if input_data:
            return daft.from_pydict(input_data)
        return daft.from_pydict({"__row_id__": list(range(len(execution_plans)))})

    def _is_stateful_object(self, obj: Any) -> bool:
        return (
            hasattr(obj, "__class__")
            and hasattr(obj.__class__, "__hypernode_stateful__")
            and obj.__class__.__hypernode_stateful__ is True
        )

    def _is_async_node(self, node: Any) -> bool:
        """Detect whether a node wraps an async function."""
        # Nodes expose an is_async property after wrapping. Fall back to inspecting the function.
        if hasattr(node, "is_async"):
            return bool(node.is_async)

        func = getattr(node, "func", None)
        if func is None:
            return False

        real_func = func
        while hasattr(real_func, "__wrapped__"):
            real_func = real_func.__wrapped__

        return inspect.iscoroutinefunction(real_func) or (
            hasattr(real_func, "__code__") and (real_func.__code__.co_flags & 0x80)
        )

    def _extract_single_row_outputs(
        self, result_df: "daft.DataFrame", pipeline: "Pipeline"
    ) -> Dict[str, Any]:
        output_names = []
        for node in pipeline.graph.execution_order:
            output_name = node.output_name
            if isinstance(output_name, tuple):
                output_names.extend(output_name)
            else:
                output_names.append(output_name)

        py_dict = result_df.to_pydict()
        return {
            k: v[0]
            for k, v in py_dict.items()
            if k in output_names and k != "__row_id__"
        }

    def _extract_multi_row_outputs_as_list(
        self,
        result_df: "daft.DataFrame",
        pipeline: "Pipeline",
        output_name: Optional[Union[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        if output_name:
            output_names = (
                [output_name] if isinstance(output_name, str) else output_name
            )
        else:
            output_names = []
            for node in pipeline.graph.execution_order:
                node_output_name = node.output_name
                if isinstance(node_output_name, tuple):
                    output_names.extend(node_output_name)
                else:
                    output_names.append(node_output_name)

        py_dict = result_df.to_pydict()
        num_rows = len(list(py_dict.values())[0]) if py_dict else 0

        results = []
        for row_idx in range(num_rows):
            row_dict = {}
            for output_name_key in output_names:
                if output_name_key in py_dict and output_name_key != "__row_id__":
                    row_dict[output_name_key] = py_dict[output_name_key][row_idx]
            results.append(row_dict)
        return results


if DAFT_AVAILABLE:
    from daft.context import get_context
    from daft.daft import PyMicroPartition, PyNodeInfo, PyQueryMetadata
    from daft.subscribers.abc import Subscriber

    from hypernodes.callbacks import CallbackContext

    class HyperNodesDaftSubscriber(Subscriber):
        """Subscriber that bridges Daft events to HyperNodes telemetry."""

        def __init__(self, dispatcher: Any, ctx: CallbackContext, pipeline_id: str):
            self.dispatcher = dispatcher
            self.ctx = ctx
            self.pipeline_id = pipeline_id
            self.node_map = {}  # Map node_id (int) to node name (str)
            self.total_nodes = 0
            self.completed_nodes = 0

        def on_query_start(self, query_id: str, metadata: PyQueryMetadata) -> None:
            pass

        def on_query_end(self, query_id: str) -> None:
            pass

        def on_result_out(self, query_id: str, result: PyMicroPartition) -> None:
            pass

        def on_optimization_start(self, query_id: str) -> None:
            pass

        def on_optimization_end(self, query_id: str, optimized_plan: str) -> None:
            pass

        def on_exec_start(self, query_id: str, node_infos: list[PyNodeInfo]) -> None:
            # Build map of node_id -> name/details
            self.total_nodes = len(node_infos)
            self.node_map = {}

            for info in node_infos:
                # Daft UDF nodes are named like "UDF {name}-{uuid}"
                # We enforced the name to be the node output_name in operations.py
                if info.name.startswith("UDF "):
                    # Extract name: "UDF {name}-{uuid}" -> "{name}"
                    # The UUID part is 36 chars + 1 hyphen = 37 chars
                    # But let's use a safer split approach
                    parts = info.name[4:].rsplit("-", 5)  # UUID is usually 5 parts
                    if len(parts) > 1:
                        # Reassemble name parts (in case name had hyphens)
                        # The last 5 parts are UUID, everything before is name
                        # Actually, Daft might just append a single UUID string?
                        # Let's just split by " " and take the rest, then strip the UUID suffix?
                        # The demo showed "UDF multiply_two-a66e..."
                        # So it's "UDF " + name + "-" + uuid

                        # Regex would be safer but let's try simple string manipulation
                        # Assuming UUID is at the end
                        full_name = info.name[4:]
                        # Find the last hyphen that starts the UUID
                        # UUID format: 8-4-4-4-12 (36 chars)
                        if len(full_name) > 36:
                            potential_uuid = full_name[-36:]
                            # Check if it looks like a UUID (basic check)
                            if "-" in potential_uuid:
                                name = full_name[:-37]  # Remove -{uuid}
                                self.node_map[info.id] = name
                elif info.name.startswith("Stateful_"):
                    # Handle stateful nodes if we named them "Stateful_{name}"
                    # Format might be "Stateful_{name}" or similar
                    name = info.name
                    # Try to strip UUID if present
                    if len(name) > 36 and "-" in name[-36:]:
                        name = name[:-37]

                    if name.startswith("Stateful_"):
                        self.node_map[info.id] = name[9:]  # Strip "Stateful_"

        def on_exec_operator_start(self, query_id: str, node_id: int) -> None:
            if node_id in self.node_map:
                node_name = self.node_map[node_id]
                self.dispatcher.notify_node_start(node_name, {}, self.ctx)

        def on_exec_emit_stats(
            self, query_id: str, stats: Dict[int, Dict[str, Any]]
        ) -> None:
            pass

        def on_exec_operator_end(self, query_id: str, node_id: int) -> None:
            self.completed_nodes += 1
            if node_id in self.node_map:
                node_name = self.node_map[node_id]

                outputs = {}
                # If in map context, we need to report progress for all items in the batch
                if self.ctx.get("_in_map", False):
                    total_items = self.ctx.get("_map_total_items", 1)
                    outputs["_progress_increment"] = total_items

                # We don't have actual outputs or duration easily here
                self.dispatcher.notify_node_end(node_name, outputs, 0.0, self.ctx)

        def on_exec_end(self, query_id: str) -> None:
            pass
