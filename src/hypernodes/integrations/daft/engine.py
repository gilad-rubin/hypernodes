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

from hypernodes.map_planner import MapPlanner
from hypernodes.protocols import Engine
from hypernodes.integrations.daft.operations import (
    DaftOperation,
    FunctionNodeOperation,
    BatchNodeOperation,
    PipelineNodeOperation,
    SimplePipelineOperation,
    DualNodeOperation,
    ExecutionContext
)
from hypernodes.integrations.daft.codegen import CodeGenContext

if TYPE_CHECKING:
    from hypernodes.pipeline import Pipeline


class DaftEngine(Engine):
    """DaftEngine using modular operations for execution and code generation."""

    def __init__(
        self,
        use_batch_udf: bool = True,
        default_daft_config: Optional[Dict[str, Any]] = None,
    ):
        if not DAFT_AVAILABLE:
            raise ImportError("daft is required. pip install getdaft")
        
        self.use_batch_udf = use_batch_udf
        self.default_daft_config = default_daft_config or {}
        self._is_map_context = False
        self._stateful_inputs = {}

    def run(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        output_name: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute pipeline with 1-row DataFrame."""
        df = self._build_dataframe(pipeline, inputs)
        
        # Execute
        context = ExecutionContext(self, self._stateful_inputs)
        try:
            available_columns = set(df.column_names)
            for node in pipeline.graph.execution_order:
                op = self._create_operation(node)
                df = op.execute(df, available_columns, context)
                if hasattr(node, "output_name"):
                    out = node.output_name
                    if isinstance(out, (list, tuple)):
                        available_columns.update(out)
                    else:
                        available_columns.add(out)
            
            result_df = df.collect()
        finally:
            context.shutdown()

        outputs = self._extract_single_row_outputs(result_df, pipeline)
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
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Execute pipeline over multiple inputs."""
        map_over_list = [map_over] if isinstance(map_over, str) else map_over
        planner = MapPlanner()
        execution_plans = planner.plan_execution(inputs, map_over_list, map_mode)

        if not execution_plans:
            return []

        self._is_map_context = True
        try:
            df = self._build_dataframe_from_plans(pipeline, execution_plans)
            
            context = ExecutionContext(self, self._stateful_inputs)
            try:
                available_columns = set(df.column_names)
                for node in pipeline.graph.execution_order:
                    op = self._create_operation(node)
                    df = op.execute(df, available_columns, context)
                    if hasattr(node, "output_name"):
                        out = node.output_name
                        if isinstance(out, (list, tuple)):
                            available_columns.update(out)
                        else:
                            available_columns.add(out)
                
                result_df = df.collect()
            finally:
                context.shutdown()
        finally:
            self._is_map_context = False

        return self._extract_multi_row_outputs_as_list(result_df, pipeline, output_name)

    def generate_code(self, pipeline: "Pipeline", inputs: Optional[Dict[str, Any]] = None) -> str:
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
            operation_lines.append(f"# Load inputs")
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
            if origin in (list, dict): return False
            
        for param_name in node.root_args:
            param = sig.parameters.get(param_name)
            if param and param.annotation != inspect.Parameter.empty:
                origin = get_origin(param.annotation)
                if origin in (list, dict): return False
        return True

    def _build_dataframe(self, pipeline: "Pipeline", inputs: Dict[str, Any]) -> "daft.DataFrame":
        """Build 1-row Daft DataFrame."""
        df_inputs = {}
        stateful_inputs = {}
        
        for k, v in inputs.items():
            is_stateful = self._is_stateful_object(v)
            print(f"DEBUG: Input {k}, Type: {type(v)}, IsStateful: {is_stateful}")
            if is_stateful:
                stateful_inputs[k] = v
            else:
                df_inputs[k] = v
        
        self._stateful_inputs = stateful_inputs
        print(f"DEBUG: Stateful Inputs: {list(stateful_inputs.keys())}")
        
        if df_inputs:
            return daft.from_pydict({k: [v] for k, v in df_inputs.items()})
        return daft.from_pydict({"__placeholder__": [0]})

    def _build_dataframe_from_plans(self, pipeline: "Pipeline", execution_plans: List[Dict[str, Any]]) -> "daft.DataFrame":
        """Build N-row Daft DataFrame."""
        input_data = {}
        stateful_inputs = {}
        
        for key in execution_plans[0].keys():
            values = [plan[key] for plan in execution_plans]
            if values:
                is_stateful = self._is_stateful_object(values[0])
                print(f"DEBUG: Plan Key {key}, Type: {type(values[0])}, IsStateful: {is_stateful}")
                if is_stateful:
                    # Check constant
                    is_constant = all(v is values[0] for v in values[1:])
                    print(f"DEBUG: Key {key} IsConstant: {is_constant}")
                    if is_constant:
                        stateful_inputs[key] = values[0]
                        continue
            input_data[key] = values
            
        self._stateful_inputs = stateful_inputs
        print(f"DEBUG: Map Stateful Inputs: {list(stateful_inputs.keys())}")
        
        if input_data:
            return daft.from_pydict(input_data)
        return daft.from_pydict({"__placeholder__": [0] * len(execution_plans)})

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

    def _extract_single_row_outputs(self, result_df: "daft.DataFrame", pipeline: "Pipeline") -> Dict[str, Any]:
        output_names = []
        for node in pipeline.graph.execution_order:
            output_name = node.output_name
            if isinstance(output_name, tuple):
                output_names.extend(output_name)
            else:
                output_names.append(output_name)
        
        py_dict = result_df.to_pydict()
        return {k: v[0] for k, v in py_dict.items() if k in output_names}

    def _extract_multi_row_outputs_as_list(
        self,
        result_df: "daft.DataFrame",
        pipeline: "Pipeline",
        output_name: Optional[Union[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        if output_name:
            output_names = [output_name] if isinstance(output_name, str) else output_name
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
                if output_name_key in py_dict:
                    row_dict[output_name_key] = py_dict[output_name_key][row_idx]
            results.append(row_dict)
        return results
