"""Conversion of HyperNodes nodes into Daft DataFrame operations."""

from __future__ import annotations

import ast
import inspect
import itertools
import textwrap
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

try:
    import daft
    from daft import DataFrame as DaftDataFrame
    DAFT_AVAILABLE = True
except ImportError:  # pragma: no cover - handled by caller
    DAFT_AVAILABLE = False
    DaftDataFrame = Any  # type: ignore[misc,assignment]

from hypernodes.node import Node
from hypernodes.pipeline import PipelineNode

from .stateful_udf import StatefulUDFBuilder

if TYPE_CHECKING:
    from .code_generator import CodeGenerator


class NodeConverter:
    """Convert nodes to Daft column expressions.

    This class owns all logic for turning a HyperNodes ``Node`` (or
    ``PipelineNode``) into Daft transformations. The converter only mutates the
    provided DataFrame, keeping orchestration logic elsewhere.
    """

    def __init__(
        self, 
        stateful_builder: StatefulUDFBuilder,
        code_generator: Optional["CodeGenerator"] = None,
    ):
        if not DAFT_AVAILABLE:  # pragma: no cover - enforced by engine
            raise ImportError("Daft is not installed. Install with `pip install daft`.")

        self._stateful_builder = stateful_builder
        self._temp_counter = itertools.count()
        self._dict_getters: Dict[str, Any] = {}
        self._code_generator = code_generator

    def convert(
        self,
        node: Node | PipelineNode,
        df: DaftDataFrame,
        stateful_inputs: Dict[str, Any],
    ) -> DaftDataFrame:
        """Convert a node into Daft operations and append result columns."""
        if isinstance(node, PipelineNode):
            return self._convert_pipeline_node(node, df, stateful_inputs)
        return self._convert_function_node(node, df, stateful_inputs)

    # --------------------------------------------------------------------- #
    # Function nodes
    # --------------------------------------------------------------------- #
    def _convert_function_node(
        self,
        node: Node,
        df: DaftDataFrame,
        stateful_inputs: Dict[str, Any],
    ) -> DaftDataFrame:
        params = list(node.parameters)
        stateful_params = {
            param: stateful_inputs[param]
            for param in params
            if param in stateful_inputs
        }
        dynamic_params = [param for param in params if param not in stateful_params]

        missing = [param for param in dynamic_params if param not in df.column_names]
        if missing:
            formatted = ", ".join(missing)
            raise ValueError(f"Missing columns for node '{node.output_name}': {formatted}")

        if not dynamic_params:
            constant_value = node.func(**stateful_params)
            if self._code_generator:
                value_str = self._code_generator.format_value_for_code(constant_value)
                self._code_generator.add_operation(
                    f'df = df.with_column("{node.output_name}", daft.lit({value_str}))'
                )
            return df.with_column(node.output_name, daft.lit(constant_value))

        if stateful_params:
            udf = self._stateful_builder.build(node.func, stateful_params, dynamic_params)
            if self._code_generator:
                # Generate stateful UDF code
                udf_name = self._generate_stateful_udf_code(
                    node.func, stateful_params, dynamic_params, node.output_name
                )
                args_str = ", ".join([f'df["{p}"]' for p in dynamic_params])
                self._code_generator.add_operation(
                    f'df = df.with_column("{node.output_name}", {udf_name}({args_str}))'
                )
        else:
            udf = self._build_stateless_udf(node.func)
            if self._code_generator:
                # Generate stateless UDF code
                udf_name = self._generate_stateless_udf_code(node.func, params)
                args_str = ", ".join([f'df["{p}"]' for p in dynamic_params])
                self._code_generator.add_operation(
                    f'df = df.with_column("{node.output_name}", {udf_name}({args_str}))'
                )

        args = [df[param] for param in dynamic_params]
        return df.with_column(node.output_name, udf(*args))

    # --------------------------------------------------------------------- #
    # Pipeline nodes
    # --------------------------------------------------------------------- #
    def _convert_pipeline_node(
        self,
        node: PipelineNode,
        df: DaftDataFrame,
        stateful_inputs: Dict[str, Any],
    ) -> DaftDataFrame:
        if node.map_over:
            return self._convert_mapped_pipeline_node(node, df, stateful_inputs)

        params = list(node.parameters)
        stateful_params = {
            param: stateful_inputs[param]
            for param in params
            if param in stateful_inputs
        }
        dynamic_params = [param for param in params if param not in stateful_params]

        missing = [param for param in dynamic_params if param not in df.column_names]
        if missing:
            formatted = ", ".join(missing)
            raise ValueError(
                f"Missing columns for PipelineNode '{node.name or node.pipeline}': {formatted}"
            )

        temp_column = f"__pipeline_result_{next(self._temp_counter)}"
        udf = self._build_pipeline_udf(node, dynamic_params, stateful_params)
        args = [df[param] for param in dynamic_params]
        
        if self._code_generator:
            # Generate code for nested pipeline UDF call
            udf_name = self._generate_pipeline_udf_code(node, dynamic_params, stateful_params)
            args_str = ", ".join([f'df["{p}"]' for p in dynamic_params])
            self._code_generator.add_operation(
                f'df = df.with_column("{temp_column}", {udf_name}({args_str}))'
            )
        
        df = df.with_column(temp_column, udf(*args))

        for output_name in _normalize_outputs(node.output_name):
            getter = self._dict_getters.get(output_name)
            if getter is None:
                getter = self._build_dict_getter(output_name)
                self._dict_getters[output_name] = getter
            
            if self._code_generator:
                getter_name = self._generate_dict_getter_code(output_name)
                self._code_generator.add_operation(
                    f'df = df.with_column("{output_name}", {getter_name}(df["{temp_column}"]))'
                )
            
            df = df.with_column(output_name, getter(df[temp_column]))

        remaining = [name for name in df.column_names if name != temp_column]
        
        if self._code_generator:
            remaining_str = ", ".join([f'"{name}"' for name in remaining])
            self._code_generator.add_operation(f"df = df.select({remaining_str})")
        
        return df.select(*remaining)

    def _convert_mapped_pipeline_node(
        self,
        node: PipelineNode,
        df: DaftDataFrame,
        stateful_inputs: Dict[str, Any],
    ) -> DaftDataFrame:
        params = list(node.parameters)
        stateful_params = {
            param: stateful_inputs[param]
            for param in params
            if param in stateful_inputs
        }
        dynamic_params = [param for param in params if param not in stateful_params]

        missing = [param for param in dynamic_params if param not in df.column_names]
        if missing:
            formatted = ", ".join(missing)
            raise ValueError(
                f"Missing columns for PipelineNode '{node.name or node.pipeline}': {formatted}"
            )

        temp_column = f"__pipeline_result_{next(self._temp_counter)}"
        udf = self._build_mapped_pipeline_udf(node, dynamic_params, stateful_params)
        args = [df[param] for param in dynamic_params]
        
        if self._code_generator:
            # Generate code for mapped pipeline (explode/groupby pattern)
            udf_name = self._generate_mapped_pipeline_code(
                node, dynamic_params, stateful_params, temp_column
            )
            args_str = ", ".join([f'df["{p}"]' for p in dynamic_params])
            self._code_generator.add_operation(
                f'# Map over: {", ".join(node.map_over or [])} (uses explode/groupby)'
            )
            self._code_generator.add_operation(
                f'df = df.with_column("{temp_column}", {udf_name}({args_str}))'
            )
        
        df = df.with_column(temp_column, udf(*args))

        for output_name in _normalize_outputs(node.output_name):
            getter = self._dict_getters.get(output_name)
            if getter is None:
                getter = self._build_dict_getter(output_name)
                self._dict_getters[output_name] = getter
            
            if self._code_generator:
                getter_name = self._generate_dict_getter_code(output_name)
                self._code_generator.add_operation(
                    f'df = df.with_column("{output_name}", {getter_name}(df["{temp_column}"]))'
                )
            
            df = df.with_column(output_name, getter(df[temp_column]))

        remaining = [name for name in df.column_names if name != temp_column]
        
        if self._code_generator:
            remaining_str = ", ".join([f'"{name}"' for name in remaining])
            self._code_generator.add_operation(f"df = df.select({remaining_str})")
        
        return df.select(*remaining)

    def _build_pipeline_udf(
        self,
        node: PipelineNode,
        dynamic_params: Sequence[str],
        stateful_params: Dict[str, Any],
    ):
        pipeline = node.pipeline
        input_mapping = node.input_mapping or {}
        output_mapping = node.output_mapping or {}

        def _pipeline_runner(*args):
            values = dict(zip(dynamic_params, args))
            values.update(stateful_params)

            inner_inputs: Dict[str, Any] = {}
            for outer_name, value in values.items():
                inner_name = input_mapping.get(outer_name, outer_name)
                inner_inputs[inner_name] = value

            results = pipeline.run(inputs=inner_inputs)
            mapped = {}
            for inner_name, value in results.items():
                outer_name = output_mapping.get(inner_name, inner_name)
                mapped[outer_name] = value
            return mapped

        return daft.func(_pipeline_runner, return_dtype=daft.DataType.python())

    def _build_mapped_pipeline_udf(
        self,
        node: PipelineNode,
        dynamic_params: Sequence[str],
        stateful_params: Dict[str, Any],
    ):
        pipeline = node.pipeline
        input_mapping = node.input_mapping or {}
        output_mapping = node.output_mapping or {}
        map_over = list(node.map_over or [])

        def _pipeline_runner(*args):
            values = dict(zip(dynamic_params, args))
            values.update(stateful_params)

            inner_inputs: Dict[str, Any] = {}
            for outer_name, value in values.items():
                inner_name = input_mapping.get(outer_name, outer_name)
                inner_inputs[inner_name] = value

            inner_map_over = [
                input_mapping.get(name, name)
                for name in map_over
            ]
            mapped = pipeline.map(
                inputs=inner_inputs,
                map_over=inner_map_over,
                map_mode="zip",
                return_format="python",
            )

            remapped = {}
            for inner_name, value in mapped.items():
                outer_name = output_mapping.get(inner_name, inner_name)
                remapped[outer_name] = value
            return remapped

        return daft.func(_pipeline_runner, return_dtype=daft.DataType.python())

    def _build_dict_getter(self, key: str):
        @daft.func(return_dtype=daft.DataType.python())
        def getter(payload):
            if payload is None:
                return None
            return payload.get(key)

        return getter

    def _build_stateless_udf(self, func: Any):
        return daft.func(func, return_dtype=daft.DataType.python())

    # --------------------------------------------------------------------- #
    # Code generation helpers
    # --------------------------------------------------------------------- #
    def _generate_stateless_udf_code(
        self, 
        func: Any, 
        params: List[str],
    ) -> str:
        """Generate code for a stateless UDF.
        
        Returns:
            UDF name to use in with_column calls
        """
        if not self._code_generator:
            return ""
        
        func_name = func.__name__
        udf_name = self._code_generator.generate_udf_name(func_name)
        
        lines = []
        param_list = ", ".join([f"{p}: Any" for p in params])
        
        self._code_generator.add_import("typing", "Any")
        
        lines.append("@daft.func(return_dtype=daft.DataType.python())")
        lines.append(f"def {udf_name}({param_list}):")
        
        # Try to get function body
        body = self._extract_function_body(func)
        if body:
            body = textwrap.dedent(body)
            for line in body.split("\n"):
                lines.append(f"    {line}" if line else "")
        else:
            lines.append("    # Source code not available")
            lines.append(f"    return {func_name}({', '.join(params)})")
        
        self._code_generator.add_udf_definition("\n".join(lines))
        return udf_name

    def _generate_stateful_udf_code(
        self,
        func: Any,
        stateful_params: Dict[str, Any],
        dynamic_params: List[str],
        output_name: str,
    ) -> str:
        """Generate code for a stateful UDF.
        
        Returns:
            UDF name to use in with_column calls
        """
        if not self._code_generator:
            return ""
        
        # Track stateful inputs
        for name, obj in stateful_params.items():
            self._code_generator.add_stateful_input(name, obj)
        
        func_name = func.__name__
        udf_name = self._code_generator.generate_udf_name(func_name)
        class_name = f"{func_name.title().replace('_', '')}Wrapper"
        
        self._code_generator.add_import("typing", "Any")
        
        lines = []
        lines.append("@daft.cls(use_process=False)")
        lines.append(f"class {class_name}:")
        
        # __init__ with stateful parameters
        stateful_params_str = ", ".join([f"{k}: Any" for k in stateful_params.keys()])
        lines.append(f"    def __init__(self, {stateful_params_str}):")
        for k in stateful_params.keys():
            lines.append(f"        self.{k} = {k}")
        lines.append("")
        
        # __call__ method
        method_params = ", ".join([f"{p}: Any" for p in dynamic_params])
        lines.append("    @daft.method(return_dtype=daft.DataType.python())")
        lines.append(f"    def __call__(self, {method_params}):")
        
        # Generate function call
        all_params = list(stateful_params.keys()) + dynamic_params
        call_args = ", ".join(
            [f"self.{p}" if p in stateful_params else p for p in all_params]
        )
        lines.append(f"        return {func_name}({call_args})")
        lines.append("")
        
        # Instantiate wrapper
        init_args = ", ".join([f"{k}={k}" for k in stateful_params.keys()])
        wrapper_instance = f"{udf_name}_wrapper"
        lines.append(f"{wrapper_instance} = {class_name}({init_args})")
        lines.append(f"{udf_name} = {wrapper_instance}")
        
        # Add helper function to show signature (for documentation)
        helper_name = f"{func_name}_signature"
        if method_params:
            lines.append("")
            lines.append(f"def {helper_name}({method_params}):")
        else:
            lines.append("")
            lines.append(f"def {helper_name}():")
        lines.append('    """Helper to illustrate how to invoke this stateful UDF."""')
        call_args_list = ", ".join(dynamic_params)
        if call_args_list:
            lines.append(f"    return {udf_name}({call_args_list})")
        else:
            lines.append(f"    return {udf_name}()")
        
        self._code_generator.add_udf_definition("\n".join(lines))
        return udf_name

    def _extract_function_body(self, func: Any) -> Optional[str]:
        """Extract and return the dedented body of a function."""
        try:
            raw_source = inspect.getsource(func)
        except (OSError, TypeError):
            return None
        
        dedented = textwrap.dedent(raw_source)
        try:
            module = ast.parse(dedented)
        except SyntaxError:
            return None
        
        func_nodes = [
            node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if not func_nodes:
            return None
        
        func_node = func_nodes[0]
        if not func_node.body:
            return None
        
        lines = dedented.split("\n")
        start = max(func_node.body[0].lineno - 1, 0)
        end_node = func_node.body[-1]
        end = getattr(end_node, "end_lineno", end_node.lineno)
        end = min(end, len(lines))
        body = "\n".join(lines[start:end])
        return body

    def _generate_pipeline_udf_code(
        self,
        node: PipelineNode,
        dynamic_params: Sequence[str],
        stateful_params: Dict[str, Any],
    ) -> str:
        """Generate code for a pipeline UDF (nested pipeline execution).
        
        Returns:
            UDF name to use in with_column calls
        """
        if not self._code_generator:
            return ""
        
        pipeline_name = node.name or "pipeline"
        udf_name = self._code_generator.generate_udf_name(f"{pipeline_name}_runner")
        
        self._code_generator.add_import("typing", "Any")
        
        lines = []
        lines.append("@daft.func(return_dtype=daft.DataType.python())")
        param_list = ", ".join([f"{p}: Any" for p in dynamic_params])
        lines.append(f"def {udf_name}({param_list}):")
        lines.append("    # This UDF runs a nested pipeline")
        lines.append("    # Note: Nested pipelines are executed row-by-row (may be slow)")
        
        # Add warning comment about performance
        lines.append("    # TODO: Consider flattening pipeline structure for better performance")
        
        # For now, just indicate that the pipeline runs
        lines.append(f"    # Executes pipeline: {pipeline_name}")
        lines.append("    # Input mapping: " + str(node.input_mapping))
        lines.append("    # Output mapping: " + str(node.output_mapping))
        lines.append("    pass  # Actual implementation requires pipeline.run()")
        
        self._code_generator.add_udf_definition("\n".join(lines))
        return udf_name

    def _generate_dict_getter_code(self, key: str) -> str:
        """Generate code for a dictionary getter UDF.
        
        Returns:
            UDF name to use for extracting values from dict results
        """
        if not self._code_generator:
            return ""
        
        udf_name = self._code_generator.generate_udf_name(f"get_{key}")
        
        self._code_generator.add_import("typing", "Any")
        
        lines = []
        lines.append("@daft.func(return_dtype=daft.DataType.python())")
        lines.append(f"def {udf_name}(payload: Any):")
        lines.append("    if payload is None:")
        lines.append("        return None")
        lines.append(f'    return payload.get("{key}")')
        
        self._code_generator.add_udf_definition("\n".join(lines))
        return udf_name

    def _generate_mapped_pipeline_code(
        self,
        node: PipelineNode,
        dynamic_params: Sequence[str],
        stateful_params: Dict[str, Any],
        temp_column: str,
    ) -> str:
        """Generate code for a mapped pipeline (with explode/groupby pattern).
        
        Returns:
            UDF name to use in with_column calls
        """
        if not self._code_generator:
            return ""
        
        pipeline_name = node.name or "mapped_pipeline"
        udf_name = self._code_generator.generate_udf_name(f"{pipeline_name}_mapper")
        
        self._code_generator.add_import("typing", "Any")
        
        lines = []
        lines.append("@daft.func(return_dtype=daft.DataType.python())")
        param_list = ", ".join([f"{p}: Any" for p in dynamic_params])
        lines.append(f"def {udf_name}({param_list}):")
        lines.append("    # This UDF maps a pipeline over list inputs")
        lines.append(f"    # Map over parameters: {list(node.map_over or [])}")
        lines.append("    # Input mapping: " + str(node.input_mapping))
        lines.append("    # Output mapping: " + str(node.output_mapping))
        lines.append("    #")
        lines.append("    # PERFORMANCE WARNING:")
        lines.append("    # This creates an explode → process → groupby cycle")
        lines.append("    # Consider using @daft.func.batch for 10-100x speedup")
        lines.append("    pass  # Actual implementation requires pipeline.map()")
        
        self._code_generator.add_udf_definition("\n".join(lines))
        return udf_name


def _normalize_outputs(outputs: Any) -> List[str]:
    if outputs is None:
        return []
    if isinstance(outputs, tuple):
        return [output for output in outputs if output]
    return [outputs]
