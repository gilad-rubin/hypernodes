"""Pipeline → Daft DataFrame compiler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

try:
    import daft
    from daft import DataFrame as DaftDataFrame
    DAFT_AVAILABLE = True
except ImportError:  # pragma: no cover - handled upstream
    DAFT_AVAILABLE = False
    DaftDataFrame = Any  # type: ignore[misc,assignment]

from hypernodes.pipeline import Pipeline

from .node_converter import NodeConverter
from .stateful_udf import StatefulUDFBuilder

if TYPE_CHECKING:
    from .code_generator import CodeGenerator


@dataclass
class CompilationResult:
    """DataFrame plus ordered list of output columns."""

    dataframe: DaftDataFrame
    columns: List[str]


class PipelineCompiler:
    """Compile HyperNodes pipelines into Daft DataFrame operations."""

    def __init__(
        self,
        node_converter: NodeConverter,
        stateful_builder: StatefulUDFBuilder,
        code_generator: Optional["CodeGenerator"] = None,
    ):
        if not DAFT_AVAILABLE:  # pragma: no cover - enforced by engine
            raise ImportError("Daft is not installed. Install with `pip install daft`.")

        self._node_converter = node_converter
        self._stateful_builder = stateful_builder
        self._code_generator = code_generator

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def compile(
        self,
        pipeline: Pipeline,
        inputs: Dict[str, Any],
        output_name: Union[str, List[str], None] = None,
        row_count: Optional[int] = None,
    ) -> CompilationResult:
        """Compile a pipeline for the provided inputs."""
        plan = pipeline.graph.get_required_nodes(output_name)
        nodes = plan if plan is not None else pipeline.graph.execution_order

        df_inputs, stateful_inputs = self._prepare_inputs(inputs, row_count)
        
        # Generate DataFrame creation code if in code generation mode
        if self._code_generator:
            self._generate_dataframe_creation(df_inputs, stateful_inputs)
        
        df = daft.from_pydict(df_inputs)

        for node in nodes:
            df = self._node_converter.convert(node, df, stateful_inputs)

        all_columns = self._ordered_output_names(nodes)
        requested = self._resolve_requested_columns(output_name, all_columns)

        if requested:
            if self._code_generator:
                self._code_generator.add_operation("")
                self._code_generator.add_operation("# Select output columns")
                columns_str = ", ".join([f'"{col}"' for col in requested])
                self._code_generator.add_operation(f"df = df.select({columns_str})")
            df = df.select(*requested)
        
        if self._code_generator:
            self._code_generator.add_operation("")
            self._code_generator.add_operation("# Collect result")
            self._code_generator.add_operation("result = df.collect()")
            self._code_generator.add_operation("print(result.to_pydict())")

        return CompilationResult(dataframe=df, columns=requested)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _prepare_inputs(
        self,
        inputs: Dict[str, Any],
        row_count: Optional[int],
    ) -> tuple[Dict[str, List[Any]], Dict[str, Any]]:
        expected_rows = row_count if row_count is not None else 1
        df_inputs: Dict[str, List[Any]] = {}
        stateful_inputs: Dict[str, Any] = {}

        dynamic_row_count = None if row_count is None else row_count

        for name, value in inputs.items():
            # Check if it's a stateful object (has __daft_hint__)
            if self._stateful_builder.is_stateful(value):
                stateful_inputs[name] = value
                continue
            
            # For code generation, also treat complex objects as stateful
            # (they can't be easily serialized in generated code)
            if self._code_generator and not self._is_simple_type(value):
                stateful_inputs[name] = value
                continue

            column, dynamic_row_count = self._ensure_column(value, dynamic_row_count)
            df_inputs[name] = column

        if dynamic_row_count is None:
            dynamic_row_count = expected_rows

        if not df_inputs:
            df_inputs["__row__"] = [0] * dynamic_row_count

        return df_inputs, stateful_inputs

    def _ensure_column(
        self,
        value: Any,
        row_count: Optional[int],
    ) -> tuple[List[Any], Optional[int]]:
        if isinstance(value, list):
            if row_count is None:
                return value, len(value)
            if len(value) != row_count:
                raise ValueError(
                    f"Inconsistent column lengths: expected {row_count}, got {len(value)}"
                )
            return value, row_count

        if row_count is None:
            return [value], 1

        return [value] * row_count, row_count

    def _ordered_output_names(self, nodes: Sequence[Any]) -> List[str]:
        ordered: List[str] = []
        for node in nodes:
            outputs = node.output_name
            if isinstance(outputs, tuple):
                ordered.extend([name for name in outputs if name])
            elif outputs:
                ordered.append(outputs)
        return ordered

    def _resolve_requested_columns(
        self,
        output_name: Union[str, List[str], None],
        available: Sequence[str],
    ) -> List[str]:
        if output_name is None:
            return list(available)

        requested = [output_name] if isinstance(output_name, str) else list(output_name)

        missing = [name for name in requested if name not in available]
        if missing:
            available_str = ", ".join(sorted(set(available)))
            missing_str = ", ".join(missing)
            raise ValueError(
                f"Requested outputs not found: {missing_str}. "
                f"Available outputs: {available_str}"
            )

        return requested

    def _generate_dataframe_creation(
        self,
        df_inputs: Dict[str, List[Any]],
        stateful_inputs: Dict[str, Any],
    ) -> None:
        """Generate code for DataFrame creation."""
        if not self._code_generator:
            return
        
        # Track stateful inputs
        for name, obj in stateful_inputs.items():
            self._code_generator.add_stateful_input(name, obj)
        
        self._code_generator.add_operation("# Create DataFrame with input data")
        self._code_generator.add_operation("df = daft.from_pydict({")
        
        for key, value in df_inputs.items():
            value_repr = self._code_generator.format_value_for_code(value)
            self._code_generator.add_operation(f'    "{key}": {value_repr},')
        
        self._code_generator.add_operation("})")
        self._code_generator.add_operation("")
        
        # Add stateful objects as columns using daft.lit()
        if stateful_inputs:
            self._code_generator.add_operation("# Add stateful objects as columns")
            self._code_generator.add_operation(
                "# Note: These objects must be initialized before running this code"
            )
            for key in stateful_inputs.keys():
                self._code_generator.add_operation(f'df = df.with_column("{key}", daft.lit({key}))')
            self._code_generator.add_operation("")

    def _is_simple_type(self, value: Any) -> bool:
        """Check if value is a simple type that can be serialized."""
        if value is None:
            return True
        if isinstance(value, (bool, int, float, str, bytes)):
            return True
        if isinstance(value, list):
            return all(self._is_simple_type(item) for item in value)
        if isinstance(value, tuple):
            return all(self._is_simple_type(item) for item in value)
        if isinstance(value, dict):
            return all(
                self._is_simple_type(k) and self._is_simple_type(v)
                for k, v in value.items()
            )
        return False

