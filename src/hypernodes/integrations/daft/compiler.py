"""Pipeline → Daft DataFrame compiler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

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
    ):
        if not DAFT_AVAILABLE:  # pragma: no cover - enforced by engine
            raise ImportError("Daft is not installed. Install with `pip install daft`.")

        self._node_converter = node_converter
        self._stateful_builder = stateful_builder

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
        plan = pipeline._compute_required_nodes(output_name)
        nodes = plan if plan is not None else pipeline.execution_order

        df_inputs, stateful_inputs = self._prepare_inputs(inputs, row_count)
        df = daft.from_pydict(df_inputs)

        for node in nodes:
            df = self._node_converter.convert(node, df, stateful_inputs)

        all_columns = self._ordered_output_names(nodes)
        requested = self._resolve_requested_columns(output_name, all_columns)

        if requested:
            df = df.select(*requested)

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
            if self._stateful_builder.is_stateful(value):
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
