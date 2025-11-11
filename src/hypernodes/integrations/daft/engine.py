"""Minimal DaftEngine built on top of small, testable components."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from hypernodes.engine import Engine
from hypernodes.pipeline import Pipeline

try:
    import daft  # noqa: F401  # Ensures Daft is available
    DAFT_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime
    DAFT_AVAILABLE = False

from .compiler import PipelineCompiler
from .map_operations import MapOperationHandler
from .node_converter import NodeConverter
from .output_materializer import OutputMaterializer
from .stateful_udf import StatefulUDFBuilder


class DaftEngine(Engine):
    """Compile HyperNodes pipelines into Daft DataFrame operations."""

    def __init__(
        self,
        collect: bool = True,
        output_mode: str = "dict",
    ):
        if not DAFT_AVAILABLE:  # pragma: no cover - enforced by import
            raise ImportError(
                "Daft is not installed. Install Daft with `pip install daft`."
            )

        self.collect = collect
        self._stateful_builder = StatefulUDFBuilder()
        self._node_converter = NodeConverter(self._stateful_builder)
        self._compiler = PipelineCompiler(self._node_converter, self._stateful_builder)
        self._materializer = OutputMaterializer(mode=output_mode)
        self._map_handler = MapOperationHandler(self._compiler)

    # ------------------------------------------------------------------ #
    # Engine API
    # ------------------------------------------------------------------ #
    def run(
        self,
        pipeline: Pipeline,
        inputs: Dict[str, Any],
        output_name: Union[str, List[str], None] = None,
        _ctx: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Compile to Daft, execute, and materialize to Python scalars."""
        compilation = self._compiler.compile(
            pipeline=pipeline,
            inputs=inputs,
            output_name=output_name,
        )

        df = compilation.dataframe
        if self.collect:
            df = df.collect()

        materialized = self._materializer.materialize(df, squeeze=True)
        return {name: materialized[name] for name in compilation.columns}

    def map(
        self,
        pipeline: Pipeline,
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        output_name: Union[str, List[str], None] = None,
        _ctx: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Fallback map implementation when columnar path is unavailable."""
        results: List[Dict[str, Any]] = []
        for item in items:
            merged = {**inputs, **item}
            results.append(self.run(pipeline, merged, output_name=output_name, _ctx=_ctx))
        return results

    def map_columnar(
        self,
        pipeline: Pipeline,
        varying_inputs: Dict[str, List[Any]],
        fixed_inputs: Dict[str, Any],
        output_name: Union[str, List[str], None] = None,
        return_format: str = "python",
        _ctx: Optional[Any] = None,
    ):
        """Fast-path map execution that stays in Daft as long as possible."""
        compilation = self._map_handler.execute(
            pipeline=pipeline,
            varying_inputs=varying_inputs,
            fixed_inputs=fixed_inputs,
            output_name=output_name,
        )

        if return_format == "daft":
            return compilation.dataframe
        if return_format != "python":
            raise ValueError(
                f"Unsupported return_format '{return_format}'. "
                "DaftEngine supports 'python' and 'daft'."
            )

        df = compilation.dataframe
        df = df.collect()
        return self._materializer.materialize(df, squeeze=False)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def shutdown(self, wait: bool = True) -> None:  # pragma: no cover - no resources
        """Provided for parity with other engines (no-op)."""
        return None


def fix_script_classes_for_modal() -> None:
    """Backward-compatible entry point for legacy serialization helper."""
    from . import engine_legacy  # Imported lazily to avoid heavy dependency

    return engine_legacy.fix_script_classes_for_modal()
