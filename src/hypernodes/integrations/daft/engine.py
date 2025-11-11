"""Minimal DaftEngine built on top of small, testable components."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from hypernodes.engine import Engine
from hypernodes.pipeline import Pipeline, PipelineNode

from . import engine_legacy as legacy_engine

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
        *,
        debug: bool = False,
        python_return_strategy: str = "auto",
        code_generation_mode: bool = False,
        show_plan: bool = False,
        force_spawn_method: bool = True,
    ):
        if not DAFT_AVAILABLE:  # pragma: no cover - enforced by import
            raise ImportError(
                "Daft is not installed. Install Daft with `pip install daft`."
            )

        self.collect = collect
        self.output_mode = output_mode
        self.debug = debug
        self.python_return_strategy = python_return_strategy
        self.code_generation_mode = code_generation_mode
        self._legacy_runtime = None
        self._legacy_kwargs = dict(
            collect=collect,
            show_plan=show_plan,
            debug=debug,
            python_return_strategy=python_return_strategy,
            force_spawn_method=force_spawn_method,
            code_generation_mode=False,
        )

        if self.code_generation_mode:
            self._legacy_codegen = legacy_engine.DaftEngine(
                **{**self._legacy_kwargs, "code_generation_mode": True}
            )
            return

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
        if self.code_generation_mode:
            return self._legacy_codegen.run(
                pipeline, inputs, output_name=output_name, _ctx=_ctx
            )
        if self._requires_legacy_runtime(pipeline):
            return self._get_legacy_runtime().run(
                pipeline, inputs, output_name=output_name, _ctx=_ctx
            )

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
        if self.code_generation_mode:
            return self._legacy_codegen.map(
                pipeline, items, inputs, output_name=output_name, _ctx=_ctx
            )
        if self._requires_legacy_runtime(pipeline):
            return self._get_legacy_runtime().map(
                pipeline, items, inputs, output_name=output_name, _ctx=_ctx
            )

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
        if self.code_generation_mode:
            return self._legacy_codegen.map_columnar(
                pipeline=pipeline,
                varying_inputs=varying_inputs,
                fixed_inputs=fixed_inputs,
                output_name=output_name,
                return_format=return_format,
                _ctx=_ctx,
            )
        if self._requires_legacy_runtime(pipeline):
            return self._get_legacy_runtime().map_columnar(
                pipeline=pipeline,
                varying_inputs=varying_inputs,
                fixed_inputs=fixed_inputs,
                output_name=output_name,
                return_format=return_format,
                _ctx=_ctx,
            )

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
        if self.code_generation_mode:
            return self._legacy_codegen.shutdown(wait=wait)
        if self._legacy_runtime is not None:
            return self._legacy_runtime.shutdown(wait=wait)
        return None

    # ------------------------------------------------------------------ #
    # Code generation helpers
    # ------------------------------------------------------------------ #
    def get_generated_code(self) -> str:
        if self.code_generation_mode:
            return self._legacy_codegen.get_generated_code()
        return "Code generation mode not enabled."

    @property
    def generated_code(self) -> str:
        if self.code_generation_mode:
            return getattr(self._legacy_codegen, "generated_code", "")
        return ""

    @property
    def _udf_definitions(self):
        if self.code_generation_mode:
            return getattr(self._legacy_codegen, "_udf_definitions", {})
        return {}

    @property
    def _imports(self):
        if self.code_generation_mode:
            return getattr(self._legacy_codegen, "_imports", set())
        return set()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _requires_legacy_runtime(self, pipeline: Pipeline) -> bool:
        for node in pipeline.nodes:
            if isinstance(node, PipelineNode) and node.map_over:
                return True
        return False

    def _get_legacy_runtime(self):
        if self._legacy_runtime is None:
            self._legacy_runtime = legacy_engine.DaftEngine(**self._legacy_kwargs)
        return self._legacy_runtime


def fix_script_classes_for_modal() -> None:
    """Backward-compatible entry point for legacy serialization helper."""

    return legacy_engine.fix_script_classes_for_modal()
