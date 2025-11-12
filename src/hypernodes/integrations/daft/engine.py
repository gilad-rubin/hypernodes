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

from .code_generator import CodeGenerator
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
        self.code_generation_mode = code_generation_mode
        self._legacy_runtime = None
        self._legacy_kwargs = dict(
            collect=collect,
            show_plan=show_plan,
            debug=debug,
            python_return_strategy="auto",
            force_spawn_method=force_spawn_method,
            code_generation_mode=False,
        )

        # Initialize code generator if in code generation mode
        self._code_generator = CodeGenerator(debug=debug) if code_generation_mode else None

        # Initialize core components
        self._stateful_builder = StatefulUDFBuilder()
        self._node_converter = NodeConverter(
            self._stateful_builder,
            code_generator=self._code_generator
        )
        self._compiler = PipelineCompiler(
            self._node_converter,
            self._stateful_builder,
            code_generator=self._code_generator
        )
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
        # Only use legacy runtime if NOT in code generation mode
        if not self.code_generation_mode and self._requires_legacy_runtime(pipeline):
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
        if self._legacy_runtime is not None:
            # Legacy runtime doesn't have shutdown
            pass
        return None

    # ------------------------------------------------------------------ #
    # Code generation helpers
    # ------------------------------------------------------------------ #
    def get_generated_code(self) -> str:
        """Return complete executable Daft code that exactly matches the pipeline execution."""
        if not self.code_generation_mode:
            return "Code generation mode not enabled."
        
        if not self._code_generator:
            return "Code generator not initialized."
        
        return self._code_generator.generate_code()

    @property
    def generated_code(self) -> str:
        """Property accessor for generated code (for compatibility)."""
        return self.get_generated_code()

    @property
    def _udf_definitions(self):
        """Access UDF definitions (for compatibility with tests)."""
        if self._code_generator:
            return self._code_generator._udf_definitions
        return []

    @property
    def _imports(self):
        """Access imports (for compatibility with tests)."""
        if self._code_generator:
            return self._code_generator._imports
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
