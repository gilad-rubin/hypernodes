"""Columnar map execution helpers for DaftEngine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from hypernodes.pipeline import Pipeline

from .compiler import PipelineCompiler, CompilationResult


class MapOperationHandler:
    """Execute pipeline maps using a columnar compilation strategy."""

    def __init__(self, compiler: PipelineCompiler):
        self._compiler = compiler

    def execute(
        self,
        pipeline: Pipeline,
        varying_inputs: Dict[str, List[Any]],
        fixed_inputs: Dict[str, Any],
        output_name: Union[str, List[str], None],
    ) -> CompilationResult:
        """Compile the pipeline for columnar map execution."""
        row_count = self._infer_row_count(varying_inputs)
        combined_inputs: Dict[str, Any] = {}
        combined_inputs.update(fixed_inputs)
        combined_inputs.update(varying_inputs)
        return self._compiler.compile(
            pipeline=pipeline,
            inputs=combined_inputs,
            output_name=output_name,
            row_count=row_count,
        )

    def _infer_row_count(self, varying_inputs: Dict[str, List[Any]]) -> int:
        if not varying_inputs:
            return 0
        lengths = {key: len(values) for key, values in varying_inputs.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(
                "Map inputs must have the same length when using columnar execution. "
                f"Got lengths: {lengths}"
            )
        return next(iter(unique_lengths))
