"""Conversion of HyperNodes nodes into Daft DataFrame operations."""

from __future__ import annotations

import itertools
from typing import Any, Dict, Iterable, List, Sequence

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


class NodeConverter:
    """Convert nodes to Daft column expressions.

    This class owns all logic for turning a HyperNodes ``Node`` (or
    ``PipelineNode``) into Daft transformations. The converter only mutates the
    provided DataFrame, keeping orchestration logic elsewhere.
    """

    def __init__(self, stateful_builder: StatefulUDFBuilder):
        if not DAFT_AVAILABLE:  # pragma: no cover - enforced by engine
            raise ImportError("Daft is not installed. Install with `pip install daft`.")

        self._stateful_builder = stateful_builder
        self._temp_counter = itertools.count()
        self._dict_getters: Dict[str, Any] = {}

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

        if stateful_params:
            udf = self._stateful_builder.build(node.func, stateful_params, dynamic_params)
        else:
            udf = self._build_stateless_udf(node.func)

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
        df = df.with_column(temp_column, udf(*args))

        for output_name in _normalize_outputs(node.output_name):
            getter = self._dict_getters.get(output_name)
            if getter is None:
                getter = self._build_dict_getter(output_name)
                self._dict_getters[output_name] = getter
            df = df.with_column(output_name, getter(df[temp_column]))

        remaining = [name for name in df.column_names if name != temp_column]
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
        df = df.with_column(temp_column, udf(*args))

        for output_name in _normalize_outputs(node.output_name):
            getter = self._dict_getters.get(output_name)
            if getter is None:
                getter = self._build_dict_getter(output_name)
                self._dict_getters[output_name] = getter
            df = df.with_column(output_name, getter(df[temp_column]))

        remaining = [name for name in df.column_names if name != temp_column]
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


def _normalize_outputs(outputs: Any) -> List[str]:
    if outputs is None:
        return []
    if isinstance(outputs, tuple):
        return [output for output in outputs if output]
    return [outputs]
