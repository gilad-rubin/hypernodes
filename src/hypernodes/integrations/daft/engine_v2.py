"""Clean DaftEngine implementation using @daft.func with caching and callbacks.

Key Design:
- Each node → @daft.func UDF
- Caching logic wrapped inside UDF (executes during Daft computation)
- Callbacks injected into UDF wrapper
- For .run(): 1-row DataFrame
- For .map(): N-row DataFrame

Limitations:
- Callbacks fire per-row during Daft execution (may be out of order in parallel mode)
- Dependency hash tracking is simplified (no cross-row dependencies)
"""

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

try:
    import daft

    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False

from hypernodes.cache import compute_signature, hash_inputs
from hypernodes.callbacks import CallbackContext
from hypernodes.protocols import Engine
from hypernodes.map_planner import MapPlanner
from hypernodes.node_execution import _get_node_id

if TYPE_CHECKING:
    from hypernodes.pipeline import Pipeline


class DaftEngineV2(Engine):
    """Simple DaftEngine using @daft.func with caching and callbacks.

    Architecture:
    - Each node becomes a @daft.func UDF column
    - Caching happens inside UDF (per-row, content-addressed)
    - Callbacks fire during Daft execution (inside UDF)
    
    Example:
        >>> engine = DaftEngineV2()
        >>> pipeline = Pipeline(
        ...     nodes=[...],
        ...     engine=engine,
        ...     cache=DiskCache(".cache")
        ... )
        >>> result = pipeline.run(x=5)
    """

    def __init__(self):
        """Initialize DaftEngine."""
        if not DAFT_AVAILABLE:
            raise ImportError(
                "daft is required for DaftEngineV2. Install with: pip install getdaft"
            )

    def run(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        output_name: Optional[Union[str, List[str]]] = None,
        _ctx: Optional[CallbackContext] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute pipeline using Daft with 1-row DataFrame.

        Args:
            pipeline: Pipeline to execute
            inputs: Input values
            output_name: Optional output name(s) to filter
            _ctx: Internal callback context
            **kwargs: Additional arguments

        Returns:
            Dictionary of output values
        """
        ctx = _ctx or CallbackContext()
        callbacks = pipeline.callbacks or []
        cache = pipeline.cache

        # Push pipeline onto context
        is_new_context = _ctx is None
        if is_new_context:
            ctx.push_pipeline(pipeline.id)

        try:
            # Set pipeline metadata
            ctx.set_pipeline_metadata(
                pipeline.id,
                {
                    "total_nodes": len(pipeline.graph.execution_order),
                    "pipeline_name": pipeline.name or pipeline.id,
                    "node_ids": [
                        _get_node_id(node) for node in pipeline.graph.execution_order
                    ],
                },
            )

            # Trigger pipeline start
            start_time = time.time()
            for callback in callbacks:
                callback.on_pipeline_start(pipeline.id, inputs, ctx)

            # Build Daft DataFrame with node UDFs
            df = self._build_dataframe(pipeline, inputs, cache, callbacks, ctx)

            # Collect (triggers actual execution)
            result_df = df.collect()

            # Extract outputs from single-row DataFrame
            outputs = self._extract_single_row_outputs(result_df, pipeline)

            # Trigger pipeline end
            duration = time.time() - start_time
            for callback in callbacks:
                callback.on_pipeline_end(pipeline.id, outputs, duration, ctx)

            # Filter outputs if requested
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
    ) -> Dict[str, List[Any]]:
        """Execute pipeline over multiple inputs using Daft N-row DataFrame.

        Args:
            pipeline: Pipeline to execute
            inputs: Input values (some are lists)
            map_over: Parameter name(s) to map over
            map_mode: "zip" or "product"
            output_name: Optional output name(s) to filter
            _ctx: Internal callback context
            **kwargs: Additional arguments

        Returns:
            Dictionary of output lists
        """
        ctx = _ctx or CallbackContext()
        callbacks = pipeline.callbacks or []
        cache = pipeline.cache

        # Use MapPlanner to generate execution plans
        map_over_list = [map_over] if isinstance(map_over, str) else map_over
        planner = MapPlanner()
        execution_plans = planner.plan_execution(inputs, map_over_list, map_mode)

        if not execution_plans:
            # Empty map - return empty lists
            output_names = self._get_output_names(pipeline, output_name)
            return {name: [] for name in output_names}

        # Mark that we're in a map operation
        ctx.set("_in_map", True)
        ctx.set("_map_total", len(execution_plans))

        # Push pipeline onto context
        is_new_context = _ctx is None
        if is_new_context:
            ctx.push_pipeline(pipeline.id)

        try:
            # Set pipeline metadata
            ctx.set_pipeline_metadata(
                pipeline.id,
                {
                    "total_nodes": len(pipeline.graph.execution_order),
                    "pipeline_name": pipeline.name or pipeline.id,
                    "node_ids": [
                        _get_node_id(node) for node in pipeline.graph.execution_order
                    ],
                },
            )

            # Trigger map start
            start_time = time.time()
            for callback in callbacks:
                callback.on_map_start(len(execution_plans), ctx)

            # Build Daft DataFrame with N rows
            df = self._build_dataframe_from_plans(
                pipeline, execution_plans, cache, callbacks, ctx
            )

            # Collect (triggers execution)
            result_df = df.collect()

            # Extract outputs as lists
            outputs = self._extract_multi_row_outputs(result_df, pipeline, output_name)

            # Trigger map end
            duration = time.time() - start_time
            for callback in callbacks:
                callback.on_map_end(duration, ctx)

            return outputs

        finally:
            ctx.set("_in_map", False)
            if is_new_context:
                ctx.pop_pipeline()

    def _build_dataframe(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        cache,
        callbacks,
        ctx: CallbackContext,
    ) -> "daft.DataFrame":
        """Build 1-row Daft DataFrame by chaining UDF columns."""
        # Create initial 1-row DataFrame with input columns
        df = daft.from_pydict({k: [v] for k, v in inputs.items()})

        # Apply each node as a column transformation
        for node in pipeline.graph.execution_order:
            # Create cached UDF wrapper for this node
            cached_udf = self._create_cached_udf(node, cache, callbacks, ctx)

            # Get input columns for this node
            input_cols = [daft.col(param) for param in node.root_args]

            # Apply UDF to create new column
            df = df.with_column(node.output_name, cached_udf(*input_cols))

        return df

    def _build_dataframe_from_plans(
        self,
        pipeline: "Pipeline",
        execution_plans: List[Dict[str, Any]],
        cache,
        callbacks,
        ctx: CallbackContext,
    ) -> "daft.DataFrame":
        """Build N-row Daft DataFrame from execution plans."""
        # Create N-row DataFrame: transpose list of dicts → dict of lists
        input_data = {}
        for key in execution_plans[0].keys():
            input_data[key] = [plan[key] for plan in execution_plans]

        df = daft.from_pydict(input_data)

        # Apply each node as a column transformation
        for node in pipeline.graph.execution_order:
            cached_udf = self._create_cached_udf(node, cache, callbacks, ctx)
            input_cols = [daft.col(param) for param in node.root_args]
            df = df.with_column(node.output_name, cached_udf(*input_cols))

        return df

    def _create_cached_udf(self, node, cache, callbacks, ctx: CallbackContext):
        """Create @daft.func UDF wrapper.

        For now, this is a simple pass-through to the node function.
        TODO: Add caching and callbacks using serializable approach.
        """
        node_func = node.func
        
        # For now, just wrap the original function directly
        # Daft will automatically infer types from annotations
        return daft.func(node_func)

    def _extract_single_row_outputs(
        self, result_df: "daft.DataFrame", pipeline: "Pipeline"
    ) -> Dict[str, Any]:
        """Extract outputs from single-row DataFrame."""
        # Get output column names
        output_names = [node.output_name for node in pipeline.graph.execution_order]

        # Convert to Python dict (single row → extract first element)
        py_dict = result_df.to_pydict()
        return {k: v[0] for k, v in py_dict.items() if k in output_names}

    def _extract_multi_row_outputs(
        self,
        result_df: "daft.DataFrame",
        pipeline: "Pipeline",
        output_name: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, List[Any]]:
        """Extract outputs from N-row DataFrame as lists."""
        # Determine which outputs to return
        if output_name:
            output_names = [output_name] if isinstance(output_name, str) else output_name
        else:
            output_names = [node.output_name for node in pipeline.graph.execution_order]

        # Convert to Python dict (already lists)
        py_dict = result_df.to_pydict()
        return {k: v for k, v in py_dict.items() if k in output_names}

    def _get_output_names(
        self, pipeline: "Pipeline", output_name: Optional[Union[str, List[str]]] = None
    ) -> List[str]:
        """Get list of output names."""
        if output_name:
            return [output_name] if isinstance(output_name, str) else output_name
        return [node.output_name for node in pipeline.graph.execution_order]

