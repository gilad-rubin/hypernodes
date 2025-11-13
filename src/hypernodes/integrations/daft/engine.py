"""Clean DaftEngine implementation using @daft.func.

Key Design:
- Each node → @daft.func UDF
- All operations are lazy (build computation graph, then collect)
- For .run(): 1-row DataFrame
- For .map(): N-row DataFrame
- Nested pipelines: explode → transform → aggregate pattern
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

try:
    import daft

    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False

from hypernodes.map_planner import MapPlanner
from hypernodes.protocols import Engine

if TYPE_CHECKING:
    from hypernodes.pipeline import Pipeline


class DaftEngine(Engine):
    """Simple DaftEngine using @daft.func for lazy execution.

    Architecture:
    - Each node becomes a @daft.func UDF column
    - All transformations are lazy (no execution until .collect())
    - Nested pipelines use explode → transform → aggregate pattern

    Example:
        >>> engine = DaftEngine()
        >>> pipeline = Pipeline(nodes=[...], engine=engine)
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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute pipeline using Daft with 1-row DataFrame.

        Args:
            pipeline: Pipeline to execute
            inputs: Input values
            output_name: Optional output name(s) to filter
            **kwargs: Additional arguments

        Returns:
            Dictionary of output values
        """
        # Build Daft DataFrame with node UDFs (lazy)
        df = self._build_dataframe(pipeline, inputs)

        # Collect (triggers actual execution)
        result_df = df.collect()

        # Extract outputs from single-row DataFrame
        outputs = self._extract_single_row_outputs(result_df, pipeline)

        # Filter outputs if requested
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
        """Execute pipeline over multiple inputs using Daft N-row DataFrame.

        Args:
            pipeline: Pipeline to execute
            inputs: Input values (some are lists)
            map_over: Parameter name(s) to map over
            map_mode: "zip" or "product"
            output_name: Optional output name(s) to filter
            **kwargs: Additional arguments

        Returns:
            List of output dictionaries, one per item
        """
        # Use MapPlanner to generate execution plans
        map_over_list = [map_over] if isinstance(map_over, str) else map_over
        planner = MapPlanner()
        execution_plans = planner.plan_execution(inputs, map_over_list, map_mode)

        if not execution_plans:
            # Empty map - return empty list
            return []

        # Build Daft DataFrame with N rows (lazy)
        df = self._build_dataframe_from_plans(pipeline, execution_plans)

        # Collect (triggers execution)
        result_df = df.collect()

        # Extract outputs as list of dicts
        outputs = self._extract_multi_row_outputs_as_list(
            result_df, pipeline, output_name
        )

        return outputs

    def _build_dataframe(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
    ) -> "daft.DataFrame":
        """Build 1-row Daft DataFrame by chaining UDF columns (lazy)."""
        # Create initial 1-row DataFrame with input columns
        df = daft.from_pydict({k: [v] for k, v in inputs.items()})

        # Apply each node/pipeline as column transformations
        available_columns = set(inputs.keys())
        for node in pipeline.graph.execution_order:
            df, available_columns = self._apply_node_transformation(
                df, node, available_columns
            )

        return df

    def _build_dataframe_from_plans(
        self,
        pipeline: "Pipeline",
        execution_plans: List[Dict[str, Any]],
    ) -> "daft.DataFrame":
        """Build N-row Daft DataFrame from execution plans (lazy)."""
        # Create N-row DataFrame: transpose list of dicts → dict of lists
        input_data = {}
        for key in execution_plans[0].keys():
            input_data[key] = [plan[key] for plan in execution_plans]

        df = daft.from_pydict(input_data)

        # Apply each node/pipeline as column transformations
        available_columns = set(input_data.keys())
        for node in pipeline.graph.execution_order:
            df, available_columns = self._apply_node_transformation(
                df, node, available_columns
            )

        return df

    def _apply_node_transformation(
        self,
        df: "daft.DataFrame",
        node: Any,
        available_columns: set,
    ) -> tuple["daft.DataFrame", set]:
        """Apply transformation for a node (regular or pipeline).

        Returns: (transformed_df, updated_available_columns)
        """
        # Check if this is a PipelineNode with map_over
        if hasattr(node, "pipeline") and hasattr(node, "map_over") and node.map_over:
            return self._apply_mapped_pipeline_transformation(
                df, node, available_columns
            )

        # Check if this is a PipelineNode without map_over (simple nesting)
        if hasattr(node, "pipeline"):
            return self._apply_simple_pipeline_transformation(
                df, node, available_columns
            )

        # Regular node with a function
        if hasattr(node, "func"):
            return self._apply_simple_node_transformation(df, node, available_columns)

        raise ValueError(f"Unknown node type: {type(node)}")

    def _apply_simple_node_transformation(
        self,
        df: "daft.DataFrame",
        node: Any,
        available_columns: set,
    ) -> tuple["daft.DataFrame", set]:
        """Apply a regular node as a UDF column."""
        # Wrap function with @daft.func
        udf = daft.func(node.func)

        # Get input columns for this node
        input_cols = [daft.col(param) for param in node.root_args]

        # Apply UDF to create new column
        df = df.with_column(node.output_name, udf(*input_cols))

        # Update available columns
        available_columns = available_columns.copy()
        available_columns.add(node.output_name)

        return df, available_columns

    def _apply_simple_pipeline_transformation(
        self,
        df: "daft.DataFrame",
        pipeline_node: Any,
        available_columns: set,
    ) -> tuple["daft.DataFrame", set]:
        """Apply a nested pipeline (without map_over) by recursively applying nodes.

        This handles input/output mapping between outer and inner pipelines.
        """
        inner_pipeline = pipeline_node.pipeline
        input_mapping = pipeline_node.input_mapping or {}
        output_mapping = pipeline_node.output_mapping or {}

        # Apply input mapping (rename columns)
        if input_mapping:
            rename_exprs = []
            inner_available = set()

            for outer_name, inner_name in input_mapping.items():
                if outer_name in available_columns:
                    rename_exprs.append(df[outer_name].alias(inner_name))
                    inner_available.add(inner_name)

            # Keep unmapped columns
            for col in available_columns:
                if col not in input_mapping:
                    rename_exprs.append(df[col])
                    inner_available.add(col)

            df = df.select(*rename_exprs)
        else:
            inner_available = available_columns.copy()

        # Apply inner pipeline nodes
        for inner_node in inner_pipeline.graph.execution_order:
            df, inner_available = self._apply_node_transformation(
                df, inner_node, inner_available
            )

        # Apply output mapping (rename outputs)
        if output_mapping:
            inner_outputs = {
                node.output_name for node in inner_pipeline.graph.execution_order
            }

            rename_exprs = []
            for col in df.column_names:
                if col in inner_outputs:
                    outer_name = output_mapping.get(col, col)
                    rename_exprs.append(df[col].alias(outer_name))
                else:
                    rename_exprs.append(df[col])

            df = df.select(*rename_exprs)

        # Update available columns with mapped outputs
        output_names = [
            output_mapping.get(node.output_name, node.output_name)
            for node in inner_pipeline.graph.execution_order
        ]
        available_columns = available_columns.copy()
        available_columns.update(output_names)

        return df, available_columns

    def _apply_mapped_pipeline_transformation(
        self,
        df: "daft.DataFrame",
        pipeline_node: Any,
        available_columns: set,
    ) -> tuple["daft.DataFrame", set]:
        """Apply nested pipeline with map_over using explode → transform → aggregate.

        This is the clean implementation of the pattern:
        1. Add row_id for tracking
        2. Explode list column into rows
        3. Apply input mapping
        4. Apply inner pipeline transformations
        5. Apply output mapping
        6. Aggregate back into lists by row_id

        All operations are lazy.
        """
        inner_pipeline = pipeline_node.pipeline
        input_mapping = pipeline_node.input_mapping or {}
        output_mapping = pipeline_node.output_mapping or {}
        map_over = pipeline_node.map_over

        # Validate map_over is a single column
        if isinstance(map_over, list):
            if len(map_over) != 1:
                raise ValueError(
                    f"DaftEngineV2 only supports map_over with a single column. "
                    f"Got: {map_over}"
                )
            map_over_col = map_over[0]
        else:
            map_over_col = map_over

        if map_over_col not in input_mapping:
            raise ValueError(
                f"map_over column '{map_over_col}' not found in input_mapping. "
                f"Expected one of: {list(input_mapping.keys())}"
            )

        # Step 1: Add row_id for tracking during aggregation
        row_id_col = "__daft_row_id__"

        # We need unique row IDs, but daft.lit(0) gives all rows the same ID
        # Materialize to assign proper row IDs (needed for double-nested map_over)
        df_collected = df.to_pydict()
        num_rows = len(list(df_collected.values())[0]) if df_collected else 0
        df_collected[row_id_col] = list(range(num_rows))
        df = daft.from_pydict(df_collected)

        # Step 2: Explode list column into multiple rows
        df = df.explode(daft.col(map_over_col))

        # Step 3: Apply input mapping (rename columns for inner pipeline)
        if input_mapping:
            rename_exprs = []
            inner_available = set()

            for outer_name, inner_name in input_mapping.items():
                if outer_name in available_columns or outer_name == map_over_col:
                    rename_exprs.append(df[outer_name].alias(inner_name))
                    inner_available.add(inner_name)

            # Keep unmapped columns (except row_id which we track separately)
            for col in df.column_names:
                if (
                    col not in input_mapping
                    and col != map_over_col
                    and col != row_id_col
                ):
                    rename_exprs.append(df[col])
                    inner_available.add(col)

            # Keep row_id
            rename_exprs.append(df[row_id_col])

            df = df.select(*rename_exprs)
        else:
            inner_available = available_columns.copy()

        # Step 4: Apply inner pipeline transformations
        for inner_node in inner_pipeline.graph.execution_order:
            df, inner_available = self._apply_node_transformation(
                df, inner_node, inner_available
            )

        # Step 5: Apply output mapping
        inner_outputs = [
            node.output_name for node in inner_pipeline.graph.execution_order
        ]
        final_output_names = [output_mapping.get(name, name) for name in inner_outputs]

        if output_mapping:
            rename_exprs = []
            for col in df.column_names:
                if col in inner_outputs:
                    outer_name = output_mapping.get(col, col)
                    rename_exprs.append(df[col].alias(outer_name))
                elif col != row_id_col:
                    rename_exprs.append(df[col])

            # Keep row_id for aggregation
            rename_exprs.append(df[row_id_col])

            df = df.select(*rename_exprs)

        # Step 6: Aggregate back into lists by row_id
        # Group by row_id
        df_grouped = df.groupby(daft.col(row_id_col))

        # Aggregate outputs into lists
        agg_exprs = []
        for output_name in final_output_names:
            agg_exprs.append(daft.col(output_name).list_agg().alias(output_name))

        # Also preserve any non-output columns (take first value)
        for col in df.column_names:
            if col not in final_output_names and col != row_id_col:
                agg_exprs.append(daft.col(col).any_value().alias(col))

        df = df_grouped.agg(*agg_exprs)

        # Remove row_id column (cleanup)
        select_exprs = [df[col] for col in df.column_names if col != row_id_col]
        df = df.select(*select_exprs)

        # Update available columns
        available_columns = available_columns.copy()
        available_columns.update(final_output_names)

        return df, available_columns

    def _extract_single_row_outputs(
        self, result_df: "daft.DataFrame", pipeline: "Pipeline"
    ) -> Dict[str, Any]:
        """Extract outputs from single-row DataFrame."""
        # Get output column names from pipeline nodes
        output_names = []
        for node in pipeline.graph.execution_order:
            output_name = node.output_name
            if isinstance(output_name, tuple):
                output_names.extend(output_name)
            else:
                output_names.append(output_name)

        # Convert to Python dict (single row → extract first element)
        py_dict = result_df.to_pydict()
        return {k: v[0] for k, v in py_dict.items() if k in output_names}

    def _extract_multi_row_outputs_as_list(
        self,
        result_df: "daft.DataFrame",
        pipeline: "Pipeline",
        output_name: Optional[Union[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Extract outputs from N-row DataFrame as list of dicts.

        Converts from N-row DataFrame format to list of dicts format,
        matching the SequentialEngine behavior.
        """
        # Determine which outputs to return
        if output_name:
            output_names = (
                [output_name] if isinstance(output_name, str) else output_name
            )
        else:
            # Extract output names, handling tuples from PipelineNodes
            output_names = []
            for node in pipeline.graph.execution_order:
                node_output_name = node.output_name
                if isinstance(node_output_name, tuple):
                    output_names.extend(node_output_name)
                else:
                    output_names.append(node_output_name)

        # Convert to Python dict (DataFrame rows)
        py_dict = result_df.to_pydict()

        # Get number of rows
        num_rows = len(list(py_dict.values())[0]) if py_dict else 0

        # Convert from {col: [val1, val2, ...]} to [{col: val1}, {col: val2}, ...]
        results = []
        for row_idx in range(num_rows):
            row_dict = {}
            for output_name_key in output_names:
                if output_name_key in py_dict:
                    row_dict[output_name_key] = py_dict[output_name_key][row_idx]
            results.append(row_dict)

        return results
