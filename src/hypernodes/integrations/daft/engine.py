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

    @staticmethod
    def _make_serializable_by_value(func):
        """Force cloudpickle to serialize function by value instead of by reference.

        By setting __module__ = "__main__", cloudpickle will serialize the entire
        function bytecode instead of just storing an import path. This allows:
        - Nodes defined inside functions (Modal pattern)
        - Script files that aren't proper packages
        - Functions with closure captures

        Args:
            func: Function to make serializable

        Returns:
            Same function with modified __module__ and __qualname__
        """
        try:
            func.__module__ = "__main__"
            func.__qualname__ = func.__name__
        except (AttributeError, TypeError):
            pass
        return func

    def __init__(
        self,
        use_batch_udf: bool = True,
        default_daft_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize DaftEngine.

        Args:
            use_batch_udf: If True, use batch UDFs for map operations (default: True for performance)
            default_daft_config: Default Daft configuration:
                - batch_size: Batch size for batch UDFs (default: auto-calculated)
                - max_workers: ThreadPoolExecutor workers (default: auto-calculated)
                - max_concurrency: Number of concurrent UDF instances (@daft.cls)
                - use_process: Use process isolation (avoids Python GIL)
                - gpus: Number of GPUs to request
        """
        if not DAFT_AVAILABLE:
            raise ImportError(
                "daft is required for DaftEngineV2. Install with: pip install getdaft"
            )
        self.use_batch_udf = use_batch_udf
        self.default_daft_config = default_daft_config or {}
        self._is_map_context = False  # Track if we're in map operation
        self._map_over_params = set()  # Track which params are mapped over
        self._stateful_wrappers = {}  # Cache for stateful UDF wrappers
        self._stateful_inputs = {}  # Track stateful inputs during execution

        # Cache for auto-calculated values
        self._cpu_count = None

    def _get_cpu_count(self) -> int:
        """Get CPU count (cached)."""
        if self._cpu_count is None:
            import multiprocessing

            self._cpu_count = multiprocessing.cpu_count()
        return self._cpu_count

    def _calculate_max_workers(self, num_items: int) -> int:
        """Calculate optimal max_workers for ThreadPoolExecutor.

        Based on grid search findings:
        - For I/O-bound tasks, ThreadPoolExecutor can benefit from 8-16x CPU cores
        - Optimal value depends on scale:
          - Small (<50): 8x cores
          - Medium (50-200): 16x cores
          - Large (>200): 16x cores

        Args:
            num_items: Number of items to process

        Returns:
            Optimal number of workers
        """
        cpu_count = self._get_cpu_count()

        # Heuristic based on grid search results
        if num_items < 50:
            multiplier = 8  # 8x cores for small batches
        else:
            multiplier = 16  # 16x cores for medium/large batches (best performance)

        return multiplier * cpu_count

    def _calculate_batch_size(self, num_items: int) -> int:
        """Calculate optimal batch_size for batch UDFs.

        Based on grid search findings:
        - Optimal batch size is 1024 for most cases
        - For small datasets, use 64 as minimum effective batch

        Args:
            num_items: Number of items to process

        Returns:
            Optimal batch size
        """
        # Grid search showed 1024 is optimal, with 64 as minimum effective batch
        optimal_batch_size = 1024
        min_effective_batch = 64

        # Scale-based heuristic
        if num_items < 100:
            return min_effective_batch  # Use 64 for small batches
        elif num_items < 500:
            return 256  # Medium batches
        else:
            return optimal_batch_size  # Large batches

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

        # Set context flag for batch UDF usage
        self._is_map_context = True
        self._map_over_params = set(map_over_list)  # Track which params are mapped
        self._num_items = len(execution_plans)  # Store for auto-calculation

        try:
            # Build Daft DataFrame with N rows (lazy)
            df = self._build_dataframe_from_plans(pipeline, execution_plans)

            # Collect (triggers execution)
            result_df = df.collect()
        finally:
            # Reset context
            self._is_map_context = False
            self._map_over_params = set()

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
        # Separate stateful objects from regular inputs
        # Stateful objects (StatefulWrapper) should be:
        # 1. Extracted from DataFrame columns
        # 2. Stored for later use in @daft.cls wrappers
        # 3. Node functions with stateful params get wrapped with @daft.cls

        df_inputs = {}
        stateful_inputs = {}

        for k, v in inputs.items():
            if self._is_stateful_object(v):
                # Store for @daft.cls wrapping
                stateful_inputs[k] = v
            else:
                # Regular input - goes in DataFrame
                df_inputs[k] = v

        # Store stateful inputs for use during node transformation
        self._stateful_inputs = stateful_inputs

        # Create initial 1-row DataFrame with non-stateful inputs only
        if df_inputs:
            df = daft.from_pydict({k: [v] for k, v in df_inputs.items()})
        else:
            # No df inputs - create placeholder DataFrame
            df = daft.from_pydict({"__placeholder__": [0]})

        # Apply each node/pipeline as column transformations
        available_columns = set(df_inputs.keys())
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
        # Separate stateful objects from regular inputs
        # Stateful objects (constants across all plans) should be extracted
        # and used in @daft.cls wrappers, not DataFrame columns

        # Create N-row DataFrame: transpose list of dicts → dict of lists
        input_data = {}
        stateful_inputs = {}

        for key in execution_plans[0].keys():
            values = [plan[key] for plan in execution_plans]

            # Check if this is a constant stateful object
            if values and self._is_stateful_object(values[0]):
                # Check if truly constant (same instance across all plans)
                is_constant = all(v is values[0] for v in values[1:])
                if is_constant:
                    # Extract as stateful
                    stateful_inputs[key] = values[0]
                    continue

            input_data[key] = values

        # Store stateful inputs for use during node transformation
        self._stateful_inputs = stateful_inputs

        if input_data:
            df = daft.from_pydict(input_data)
        else:
            # No df inputs - create placeholder
            df = daft.from_pydict({"__placeholder__": [0] * len(execution_plans)})

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
        # Check if this is a DualNode
        if hasattr(node, "is_dual_node") and node.is_dual_node:
            return self._apply_dual_node_transformation(df, node, available_columns)

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
        """Apply a regular node as a UDF column (batch or row-wise).

        Detects stateful parameters (StatefulWrapper objects) and wraps the
        node function with @daft.cls to capture them.
        """
        import asyncio

        # Check if node has stateful parameters
        # Stateful params are in self._stateful_inputs (set during _build_dataframe)
        node_stateful_params = {}
        dynamic_params = []

        for param in node.root_args:
            if param in self._stateful_inputs:
                # This is a stateful parameter
                node_stateful_params[param] = self._stateful_inputs[param]
            elif param in available_columns:
                # This is a dynamic parameter (from DataFrame)
                dynamic_params.append(param)
            else:
                # Parameter not available - this is an error
                raise ValueError(
                    f"Parameter '{param}' required by node '{node.output_name}' "
                    f"not found in available columns {available_columns} "
                    f"or stateful inputs {list(self._stateful_inputs.keys())}"
                )

        # If we have stateful parameters, wrap with @daft.cls
        if node_stateful_params:
            # Check if there are ANY dynamic params
            if not dynamic_params:
                raise ValueError(
                    f"Node '{node.output_name}' has only stateful parameters {list(node_stateful_params.keys())}. "
                    f"DaftEngine requires at least one dynamic parameter (from DataFrame) for parallel processing. "
                    f"Either use SequentialEngine, or add a dynamic input parameter to the node."
                )

            return self._apply_node_with_stateful_params(
                df, node, dynamic_params, node_stateful_params, available_columns
            )

        # No stateful params - use existing logic
        # Check if function is async - Daft handles async concurrency natively!
        is_async = asyncio.iscoroutinefunction(node.func)

        if is_async:
            # Async functions: Use row-wise @daft.func (Daft provides concurrency)
            # This gives us 37x speedup for I/O-bound tasks!
            # Use smart type inference to handle complex types
            inferred_type = self._infer_daft_return_type(node)

            # Make function serializable for distributed execution
            serializable_func = self._make_serializable_by_value(node.func)

            # ALWAYS use explicit return_dtype to avoid inference errors
            from daft import DataType

            if inferred_type is not None:
                # Use our inferred type
                udf = daft.func(serializable_func, return_dtype=inferred_type)
            else:
                # Fallback to Python type (handles all complex types)
                udf = daft.func(serializable_func, return_dtype=DataType.python())

            input_cols = [daft.col(param) for param in node.root_args]
            df = df.with_column(node.output_name, udf(*input_cols))

            available_columns = available_columns.copy()
            available_columns.add(node.output_name)

            return df, available_columns

        # Determine if we should use batch UDF for sync functions
        use_batch = self._should_use_batch_udf(node)

        if use_batch:
            # Use batch UDF with ThreadPool for parallel execution
            return self._apply_batch_node_transformation(df, node, available_columns)
        else:
            # Use row-wise UDF (required for list/dict types, or single-row execution)
            # Use smart type inference to handle complex types
            inferred_type = self._infer_daft_return_type(node)

            # Make function serializable for distributed execution
            serializable_func = self._make_serializable_by_value(node.func)

            # ALWAYS use explicit return_dtype to avoid inference errors
            from daft import DataType

            if inferred_type is not None:
                # Use our inferred type
                udf = daft.func(serializable_func, return_dtype=inferred_type)
            else:
                # Fallback to Python type (handles all complex types)
                udf = daft.func(serializable_func, return_dtype=DataType.python())

            input_cols = [daft.col(param) for param in node.root_args]
            df = df.with_column(node.output_name, udf(*input_cols))

            available_columns = available_columns.copy()
            available_columns.add(node.output_name)

            return df, available_columns

    def _apply_dual_node_transformation(
        self,
        df: "daft.DataFrame",
        node: Any,
        available_columns: set,
    ) -> tuple["daft.DataFrame", set]:
        """Apply DualNode - choose singular or batch based on context.

        During .run() → uses node.singular with @daft.func
        During .map() → uses node.batch with @daft.func.batch (optimized for Series)
        """
        import asyncio

        # Choose implementation based on context
        func = node.batch if self._is_map_context else node.singular
        use_batch_udf = self._is_map_context

        # Check if function is async
        is_async = asyncio.iscoroutinefunction(func)

        # Use singular function for type inference (canonical signature)
        inferred_type = self._infer_daft_return_type_from_func(node.singular)

        # Make function serializable for distributed execution
        serializable_func = self._make_serializable_by_value(func)

        # ALWAYS use explicit return_dtype to avoid inference errors
        from daft import DataType

        return_dtype = inferred_type if inferred_type is not None else DataType.python()

        if use_batch_udf:
            # Use @daft.func.batch for batch operations (Series → Series)
            batch_kwargs = {"return_dtype": return_dtype}

            # Use engine-level config for batch size
            if "batch_size" in self.default_daft_config:
                batch_kwargs["batch_size"] = self.default_daft_config["batch_size"]
            else:
                # Auto-calculate optimal batch size
                num_items = getattr(self, "_num_items", 100)
                batch_kwargs["batch_size"] = self._calculate_batch_size(num_items)

            # Wrap batch function to handle constant parameters
            # In batch UDFs, ALL parameters come as Series, but constant ones
            # (like encoder, config) should be unwrapped to scalars
            from daft import Series

            def batch_wrapper(*series_args: Series) -> Series:
                """Unwrap constant parameters before calling user's batch function."""
                unwrapped_args = []

                for series_arg in series_args:
                    pylist = series_arg.to_pylist()

                    # Check if this is a constant (all values same)
                    if len(pylist) > 0:
                        first_val = pylist[0]
                        is_constant = all(
                            val is first_val or val == first_val for val in pylist[1:]
                        )

                        if is_constant:
                            # Constant parameter - unwrap to scalar
                            unwrapped_args.append(first_val)
                        else:
                            # Varying parameter - pass as list (user's batch function expects lists)
                            unwrapped_args.append(pylist)
                    else:
                        # Empty series - pass empty list
                        unwrapped_args.append([])

                # Call user's batch function (returns list)
                result = serializable_func(*unwrapped_args)

                # Convert list result back to Series for Daft
                if isinstance(result, list):
                    return Series.from_pylist(result)
                return result  # Already a Series

            # Create decorated batch UDF
            udf = daft.func.batch(**batch_kwargs)(batch_wrapper)
        else:
            # Use @daft.func for row-wise operations (scalar → scalar)
            udf = daft.func(serializable_func, return_dtype=return_dtype)

        # Apply UDF
        input_cols = [daft.col(param) for param in node.root_args]
        df = df.with_column(node.output_name, udf(*input_cols))

        available_columns = available_columns.copy()
        available_columns.add(node.output_name)

        return df, available_columns

    def _infer_daft_return_type_from_func(self, func: Any) -> Optional["daft.DataType"]:
        """Infer Daft DataType from any function's return type annotation.

        Attempts to convert Python type hints to Daft DataTypes:
        - List[T] → DataType.list(DataType.python())
        - dict → DataType.python()
        - Protocol → DataType.python()
        - Simple types (str, int, float) → Daft's type inference

        Args:
            func: Function to infer return type from

        Returns:
            DataType if we can infer it, None to let Daft infer automatically
        """
        import inspect
        from typing import get_origin

        from daft import DataType

        # Get return annotation
        sig = inspect.signature(func)
        return_annotation = sig.return_annotation

        if return_annotation == inspect.Signature.empty:
            # No annotation - let Daft try to infer
            return None

        # Check if it's a stringified annotation (forward reference)
        # This happens when using "from __future__ import annotations" (PEP 563)
        if isinstance(return_annotation, str):
            # Try to evaluate it to get the actual type
            try:
                import typing

                return_annotation = eval(
                    return_annotation, {**typing.__dict__, **func.__globals__}
                )
            except Exception:
                # If evaluation fails, fall back to Python type
                return DataType.python()

        # Check if it's a generic type (List, Dict, etc.)
        origin = get_origin(return_annotation)

        if origin is list:
            # List[T] - use Daft's list type with Python element type
            # This allows explode/list_agg to work while supporting arbitrary element types
            return DataType.list(DataType.python())

        if origin is dict:
            # dict - use Python type
            return DataType.python()

        # Check if it's a Protocol (has __protocol__ attribute)
        if hasattr(return_annotation, "__protocol__"):
            return DataType.python()

        # For other complex types (Pydantic models, custom classes), fall back to Python
        # Check if it's a class (not a built-in type)
        if inspect.isclass(return_annotation):
            # Check if it's a built-in type that Daft can handle
            if return_annotation in (str, int, float, bool):
                return None  # Let Daft infer
            else:
                # Custom class - use Python type
                return DataType.python()

        # Let Daft try to infer for other cases
        return None

    def _infer_daft_return_type(self, node: Any) -> Optional["daft.DataType"]:
        """Infer Daft DataType from node's function return type annotation.

        This is a wrapper around _infer_daft_return_type_from_func for backward compatibility.
        """
        return self._infer_daft_return_type_from_func(node.func)

    def _should_use_batch_udf(self, node: Any) -> bool:
        """Determine if a node should use batch UDF.

        Batch UDFs provide 8x speedup BUT have limitations:
        - Cannot return lists/dicts (Daft limitation: "List casting not implemented for dtype: Python")
        - Cannot receive list/dict inputs from previous nodes

        We use batch UDFs only when:
        1. In map context (multiple rows to process)
        2. Batch UDFs enabled
        3. Function works with simple types (str, int, float, bool) NOT lists/dicts
        """
        import inspect
        from typing import get_origin

        if not (self._is_map_context and self.use_batch_udf):
            return False

        # Check return type hint
        sig = inspect.signature(node.func)
        return_annotation = sig.return_annotation

        if return_annotation != inspect.Signature.empty:
            origin = get_origin(return_annotation)
            # Cannot use batch UDF if function returns list/dict
            if origin in (list, dict):
                return False

        # Check parameter type hints (inputs from previous nodes)
        for param_name in node.root_args:
            param = sig.parameters.get(param_name)
            if param and param.annotation != inspect.Parameter.empty:
                origin = get_origin(param.annotation)
                # Cannot use batch UDF if function receives list/dict
                if origin in (list, dict):
                    return False

        return True

    def _apply_batch_node_transformation(
        self,
        df: "daft.DataFrame",
        node: Any,
        available_columns: set,
    ) -> tuple["daft.DataFrame", set]:
        """Apply node as batch UDF for vectorized processing.

        Note: This should only be called for nodes with simple types.
        Nodes with list/dict types should use row-wise UDFs.
        """
        from daft import DataType, Series

        # Make function serializable for distributed execution
        node_func = self._make_serializable_by_value(node.func)

        # Use Python dtype for simple types (str, int, float, bool)
        # We only call this for simple types - list/dict use row-wise UDFs
        return_dtype = DataType.python()

        # Use engine-level config for batch size (with auto-calculation)
        batch_kwargs = {"return_dtype": return_dtype}
        if "batch_size" in self.default_daft_config:
            batch_kwargs["batch_size"] = self.default_daft_config["batch_size"]
        else:
            # Auto-calculate optimal batch size
            num_items = getattr(
                self, "_num_items", 100
            )  # Default if not in map context
            batch_kwargs["batch_size"] = self._calculate_batch_size(num_items)

        # Wrap the user's function to handle batching with parallel execution
        @daft.func.batch(**batch_kwargs)
        def batch_udf(*series_args: Series) -> Series:
            import concurrent.futures

            # Convert Series to Python types
            # For mapped params: keep as list
            # For constant params: extract scalar value
            python_args = []

            for i, series in enumerate(series_args):
                pylist = series.to_pylist()

                # Check if this is a constant (all values same)
                # Use id() comparison for objects to avoid expensive comparisons
                first_val = pylist[0]
                is_constant = all(
                    val is first_val or val == first_val for val in pylist[1:]
                )

                if is_constant:
                    # Constant parameter - use scalar
                    python_args.append(first_val)
                else:
                    # Varying parameter - pass as list for iteration
                    python_args.append(pylist)

            # Find the varying parameter to determine batch size
            first_list_idx = None
            for i, arg in enumerate(python_args):
                if isinstance(arg, list):
                    first_list_idx = i
                    break

            if first_list_idx is None:
                # All constant - call once
                results = [node_func(*python_args)]
            else:
                # Parallel execution using ThreadPoolExecutor
                # This gives us ~10x speedup for I/O-bound tasks!
                n_items = len(python_args[first_list_idx])

                def process_item(idx: int):
                    """Process a single item from the batch."""
                    call_args = []
                    for i, arg in enumerate(python_args):
                        if isinstance(arg, list):
                            call_args.append(arg[idx])
                        else:
                            call_args.append(arg)
                    return node_func(*call_args)

                # Use ThreadPoolExecutor for parallel execution
                # Get max_workers from config or auto-calculate
                if "max_workers" in self.default_daft_config:
                    max_workers = self.default_daft_config["max_workers"]
                else:
                    # Auto-calculate optimal max_workers based on grid search findings
                    max_workers = self._calculate_max_workers(n_items)

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    results = list(executor.map(process_item, range(n_items)))

            # Return as Series with Python objects
            return Series.from_pylist(results)

        # Apply batch UDF
        input_cols = [daft.col(param) for param in node.root_args]
        df = df.with_column(node.output_name, batch_udf(*input_cols))

        available_columns = available_columns.copy()
        available_columns.add(node.output_name)

        return df, available_columns

    def _is_stateful_object(self, obj: Any) -> bool:
        """Check if object is marked as stateful.

        Stateful objects are marked with @stateful decorator, which sets
        the __hypernode_stateful__ attribute.

        Args:
            obj: Object to check

        Returns:
            True if object is stateful, False otherwise
        """
        return (
            hasattr(obj, "__class__")
            and hasattr(obj.__class__, "__hypernode_stateful__")
            and obj.__class__.__hypernode_stateful__ is True
        )

    def _apply_node_with_stateful_params(
        self,
        df: "daft.DataFrame",
        node: Any,
        dynamic_params: list,
        stateful_params: Dict[str, Any],
        available_columns: set,
    ) -> tuple["daft.DataFrame", set]:
        """Apply node with stateful parameters using @daft.cls wrapper.

        Creates a @daft.cls wrapper that:
        1. __init__: Reconstructs stateful objects from original class + init args
        2. __call__ or async method: Calls node function with both stateful and dynamic params

        Args:
            df: DataFrame
            node: Node to apply
            dynamic_params: Parameters from DataFrame columns
            stateful_params: Parameters that are stateful objects
            available_columns: Available columns

        Returns:
            (transformed_df, updated_available_columns)
        """
        import asyncio

        # Check if function is async
        is_async = asyncio.iscoroutinefunction(node.func)

        # Infer return type
        inferred_type = self._infer_daft_return_type(node)
        from daft import DataType

        return_dtype = inferred_type if inferred_type is not None else DataType.python()

        # Make node function serializable
        serializable_func = self._make_serializable_by_value(node.func)

        # Extract original classes and init args from StatefulWrapper objects
        stateful_reconstructors = {}
        for param_name, wrapper in stateful_params.items():
            stateful_reconstructors[param_name] = {
                "class": self._make_serializable_by_value(wrapper._original_class),
                "args": wrapper._init_args,
                "kwargs": wrapper._init_kwargs,
            }

        # Get engine-level config for @daft.cls
        config = self.default_daft_config
        cls_kwargs = {}
        if "max_concurrency" in config:
            cls_kwargs["max_concurrency"] = config["max_concurrency"]
        if "use_process" in config:
            cls_kwargs["use_process"] = config["use_process"]
        if "gpus" in config:
            cls_kwargs["gpus"] = config["gpus"]

        # Build @daft.cls wrapper (different for async vs sync)
        all_params = node.root_args

        if is_async:
            # Async version
            @daft.cls(**cls_kwargs)
            class StatefulNodeWrapper:
                def __init__(self):
                    # Reconstruct stateful objects from original class + init args
                    # This happens ONCE per worker
                    self._stateful_objects = {}
                    for param_name, recon in stateful_reconstructors.items():
                        original_class = recon["class"]
                        init_args = recon["args"]
                        init_kwargs = recon["kwargs"]
                        # Create instance
                        self._stateful_objects[param_name] = original_class(
                            *init_args, **init_kwargs
                        )

                @daft.method(return_dtype=return_dtype)
                async def __call__(self, *args):
                    # Combine stateful and dynamic parameters
                    arg_iter = iter(args)
                    kwargs = {}

                    for param in all_params:
                        if param in self._stateful_objects:
                            kwargs[param] = self._stateful_objects[param]
                        else:
                            kwargs[param] = next(arg_iter)

                    # Call the async node function
                    return await serializable_func(**kwargs)
        else:
            # Sync version
            @daft.cls(**cls_kwargs)
            class StatefulNodeWrapper:
                def __init__(self):
                    # Reconstruct stateful objects from original class + init args
                    # This happens ONCE per worker
                    self._stateful_objects = {}
                    for param_name, recon in stateful_reconstructors.items():
                        original_class = recon["class"]
                        init_args = recon["args"]
                        init_kwargs = recon["kwargs"]
                        # Create instance
                        self._stateful_objects[param_name] = original_class(
                            *init_args, **init_kwargs
                        )

                @daft.method(return_dtype=return_dtype)
                def __call__(self, *args):
                    # Combine stateful and dynamic parameters
                    arg_iter = iter(args)
                    kwargs = {}

                    for param in all_params:
                        if param in self._stateful_objects:
                            kwargs[param] = self._stateful_objects[param]
                        else:
                            kwargs[param] = next(arg_iter)

                    # Call the node function
                    return serializable_func(**kwargs)

        # Create wrapper instance
        wrapper = StatefulNodeWrapper()

        # Apply wrapper UDF to DataFrame
        # Note: dynamic_params is always non-empty here (we error out in the caller if not)
        input_cols = [daft.col(param) for param in dynamic_params]
        df = df.with_column(node.output_name, wrapper(*input_cols))

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

        # Use Daft's built-in monotonically increasing id to avoid eager materialization
        df = df._add_monotonically_increasing_id(row_id_col)

        # Preserve the original list column before exploding (so it's available for other nodes)
        original_list_col = f"__original_{map_over_col}__"
        df = df.with_column(original_list_col, daft.col(map_over_col))

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
            if (
                col not in final_output_names
                and col != row_id_col
                and col != original_list_col
            ):
                agg_exprs.append(daft.col(col).any_value().alias(col))

        # Preserve the original list column (take first value since it's the same for all rows in the group)
        agg_exprs.append(
            daft.col(original_list_col).any_value().alias(original_list_col)
        )

        df = df_grouped.agg(*agg_exprs)

        # Restore the original list column to its original name
        df = df.with_column(map_over_col, daft.col(original_list_col))

        # Remove temporary columns (cleanup)
        select_exprs = [
            df[col]
            for col in df.column_names
            if col != row_id_col and col != original_list_col
        ]
        df = df.select(*select_exprs)

        # Update available columns
        available_columns = available_columns.copy()
        available_columns.update(final_output_names)
        # The original map_over column is now available again
        available_columns.add(map_over_col)

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
