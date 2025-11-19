"""Daft operations for HyperNodes engine.

Defines the strategy for executing different types of nodes and generating their code.
"""

import abc
import asyncio
import concurrent.futures
import inspect
from typing import Any, Dict, List, Optional, Set

try:
    import daft
    from daft import DataType, Series
except ImportError:
    daft = None
    DataType = None
    Series = None

from hypernodes.integrations.daft.codegen import CodeGenContext


class ExecutionContext:
    """Context passed to operations during execution."""

    def __init__(self, engine, stateful_inputs: Dict[str, Any]):
        self.engine = engine  # Reference back to engine for recursive calls
        self.stateful_inputs = stateful_inputs
        # Shared executor for batch operations to avoid overhead
        # Use engine's config
        max_workers = engine.default_daft_config.get("max_workers")
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def shutdown(self):
        """Clean up resources."""
        self.executor.shutdown(wait=False)


class DaftOperation(abc.ABC):
    """Abstract base class for Daft operations."""

    @abc.abstractmethod
    def execute(
        self,
        df: "daft.DataFrame",
        available_columns: Set[str],
        context: ExecutionContext,
    ) -> "daft.DataFrame":
        """Execute the operation on the DataFrame."""
        pass

    @abc.abstractmethod
    def generate_code(
        self, df_var: str, available_columns: Set[str], context: CodeGenContext
    ) -> str:
        """Generate code for this operation."""
        pass

    def _make_serializable_by_value(self, func):
        """Force cloudpickle to serialize function by value."""
        try:
            func.__module__ = "__main__"
            func.__qualname__ = func.__name__
        except (AttributeError, TypeError):
            pass
        return func

    def _infer_daft_return_type(self, func: Any) -> Optional["daft.DataType"]:
        """Infer Daft DataType from function return annotation."""
        from typing import get_origin

        sig = inspect.signature(func)
        return_annotation = sig.return_annotation

        if return_annotation == inspect.Signature.empty:
            return None

        # Handle string annotations (PEP 563)
        if isinstance(return_annotation, str):
            try:
                import typing

                # Try to evaluate the string annotation
                # This is critical for Pydantic models defined in the same file
                return_annotation = eval(
                    return_annotation, {**typing.__dict__, **func.__globals__}
                )
            except Exception:
                # Fallback if evaluation fails
                return DataType.python()

        origin = get_origin(return_annotation)
        if origin is list:
            return DataType.list(DataType.python())
        if origin is dict:
            return DataType.python()

        if hasattr(return_annotation, "__protocol__"):
            return DataType.python()

        if inspect.isclass(return_annotation):
            if return_annotation in (str, int, float, bool):
                return None  # Let Daft infer
            return DataType.python()

        return None


class FunctionNodeOperation(DaftOperation):
    """Handles standard scalar nodes (sync and async)."""

    def __init__(self, node: Any):
        self.node = node

    def execute(
        self,
        df: "daft.DataFrame",
        available_columns: Set[str],
        context: ExecutionContext,
    ) -> "daft.DataFrame":
        # Check for stateful params
        stateful_params = {}
        dynamic_params = []

        for param in self.node.root_args:
            if param in context.stateful_inputs:
                stateful_params[param] = context.stateful_inputs[param]
            elif param in available_columns:
                dynamic_params.append(param)
            else:
                # It might be an optional param or something else, but for now assume error
                # In original engine this raised ValueError
                raise ValueError(
                    f"Parameter '{param}' not found for node '{self.node.output_name}'"
                )

        # Unwrap to get the real function for async check
        # Use the node's cached property if available (for Node objects)
        is_coroutine = False
        if hasattr(self.node, "is_async"):
            is_coroutine = self.node.is_async
        else:
            # Fallback for other callables or older nodes
            real_func = self.node.func
            while hasattr(real_func, "__wrapped__"):
                real_func = real_func.__wrapped__
            is_coroutine = inspect.iscoroutinefunction(real_func) or (
                hasattr(real_func, "__code__") and (real_func.__code__.co_flags & 0x80)
            )

        # Debug prints removed
        # print(f"DEBUG: Node {self.node.output_name}, Stateful: {list(stateful_params.keys())}, Dynamic: {dynamic_params}")
        # print(f"DEBUG: Func {self.node.func}, is_coroutine: {is_coroutine}")

        if stateful_params:
            return self._execute_stateful(df, dynamic_params, stateful_params, context)

        if is_coroutine:
            return self._execute_async_stateless(df, dynamic_params)

        return self._execute_stateless(df, dynamic_params)

    def _execute_stateless(
        self, df: "daft.DataFrame", dynamic_params: List[str]
    ) -> "daft.DataFrame":
        func = self._make_serializable_by_value(self.node.func)
        # Enforce function name to match node output name for telemetry mapping
        func.__name__ = self.node.func.__name__
        return_dtype = self._infer_daft_return_type(self.node.func) or DataType.python()

        udf = daft.func(func, return_dtype=return_dtype)
        input_cols = [daft.col(p) for p in dynamic_params]
        return df.with_column(self.node.output_name, udf(*input_cols))

    def _execute_async_stateless(
        self, df: "daft.DataFrame", dynamic_params: List[str]
    ) -> "daft.DataFrame":
        # Daft doesn't support async functions in daft.func directly.
        # We must wrap it to run synchronously.
        func = self._make_serializable_by_value(self.node.func)
        return_dtype = self._infer_daft_return_type(self.node.func) or DataType.python()

        def sync_wrapper(*args):
            # We should run asyncio.run() if there is no loop, or run in a separate thread if there is one
            
            try:
                # Try to get the running loop
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # If loop is running, we can't use asyncio.run() directly or loop.run_until_complete()
                # Instead, we run the async function in a separate thread which has its own loop
                
                def target():
                    return asyncio.run(func(*args))
                
                # Run in a thread and wait for result
                # We use a simple ThreadPoolExecutor to manage the thread
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(target)
                    return future.result()

            return asyncio.run(func(*args))

        # Enforce function name to match node output name for telemetry mapping
        sync_wrapper.__name__ = self.node.func.__name__
        udf = daft.func(sync_wrapper, return_dtype=return_dtype)
        input_cols = [daft.col(p) for p in dynamic_params]
        return df.with_column(self.node.output_name, udf(*input_cols))

    def _execute_stateful(
        self,
        df: "daft.DataFrame",
        dynamic_params: List[str],
        stateful_params: Dict[str, Any],
        context: ExecutionContext,
    ) -> "daft.DataFrame":
        is_async = False
        if hasattr(self.node, "is_async"):
            is_async = self.node.is_async
        else:
            real_func = self.node.func
            while hasattr(real_func, "__wrapped__"):
                real_func = real_func.__wrapped__
            is_async = inspect.iscoroutinefunction(real_func) or (
                hasattr(real_func, "__code__") and (real_func.__code__.co_flags & 0x80)
            )

        return_dtype = self._infer_daft_return_type(self.node.func) or DataType.python()
        serializable_func = self._make_serializable_by_value(self.node.func)

        # Use stateful_params directly (StatefulWrapper handles lazy init)
        # We need to make sure we don't capture the whole context or engine
        captured_stateful_params = stateful_params

        config = context.engine.default_daft_config
        cls_kwargs = {}
        if "max_concurrency" in config:
            cls_kwargs["max_concurrency"] = config["max_concurrency"]
        if "use_process" in config:
            cls_kwargs["use_process"] = config["use_process"]
        if "gpus" in config:
            cls_kwargs["gpus"] = config["gpus"]

        all_params = self.node.root_args

        if is_async:

            @daft.cls(**cls_kwargs)
            class StatefulNodeWrapper:
                def __init__(self, stateful_objects):
                    self._stateful_objects = stateful_objects

                @daft.method(return_dtype=return_dtype)
                async def __call__(self, *args):
                    # args are values (not expressions)
                    # If we passed a dummy argument, we should ignore it if we don't need it
                    # But we need to be careful about argument mapping

                    # Filter out dummy arg if present (we'll pass it as last arg if needed)
                    relevant_args = args[: len(dynamic_params)]

                    arg_iter = iter(relevant_args)
                    kwargs = {}
                    for param in all_params:
                        if param in self._stateful_objects:
                            kwargs[param] = self._stateful_objects[param]
                        else:
                            kwargs[param] = next(arg_iter)
                    return await serializable_func(**kwargs)
        else:

            @daft.cls(**cls_kwargs)
            class StatefulNodeWrapper:
                def __init__(self, stateful_objects):
                    self._stateful_objects = stateful_objects

                @daft.method(return_dtype=return_dtype)
                def __call__(self, *args):
                    # args are values (not expressions)
                    relevant_args = args[: len(dynamic_params)]

                    arg_iter = iter(relevant_args)
                    kwargs = {}
                    for param in all_params:
                        if param in self._stateful_objects:
                            kwargs[param] = self._stateful_objects[param]
                        else:
                            kwargs[param] = next(arg_iter)
                    return serializable_func(**kwargs)

        # Pass captured_stateful_params explicitly to __init__
        wrapper = StatefulNodeWrapper(captured_stateful_params)

        # For stateful nodes, Daft uses the class/method name.
        # We can try to set the __name__ of the instance or the call method,
        # but Daft's UDF naming for classes is more complex.
        # However, since we return a column with the output_name,
        # and the UDF itself might be named after the class.
        # Let's try to set the class name dynamically if possible, or just rely on the fact
        # that we might not get perfect mapping for stateful nodes yet.
        # Actually, we can set __name__ on the wrapper instance's __call__?
        # No, Daft inspects the class.
        # Let's rename the class locally.
        StatefulNodeWrapper.__name__ = f"Stateful_{self.node.output_name}"
        StatefulNodeWrapper.__qualname__ = f"Stateful_{self.node.output_name}"

        # Ensure we only pass dynamic params as columns
        input_cols = []
        for param in dynamic_params:
            if param not in stateful_params:
                input_cols.append(daft.col(param))

        # Remove debug print
        # print(f"DEBUG: Input Cols: {input_cols}")

        # Handle stateful-only nodes (no dynamic inputs)
        if not input_cols:
            raise ValueError(
                f"Node '{self.node.output_name}' has only stateful parameters. Daft UDFs require at least one data column."
            )

        return df.with_column(self.node.output_name, wrapper(*input_cols))

    def generate_code(
        self, df_var: str, available_columns: Set[str], context: CodeGenContext
    ) -> str:
        func_name = self.node.func.__name__

        # Add import for inspect if we use it, but here we just assume function is defined
        # In a real scenario we would dump the source
        try:
            import ast
            import textwrap

            source = textwrap.dedent(inspect.getsource(self.node.func))

            # Strip @node decorator using AST to avoid NameError in generated code
            try:
                tree = ast.parse(source)
                if tree.body and isinstance(
                    tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    func_def = tree.body[0]
                    # Filter out decorators that look like 'node'
                    new_decorators = []
                    for dec in func_def.decorator_list:
                        is_node = False
                        # Check @node
                        if isinstance(dec, ast.Name) and dec.id == "node":
                            is_node = True
                        # Check @node(...)
                        elif (
                            isinstance(dec, ast.Call)
                            and isinstance(dec.func, ast.Name)
                            and dec.func.id == "node"
                        ):
                            is_node = True

                        if not is_node:
                            new_decorators.append(dec)

                    func_def.decorator_list = new_decorators
                    if hasattr(ast, "unparse"):
                        source = ast.unparse(tree)
            except Exception:
                # Fallback if AST parsing fails, just use original source
                pass

            context.add_udf_definition(f"@daft.func\n{source}")
        except Exception:
            context.add_udf_definition(f"# Definition for {func_name} missing")

        args = []
        for param in self.node.root_args:
            if param in available_columns:
                args.append(f'daft.col("{param}")')
            elif param in context._stateful_inputs:
                args.append(f"stateful_inputs['{param}']")  # Placeholder

        return f'{df_var} = {df_var}.with_column("{self.node.output_name}", {func_name}({", ".join(args)}))'


class DualNodeOperation(DaftOperation):
    """Handles DualNodes (chooses singular or batch based on context)."""

    def __init__(self, node: Any, is_map_context: bool):
        self.node = node
        self.is_map_context = is_map_context

    def execute(
        self,
        df: "daft.DataFrame",
        available_columns: Set[str],
        context: ExecutionContext,
    ) -> "daft.DataFrame":
        # Choose implementation
        func = self.node.batch if self.is_map_context else self.node.singular

        # If not in map context, treat as simple node
        if not self.is_map_context:
            # Delegate to stateless execution logic (reusing helper if possible, or just inline)
            # For simplicity, inline the stateless logic here as it's short
            serializable_func = self._make_serializable_by_value(func)
            return_dtype = (
                self._infer_daft_return_type(self.node.singular) or DataType.python()
            )
            udf = daft.func(serializable_func, return_dtype=return_dtype)
            input_cols = [daft.col(p) for p in self.node.root_args]
            return df.with_column(self.node.output_name, udf(*input_cols))

        # In map context -> Use Batch UDF (but with user's batch function)
        # We can reuse BatchNodeOperation logic but with the specific batch function
        # Or implement specific DualNode batch logic which might be simpler (no auto-batching needed)

        serializable_func = self._make_serializable_by_value(func)
        return_dtype = (
            self._infer_daft_return_type(self.node.singular) or DataType.python()
        )

        # Use engine config for batch size
        batch_kwargs = {"return_dtype": return_dtype}
        if "batch_size" in context.engine.default_daft_config:
            batch_kwargs["batch_size"] = context.engine.default_daft_config[
                "batch_size"
            ]

        def batch_wrapper_impl(*series_args: Series) -> Series:
            # Unwrap constants
            unwrapped_args = []
            for series_arg in series_args:
                pylist = series_arg.to_pylist()
                if len(pylist) > 0:
                    first_val = pylist[0]
                    is_constant = all(
                        val is first_val or val == first_val for val in pylist[1:]
                    )
                    if is_constant:
                        unwrapped_args.append(first_val)
                    else:
                        # Convert Daft Series to Python list (to_pylist()) or Arrow (to_arrow())
                        # Daft Series don't support scalar arithmetic like `x * 2`.
                        # Arrow Arrays don't support it either with `*` operator in all versions.
                        # Python lists support `*` but it means extend, not multiply.

                        # BUT, the previous implementation (that "worked") likely relied on something else.
                        # If we look at the guide code:
                        # def compute_batch(x_series):
                        #     # x_series is a Daft Series or Arrow Array
                        #     return x_series * 2

                        # If this code worked before, what was `x_series`?
                        # Maybe it was a numpy array? Or maybe Daft Series supported it?
                        # Let's try numpy.

                        arrow_arr = series_arg.to_arrow()
                        try:
                            unwrapped_args.append(arrow_arr.to_numpy())
                        except:
                            unwrapped_args.append(series_arg.to_pylist())
                else:
                    arrow_arr = series_arg.to_arrow()
                    try:
                        unwrapped_args.append(arrow_arr.to_numpy())
                    except:
                        unwrapped_args.append(series_arg.to_pylist())

            # Call user's batch function
            result = serializable_func(*unwrapped_args)

            if isinstance(result, list):
                return Series.from_pylist(result)
            return result

        # Enforce function name to match node output name for telemetry mapping
        batch_wrapper_impl.__name__ = self.node.name
        batch_wrapper_impl.__qualname__ = self.node.name
        batch_wrapper = daft.func.batch(**batch_kwargs)(batch_wrapper_impl)

        input_cols = [daft.col(p) for p in self.node.root_args]
        return df.with_column(self.node.output_name, batch_wrapper(*input_cols))

    def generate_code(
        self, df_var: str, available_columns: Set[str], context: CodeGenContext
    ) -> str:
        func_name = self.node.singular.__name__
        if self.is_map_context:
            return f"# DualNode Batch: {self.node.output_name}"
        return f'{df_var} = {df_var}.with_column("{self.node.output_name}", {func_name}(...))'


class BatchNodeOperation(DaftOperation):
    """Handles batch-optimized nodes."""

    def __init__(self, node: Any):
        self.node = node

    def execute(
        self,
        df: "daft.DataFrame",
        available_columns: Set[str],
        context: ExecutionContext,
    ) -> "daft.DataFrame":
        func = self._make_serializable_by_value(self.node.func)
        return_dtype = DataType.python()

        # Optimization: Create executor inside to avoid serialization issues
        # We cannot capture the context.executor because it's not picklable

        def batch_udf_impl(*series_args: Series) -> Series:
            python_args = []
            for s in series_args:
                python_args.append(s.to_pylist())

            list_idx = -1
            for i, arg in enumerate(python_args):
                if isinstance(arg, list):
                    list_idx = i
                    break

            if list_idx == -1:
                return Series.from_pylist([func(*python_args)])

            n_items = len(python_args[list_idx])

            # Heuristic: Small batch -> run sequentially
            if n_items < 100:
                results = []
                for i in range(n_items):
                    call_args = [
                        arg[i] if isinstance(arg, list) else arg for arg in python_args
                    ]
                    results.append(func(*call_args))
                return Series.from_pylist(results)

            # Large batch -> use local executor
            import concurrent.futures

            # Use a reasonable default for workers since we can't access engine config easily here
            # without capturing it.
            with concurrent.futures.ThreadPoolExecutor() as executor:

                def process_item(idx):
                    call_args = [
                        arg[idx] if isinstance(arg, list) else arg
                        for arg in python_args
                    ]
                    return func(*call_args)

                results = list(executor.map(process_item, range(n_items)))
            return Series.from_pylist(results)

        # Enforce function name to match node output name for telemetry mapping
        batch_udf_impl.__name__ = self.node.func.__name__
        batch_udf_impl.__qualname__ = self.node.func.__name__
        batch_udf = daft.func.batch(return_dtype=return_dtype)(batch_udf_impl)

        input_cols = [daft.col(p) for p in self.node.root_args]
        return df.with_column(self.node.output_name, batch_udf(*input_cols))

    def generate_code(
        self, df_var: str, available_columns: Set[str], context: CodeGenContext
    ) -> str:
        func_name = self.node.func.__name__
        # Generate a valid placeholder function
        context.add_udf_definition(
            f"@daft.func.batch\ndef {func_name}_batch(*args): pass"
        )
        return f'{df_var} = {df_var}.with_column("{self.node.output_name}", {func_name}_batch(*[daft.col(c) for c in {list(self.node.root_args)}]))'


class PipelineNodeOperation(DaftOperation):
    """Handles nested pipelines (map_over)."""

    def __init__(self, node: Any):
        self.node = node

    def execute(
        self,
        df: "daft.DataFrame",
        available_columns: Set[str],
        context: ExecutionContext,
    ) -> "daft.DataFrame":
        inner_pipeline = self.node.pipeline
        input_mapping = self.node.input_mapping or {}
        output_mapping = self.node.output_mapping or {}
        map_over = self.node.map_over

        # 1. Add row_id
        row_id_col = "__daft_row_id__"
        df = df._add_monotonically_increasing_id(row_id_col)

        # 2. Explode
        map_over_col = map_over[0] if isinstance(map_over, list) else map_over
        original_list_col = f"__original_{map_over_col}__"
        df = df.with_column(original_list_col, daft.col(map_over_col))
        df = df.explode(daft.col(map_over_col))

        # 3. Input Mapping
        if input_mapping:
            selects = []
            inner_available = set()
            for outer, inner in input_mapping.items():
                if outer in available_columns or outer == map_over_col:
                    selects.append(df[outer].alias(inner))
                    inner_available.add(inner)
            # Keep others
            for col in df.column_names:
                if (
                    col not in input_mapping
                    and col != map_over_col
                    and col != row_id_col
                ):
                    selects.append(df[col])
                    inner_available.add(col)
            selects.append(df[row_id_col])
            df = df.select(*selects)
        else:
            inner_available = available_columns.copy()

        # 4. Inner Pipeline Execution
        # Recursively use engine to get operations for inner nodes
        # We need to access the engine's logic to select operations
        # This is where the refactor pays off - we just use the same factory logic

        for inner_node in inner_pipeline.graph.execution_order:
            # Factory logic (should be in a factory, but for now inline or via engine method)
            op = context.engine._create_operation(inner_node)
            df = op.execute(df, inner_available, context)
            if hasattr(inner_node, "output_name"):
                inner_available.add(inner_node.output_name)

        # 5. Output Mapping & Aggregation
        inner_outputs = [n.output_name for n in inner_pipeline.graph.execution_order]
        final_outputs = [output_mapping.get(n, n) for n in inner_outputs]

        # Rename outputs
        if output_mapping:
            selects = []
            for col in df.column_names:
                if col in inner_outputs:
                    selects.append(df[col].alias(output_mapping.get(col, col)))
                elif col != row_id_col:
                    selects.append(df[col])
            selects.append(df[row_id_col])
            df = df.select(*selects)
        else:
            # Aggregate
            pass  # Logic continues...

        # Aggregate
        df_grouped = df.groupby(daft.col(row_id_col))
        agg_exprs = []
        for out in final_outputs:
            agg_exprs.append(daft.col(out).list_agg().alias(out))

        # Restore others
        for col in df.column_names:
            if (
                col not in final_outputs
                and col != row_id_col
                and col != original_list_col
            ):
                agg_exprs.append(daft.col(col).any_value().alias(col))

        agg_exprs.append(
            daft.col(original_list_col).any_value().alias(original_list_col)
        )

        df = df_grouped.agg(*agg_exprs)

        # Restore order
        df = df.sort(daft.col(row_id_col))

        df = df.with_column(map_over_col, daft.col(original_list_col))

        # Cleanup
        final_cols = [
            c for c in df.column_names if c != row_id_col and c != original_list_col
        ]
        df = df.select(*[df[c] for c in final_cols])

        return df

    def generate_code(
        self, df_var: str, available_columns: Set[str], context: CodeGenContext
    ) -> str:
        lines = []
        lines.append(f"# Pipeline {self.node.output_name}")
        lines.append(f"{df_var} = {df_var}.explode(...)")
        # ... recursive generation ...
        lines.append(f"{df_var} = {df_var}.groupby(...).agg(...)")
        return "\n".join(lines)


class SimplePipelineOperation(DaftOperation):
    """Handles simple nested pipelines (no map_over)."""

    def __init__(self, node: Any):
        self.node = node

    def execute(
        self,
        df: "daft.DataFrame",
        available_columns: Set[str],
        context: ExecutionContext,
    ) -> "daft.DataFrame":
        inner_pipeline = self.node.pipeline

        # Just execute inner nodes sequentially on the same dataframe
        # This assumes inner nodes don't conflict with outer columns or are properly scoped
        # In a full implementation we might need to handle scoping more carefully

        inner_available = available_columns.copy()

        for inner_node in inner_pipeline.graph.execution_order:
            op = context.engine._create_operation(inner_node)
            df = op.execute(df, inner_available, context)
            if hasattr(inner_node, "output_name"):
                inner_available.add(inner_node.output_name)

        return df

    def generate_code(
        self, df_var: str, available_columns: Set[str], context: CodeGenContext
    ) -> str:
        lines = []
        lines.append(f"# Nested Pipeline: {self.node.output_name}")

        inner_pipeline = self.node.pipeline
        inner_available = available_columns.copy()

        for inner_node in inner_pipeline.graph.execution_order:
            op = context.engine._create_operation(inner_node)
            code = op.generate_code(df_var, inner_available, context)
            lines.append(code)
            if hasattr(inner_node, "output_name"):
                inner_available.add(inner_node.output_name)

        return "\n".join(lines)
