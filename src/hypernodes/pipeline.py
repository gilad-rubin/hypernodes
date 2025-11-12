"""Pipeline class for managing and executing DAGs of nodes."""

import time
from typing import Any, Dict, List, Optional, Union

from .cache import Cache
from .callbacks import PipelineCallback
from .engine import Engine, HypernodesEngine
from .graph_builder import SimpleGraphBuilder
from .map_planner import MapPlanner
from .mixins import BuilderMixin
from .node import Node
from .pipeline_node import PipelineNode


class Pipeline(BuilderMixin):
    def __init__(
        self,
        nodes: List[Node],
        engine: Optional[Engine] = None,
        cache: Optional[Cache] = None,
        callbacks: Optional[List[PipelineCallback]] = None,
        name: Optional[str] = None,
    ):
        # Set engine with default fallback
        self.engine = engine if engine is not None else HypernodesEngine()

        self.cache = cache
        self.callbacks = callbacks if callbacks is not None else []
        self.name = name

        self.nodes = nodes

        graph_builder = SimpleGraphBuilder()
        self.graph = graph_builder.build_graph(self.nodes)

        self.id = self._generate_pipeline_id()

    def _generate_pipeline_id(self) -> str:
        """Generate a deterministic pipeline ID."""
        import builtins

        return f"pipeline_{builtins.id(self)}"

    def run(
        self,
        inputs: Dict[str, Any],
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        return self.engine.run(self, inputs, output_name=output_name)

    def map(
        self,
        inputs: Dict[str, Any],
        map_over: Union[str, List[str]],
        map_mode: str = "zip",
        output_name: Union[str, List[str], None] = None,
        return_format: str = "python",
    ) -> Any:
        """Execute pipeline over a collection of inputs.

        This method enables batch processing where the pipeline runs multiple times
        with different values for specified parameters. Each execution is independent
        and can be cached separately.

        Args:
            inputs: Dictionary mapping parameter names to values.
                   For parameters in map_over, values must be lists.
                   For parameters not in map_over, values are single constants.
            map_over: Parameter name(s) that vary across executions.
                     Can be a single string or list of strings.
            map_mode: How to combine multiple map_over parameters:
                     - "zip" (default): Process corresponding items together.
                       All lists must have the same length.
                     - "product": Create all combinations of items.
                       Lists can have different lengths.
            return_format: Output representation. Defaults to "python" which
                returns a dict of lists. Engines may provide additional formats
                such as "daft" or "arrow" for columnar fast-paths.
            output_name: Optional output name(s) to compute. Can be:
                - str: Single output name
                - List[str]: Multiple output names
                - None: Compute all outputs (default)
                Only the specified outputs will be returned, and only the
                minimal set of nodes needed will be executed.

        Returns:
            Dict of lists when ``return_format="python"`` (default). Engines may
            return engine-native objects for other formats (e.g., Daft DataFrame).

        Raises:
            ValueError: If map_mode is "zip" and list lengths don't match
            ValueError: If output_name specifies a non-existent output

        Example:
            >>> @node(output_name="result")
            >>> def add_one(x: int) -> int:
            ...     return x + 1
            >>>
            >>> pipeline = Pipeline(nodes=[add_one])
            >>> results = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
            >>> assert results == {"result": [2, 3, 4]}
        """
        # Normalize and validate inputs
        if isinstance(map_over, str):
            map_over = [map_over]

        if map_mode not in ("zip", "product"):
            raise ValueError(f"map_mode must be 'zip' or 'product', got '{map_mode}'")

        allowed_return_formats = {"python", "daft", "arrow"}
        if return_format not in allowed_return_formats:
            raise ValueError(
                f"return_format must be one of {sorted(allowed_return_formats)}, "
                f"got '{return_format}'"
            )

        # Prepare execution plans using MapPlanner
        planner = MapPlanner()

        engine = self.engine
        columnar_result = None
        columnar_error: Optional[Exception] = None

        # For columnar execution, we need varying/fixed params separately
        # For row-based execution, we need the full execution plans
        varying_params, fixed_params, zip_lengths = planner._prepare_inputs(
            inputs, map_over, map_mode
        )
        total_items_hint = zip_lengths[0] if zip_lengths else 0

        # Try columnar execution if available (e.g., DaftEngine)
        if (
            map_mode == "zip"
            and hasattr(engine, "map_columnar")
            and callable(getattr(engine, "map_columnar"))
        ):
            try:
                columnar_result = engine.map_columnar(  # type: ignore[attr-defined]
                    pipeline=self,
                    varying_inputs=varying_params,
                    fixed_inputs=fixed_params,
                    output_name=output_name,
                    return_format=return_format,
                )
            except Exception as exc:
                columnar_result = None
                columnar_error = exc

        # Fallback to row-based execution
        if columnar_result is None:
            if return_format != "python":
                detail = (
                    f"return_format='{return_format}' requires an engine with a "
                    f"columnar fast-path (e.g., DaftEngine.map_columnar)"
                )
                if columnar_error is not None:
                    raise ValueError(detail) from columnar_error
                raise ValueError(detail)

            # Build execution plans from prepared inputs
            execution_plans = planner._build_plans(
                varying_params, fixed_params, map_mode, zip_lengths
            )
        else:
            execution_plans = None

        # Execute pipeline for each plan with callbacks
        # Use existing context from contextvar or create new one
        from .callbacks import CallbackContext
        from .execution_context import get_callback_context, set_callback_context

        ctx = get_callback_context()
        if ctx is None:
            ctx = CallbackContext()
        ctx.push_pipeline(self.id)

        # Set context for nested engine/pipeline calls
        set_callback_context(ctx)

        # Get callbacks (with fallback to empty list)
        callbacks = self.callbacks if self.callbacks else []

        # Set pipeline metadata so callbacks can access node information
        # Use same logic as engine._get_node_id() for consistency
        node_ids = []
        for n in self.graph.execution_order:
            # PipelineNode with explicit name
            if hasattr(n, "name") and n.name:
                node_ids.append(n.name)
            # Regular node with function name
            elif hasattr(n, "func") and hasattr(n.func, "__name__"):
                node_ids.append(n.func.__name__)
            # Pipeline or object with id
            elif hasattr(n, "id"):
                node_ids.append(n.id)
            # Object with __name__
            elif hasattr(n, "__name__"):
                node_ids.append(n.__name__)
            # Fallback
            else:
                node_ids.append(str(n))

        ctx.set_pipeline_metadata(
            self.id,
            {
                "total_nodes": len(self.graph.execution_order),
                "node_ids": node_ids,
                "pipeline_name": self.name or self.id,
            },
        )

        # Trigger map start callbacks
        if columnar_result is not None:
            total_items = total_items_hint if total_items_hint is not None else 0
        else:
            total_items = len(execution_plans)
        map_start_time = time.time()
        for callback in callbacks:
            callback.on_map_start(total_items, ctx)

        if columnar_result is None:
            # Delegate to the engine's map executor for parallel execution
            # The engine.map() method will use the configured map_executor
            # (sequential, async, threaded, or parallel)
            results_list = engine.map(
                pipeline=self,
                items=execution_plans,  # List of input dicts (one per map item)
                inputs={},  # No shared inputs - all inputs are in the items
                output_name=output_name,
            )
        else:
            results_list = columnar_result

        # Trigger map end callbacks
        map_duration = time.time() - map_start_time
        for callback in callbacks:
            callback.on_map_end(map_duration, ctx)

        if columnar_result is not None:
            if return_format == "python":
                if not columnar_result:
                    output_keys = []
                    for node in self.nodes:
                        if isinstance(node, Pipeline):
                            for inner_node in node.nodes:
                                output_keys.append(inner_node.output_name)
                        else:
                            output_keys.append(node.output_name)
                    columnar_result = {key: [] for key in output_keys}
            ctx.pop_pipeline()
            return columnar_result

        # Transpose results: from list of dicts to dict of lists
        if not results_list:
            output_keys = []
            for node in self.nodes:
                if isinstance(node, Pipeline):
                    for inner_node in node.nodes:
                        output_keys.append(inner_node.output_name)
                else:
                    output_keys.append(node.output_name)
            return {key: [] for key in output_keys}

        output_keys = results_list[0].keys()
        result = {key: [result[key] for result in results_list] for key in output_keys}
        ctx.pop_pipeline()
        return result

    def as_node(
        self,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        map_over: Optional[Union[str, List[str]]] = None,
        map_mode: str = "zip",
        name: Optional[str] = None,
        engine: Optional[Engine] = None,
        cache: Optional[Cache] = None,
        callbacks: Optional[List[PipelineCallback]] = None,
    ) -> PipelineNode:
        """Wrap this pipeline as a node with custom input/output mapping.

        This method allows a pipeline to be used as a node in another pipeline
        with renamed parameters and/or internal mapping over collections.

        Args:
            input_mapping: Maps outer parameter names to inner names.
                          Format: {outer_name: inner_name}
                          Direction: outer → inner (how inputs flow IN)
            output_mapping: Maps inner output names to outer names.
                           Format: {inner_name: outer_name}
                           Direction: inner → outer (how outputs flow OUT)
            map_over: Parameter name(s) that should be mapped over.
                     From outer pipeline's perspective, this is a list parameter.
                     Internally, the pipeline maps over each item.
            map_mode: Mode for map operation ("zip" or "product")
            name: Optional name for this node (displayed in visualizations)
            engine: Override engine for this node's execution
            cache: Override cache for this node's execution
            callbacks: Override callbacks for this node's execution

        Returns:
            PipelineNode that wraps this pipeline with the specified mapping

        Example:
            >>> inner = Pipeline(nodes=[clean_text])  # expects "passage"
            >>> adapted = inner.as_node(
            ...     input_mapping={"document": "passage"},  # outer -> inner
            ...     output_mapping={"cleaned": "processed"},  # inner -> outer
            ...     map_mode="zip",
            ...     name="adapted_clean"
            ... )
            >>> outer = Pipeline(nodes=[adapted])
            >>> result = outer.run(inputs={"document": "text"})
            >>> # result["processed"] contains the cleaned text
        """
        return PipelineNode(
            pipeline=self,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            map_over=map_over,
            map_mode=map_mode,
            name=name,
            engine=engine,
            cache=cache,
            callbacks=callbacks,
        )

    def __enter__(self) -> "Pipeline":
        """Enter context manager.

        Allows using Pipeline with 'with' statement for automatic cleanup.

        Returns:
            Self for use in context

        Example:
            >>> with Pipeline(nodes=[...]) as pipeline:
            ...     results = pipeline.run(inputs={"x": 5})
            ... # Engine automatically shut down here
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and cleanup resources.

        Automatically shuts down the configured engine if we created it.
        """
        self._cleanup()
        return False  # Don't suppress exceptions

    def __del__(self):
        """Destructor to ensure cleanup happens even without context manager.

        This is called when the Pipeline object is garbage collected,
        ensuring engines are properly shut down.
        """
        self._cleanup()

    def _cleanup(self):
        """Internal method to cleanup resources (shutdown engines).

        Only shuts down engines that we created (not user-provided ones).
        Safe to call multiple times.
        """
        # Only cleanup our own engine (not inherited from parent)
        if self.engine is not None and hasattr(self.engine, "shutdown"):
            try:
                self.engine.shutdown(wait=True)
            except Exception:
                # Ignore errors during cleanup (may already be shut down)
                pass

    def visualize(
        self,
        filename: Optional[str] = None,
        orient: str = "TB",
        depth: Optional[int] = 1,
        flatten: bool = False,
        min_arg_group_size: Optional[int] = 2,
        show_legend: bool = False,
        show_types: bool = True,
        style: Union[str, Any] = "default",
        return_type: str = "auto",
    ):
        """Visualize the pipeline using Graphviz.

        Args:
            filename: Output filename (e.g., "pipeline.svg"). If None, returns object
            orient: Graph orientation ("TB", "LR", "BT", "RL")
            depth: Expansion depth for nested pipelines (1=collapsed, None=fully expand)
            flatten: If True, render nested pipelines inline without containers
            min_arg_group_size: Minimum inputs to group together (None=no grouping)
            show_legend: Whether to show a legend explaining node types
            show_types: Whether to show type hints and default values
            style: Style name from DESIGN_STYLES or GraphvizStyle object
            return_type: "auto", "graphviz", or "html"

        Returns:
            graphviz.Digraph object (or HTML in Jupyter if return_type="html")
        """
        from . import visualization as viz

        return viz.visualize(
            self,
            filename=filename,
            orient=orient,
            depth=depth,
            flatten=flatten,
            min_arg_group_size=min_arg_group_size,
            show_legend=show_legend,
            show_types=show_types,
            style=style,
            return_type=return_type,
        )

    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Make Pipeline callable like a Node.

        This allows pipelines to be used as nodes in other pipelines.

        Args:
            **kwargs: Keyword arguments for inputs

        Returns:
            Dictionary of outputs
        """
        return self.run(kwargs)

    def __repr__(self) -> str:
        """Return string representation of the Pipeline."""
        if self.name:
            return f"Pipeline(name={self.name!r}, nodes={len(self.nodes)})"
        return f"Pipeline({len(self.nodes)} nodes)"

    def __hash__(self) -> int:
        """Make Pipeline hashable for use in networkx graphs."""
        return id(self)

    def __eq__(self, other) -> bool:
        """Check equality based on identity."""
        return self is other
