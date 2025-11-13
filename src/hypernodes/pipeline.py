"""Pipeline class for managing and executing DAGs of nodes."""

from typing import Any, Dict, List, Literal, Optional, Union

from .cache import Cache
from .callbacks import PipelineCallback
from .engine import Engine, HypernodesEngine
from .graph_builder import SimpleGraphBuilder
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

    def _validate_output_names(self, output_name: Union[str, List[str], None]) -> None:
        """Validate that requested output name(s) exist in the pipeline.

        Args:
            output_name: Output name(s) to validate

        Raises:
            ValueError: If any output name doesn't exist in the pipeline
        """
        if output_name is None:
            return

        names = [output_name] if isinstance(output_name, str) else output_name
        available = set(self.graph.available_output_names)

        invalid = [name for name in names if name not in available]
        if invalid:
            raise ValueError(
                f"Output name(s) {invalid} not found in pipeline. "
                f"Available outputs: {sorted(available)}"
            )

    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate that provided inputs satisfy pipeline's root_args.

        Args:
            inputs: Input dictionary to validate

        Raises:
            ValueError: If required inputs are missing
        """
        provided = set(inputs.keys())
        required = set(self.graph.root_args)

        missing = required - provided
        if missing:
            raise ValueError(
                f"Missing required input(s): {sorted(missing)}. "
                f"Required inputs: {sorted(required)}"
            )

    def run(
        self,
        inputs: Dict[str, Any],
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        self._validate_inputs(inputs)
        self._validate_output_names(output_name)
        return self.engine.run(self, inputs, output_name=output_name)

    def map(
        self,
        inputs: Dict[str, Any],
        map_over: Union[str, List[str]],
        map_mode: Literal["zip", "product"] = "zip",
        output_name: Union[str, List[str], None] = None,
        **kwargs: Any,
    ) -> Dict[str, List[Any]]:
        self._validate_inputs(inputs)
        self._validate_output_names(output_name)

        if isinstance(map_over, str):
            map_over = [map_over]

        if map_mode not in ("zip", "product"):
            raise ValueError(f"map_mode must be 'zip' or 'product', got '{map_mode}'")

        return self.engine.map(self, inputs, map_over, map_mode, output_name, **kwargs)

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
