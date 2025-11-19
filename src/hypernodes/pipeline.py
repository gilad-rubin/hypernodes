"""Pipeline class for managing and executing DAGs of nodes."""

from typing import Any, Dict, List, Literal, Optional, Union

from .graph_builder import SimpleGraphBuilder
from .hypernode import HyperNode
from .pipeline_node import PipelineNode
from .protocols import Engine
from .sequential_engine import SeqEngine


class Pipeline:
    def __init__(
        self,
        nodes: List[HyperNode],
        engine: Optional[Engine] = None,
        name: Optional[str] = None,
    ):
        # Set engine with default fallback to SeqEngine
        self.engine = engine if engine is not None else SeqEngine()
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

    def _validate_inputs(
        self, inputs: Dict[str, Any], output_name: Union[str, List[str], None] = None
    ) -> None:
        """Validate that provided inputs satisfy pipeline's root_args.

        When output_name is specified, only validates inputs needed for those specific outputs.

        Args:
            inputs: Input dictionary to validate
            output_name: If specified, only validate inputs needed for these outputs

        Raises:
            ValueError: If required inputs are missing
        """
        provided = set(inputs.keys())

        # Compute required inputs based on output_name
        if output_name is not None:
            # Get only the nodes needed for requested outputs
            required_nodes = self.graph.get_required_nodes(output_name)
            if required_nodes is None:
                # All nodes needed (shouldn't happen since output_name is not None)
                required = set(self.graph.root_args)
            else:
                # Compute root args needed for these specific nodes
                required = self._compute_root_args_for_nodes(required_nodes)
        else:
            # No output_name specified - need all root args
            required = set(self.graph.root_args)

        missing = required - provided
        if missing:
            raise ValueError(
                f"Missing required input(s): {sorted(missing)}. "
                f"Required inputs: {sorted(required)}"
            )

    def _compute_root_args_for_nodes(self, nodes: List[HyperNode]) -> set:
        """Compute root arguments needed for a specific set of nodes.

        Args:
            nodes: List of nodes to compute root args for

        Returns:
            Set of parameter names that are external inputs for these nodes
        """
        # Collect all parameters needed by these nodes
        all_params = set()
        for node in nodes:
            all_params.update(node.root_args)

        # Subtract outputs produced by nodes in this set
        outputs_from_these_nodes = set()
        for node in nodes:
            if isinstance(node.output_name, tuple):
                outputs_from_these_nodes.update(node.output_name)
            else:
                outputs_from_these_nodes.add(node.output_name)

        # Root args are params that aren't produced by any node in the set
        return all_params - outputs_from_these_nodes

    def run(
        self,
        inputs: Dict[str, Any],
        output_name: Union[str, List[str], None] = None,
        engine: Optional[Engine] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute the pipeline.

        Args:
            inputs: Input parameters
            output_name: Optional output(s) to compute
            engine: Optional engine override for this execution
            **kwargs: Additional args passed to engine
        """
        self._validate_output_names(output_name)
        self._validate_inputs(inputs, output_name=output_name)

        exec_engine = engine if engine is not None else self.engine
        return exec_engine.run(self, inputs, output_name=output_name, **kwargs)

    def map(
        self,
        inputs: Dict[str, Any],
        map_over: Union[str, List[str]],
        map_mode: Literal["zip", "product"] = "zip",
        output_name: Union[str, List[str], None] = None,
        engine: Optional[Engine] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Execute the pipeline over multiple items.

        Args:
            inputs: Input parameters
            map_over: Parameter(s) to map over
            map_mode: "zip" or "product"
            output_name: Optional output(s) to compute
            engine: Optional engine override for this execution
            **kwargs: Additional args passed to engine
        """
        self._validate_output_names(output_name)
        self._validate_inputs(inputs, output_name=output_name)

        if isinstance(map_over, str):
            map_over = [map_over]

        exec_engine = engine if engine is not None else self.engine
        return exec_engine.map(self, inputs, map_over, map_mode, output_name, **kwargs)

    def as_node(
        self,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        map_over: Optional[Union[str, List[str]]] = None,
        map_mode: str = "zip",
        name: Optional[str] = None,
        cache: bool = True,
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
            cache: Whether to cache the node. Not to be confused with the pipeline's cache.

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
            cache=cache,
        )

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

    def with_engine(self, engine: Engine) -> "Pipeline":
        """Configure with a specific engine.

        Args:
            engine: Engine instance (HypernodesEngine, DaftEngine, etc.)

        Returns:
            Self for method chaining
        """
        self.engine = engine
        return self

    def with_name(self, name: str) -> "Pipeline":
        """Set the pipeline name.

        Args:
            name: Name for the pipeline

        Returns:
            Self for method chaining
        """
        self.name = name
        return self

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
