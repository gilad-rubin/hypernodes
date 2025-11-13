from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Union

if TYPE_CHECKING:
    from .pipeline import Pipeline


class PipelineNode:
    """Wraps a Pipeline to behave like a Node with custom input/output mapping.

    This class adapts a Pipeline interface to work as a node in another pipeline,
    with support for parameter renaming and internal mapping.

    Implements HyperNode interface to ensure compatibility when used in Pipeline.nodes lists.

    Attributes:
        pipeline: The wrapped pipeline
        input_mapping: Maps outer parameter names to inner pipeline parameters
        output_mapping: Maps inner pipeline outputs to outer names
        map_over: Optional parameter name(s) to map over (from outer perspective)
        map_mode: Mode for map operation ("zip" or "product")
        cache: Whether to cache the node. Not to be confused with the pipeline's cache.
        name: Optional name for this node (displayed in visualizations)
    """

    def __init__(
        self,
        pipeline: "Pipeline",
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        map_over: Optional[Union[str, List[str]]] = None,
        map_mode: Literal["zip", "product"] = "zip",
        cache: bool = True,  # Whether to cache the node. Not to be confused with the pipeline's cache.
        name: Optional[str] = None,
    ):
        """Initialize a PipelineNode wrapper.

        Args:
            pipeline: Pipeline to wrap
            input_mapping: Maps {outer_name: inner_name} for inputs
            output_mapping: Maps {inner_name: outer_name} for outputs
            map_over: Parameter name(s) to map over (from outer perspective)
            map_mode: Mode for map operation ("zip" or "product")
            cache: Whether to cache the node. Not to be confused with the pipeline's cache.
            name: Optional name for this node (displayed in visualizations)
        """
        self._pipeline = pipeline
        self.input_mapping = input_mapping or {}
        self.output_mapping = output_mapping or {}
        self.map_mode = map_mode
        self.map_over = [map_over] if isinstance(map_over, str) else map_over

        self.cache = cache
        self.name = name if name is not None else pipeline.name

        # Pre-compute and cache aggregated code hash to avoid expensive recomputation
        # This aggregates hashes from all inner nodes (including nested pipelines)
        from .cache import compute_pipeline_code_hash

        self._code_hash = compute_pipeline_code_hash(pipeline)

    @property
    def root_args(self) -> tuple:
        """Get outer parameter names (after applying reverse input mapping).

        Returns:
            Tuple of parameter names from outer pipeline's perspective
        """
        # Special case: map_over without input_mapping exposes only map_over params
        if self.map_over and not self.input_mapping:
            return tuple(self.map_over)

        # Get inner pipeline's root parameters
        inner_params = self._pipeline.graph.root_args

        # Create reverse mapping: inner -> outer
        reverse_mapping = {inner: outer for outer, inner in self.input_mapping.items()}

        # Map each inner param to outer name (or keep same if not in mapping)
        outer_params = [reverse_mapping.get(p, p) for p in inner_params]

        # Add any map_over params that aren't already included
        if self.map_over:
            for param in self.map_over:
                if param not in outer_params:
                    outer_params.append(param)

        return tuple(outer_params)

    @property
    def output_name(self) -> Union[str, tuple]:
        """Get outer output names (after output mapping).

        Returns:
            Output name(s) from outer pipeline's perspective
        """
        # Get inner pipeline outputs (List[str] from graph)
        inner_outputs = self._pipeline.graph.available_output_names

        # Apply output mapping
        outer_outputs = []
        for inner_output in inner_outputs:
            outer_output = self.output_mapping.get(inner_output, inner_output)
            outer_outputs.append(outer_output)

        # Return single string or tuple (matching Node convention)
        if len(outer_outputs) == 1:
            return outer_outputs[0]
        return tuple(outer_outputs)

    @property
    def code_hash(self) -> str:
        """Get cached aggregated code hash for this pipeline node.

        The hash is computed once at node creation by aggregating all inner
        node hashes. This avoids expensive recomputation on every execution.
        The cached value persists through pickling/unpickling.

        Returns:
            SHA256 hash of all inner node code
        """
        return self._code_hash

    @property
    def pipeline(self) -> "Pipeline":
        """Get the wrapped pipeline.

        Returns:
            The Pipeline instance wrapped by this PipelineNode
        """
        return self._pipeline

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PipelineNode({self._pipeline})"

    def __hash__(self) -> int:
        """Make PipelineNode hashable."""
        return hash(id(self))

    def __eq__(self, other) -> bool:
        """Check equality based on identity."""
        return self is other

    def __getstate__(self):
        """Custom pickle support to preserve code hash cache."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Custom unpickle support to restore code hash cache."""
        self.__dict__.update(state)
        # _code_hash is preserved through pickling
