from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from .pipeline import Pipeline

from .cache import Cache
from .callbacks import PipelineCallback
from .engine import Engine
from .node import Node


class PipelineNode(Node):
    """Wraps a Pipeline to behave like a Node with custom input/output mapping.

    This class adapts a Pipeline interface to work as a node in another pipeline,
    with support for parameter renaming and internal mapping.

    Inherits from Node to ensure type compatibility when used in Pipeline.nodes lists.

    Attributes:
        pipeline: The wrapped pipeline
        input_mapping: Maps outer parameter names to inner pipeline parameters
        output_mapping: Maps inner pipeline outputs to outer names
        map_over: Optional parameter name(s) to map over (from outer perspective)
    """

    def __init__(
        self,
        pipeline: "Pipeline",
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        map_over: Optional[Union[str, List[str]]] = None,
        map_mode: str = "zip",
        name: Optional[str] = None,
        # Configuration overrides
        engine: Optional[Engine] = None,
        cache: Optional[Cache] = None,
        callbacks: Optional[List[PipelineCallback]] = None,
    ):
        """Initialize a PipelineNode wrapper.

        Args:
            pipeline: Pipeline to wrap
            input_mapping: Maps {outer_name: inner_name} for inputs
            output_mapping: Maps {inner_name: outer_name} for outputs
            map_over: Parameter name(s) to map over (from outer perspective)
            map_mode: Mode for map operation ("zip" or "product")
            name: Optional name for this node (displayed in visualizations)
            engine: Override engine for this node's execution
            cache: Override cache for this node's execution
            callbacks: Override callbacks for this node's execution
        """
        # Don't call super().__init__() since we have a different initialization pattern
        # We'll override the necessary properties instead
        self.pipeline = pipeline
        self.input_mapping = input_mapping or {}
        self.output_mapping = output_mapping or {}
        self.map_over = map_over
        self.map_mode = map_mode

        # Make map_over a list if it's a string
        if isinstance(self.map_over, str):
            self.map_over = [self.map_over]

        # Store configuration directly
        self._name = name
        self._engine = engine
        self._cache = cache
        self._callbacks = callbacks

    @property
    def name(self) -> Optional[str]:
        """Node name for display."""
        return self._name

    @name.setter
    def name(self, name: Optional[str]) -> None:
        self._name = name

    # Fluent API methods
    def with_engine(self, engine: Engine) -> "PipelineNode":
        """Configure node with specific engine.

        Args:
            engine: Engine instance

        Returns:
            Self for method chaining
        """
        self._engine = engine
        return self

    def with_cache(self, cache: Cache) -> "PipelineNode":
        """Configure node with specific cache.

        Args:
            cache: Cache instance

        Returns:
            Self for method chaining
        """
        self._cache = cache
        return self

    def with_callbacks(self, callbacks: List[PipelineCallback]) -> "PipelineNode":
        """Configure node with specific callbacks.

        Args:
            callbacks: List of callback instances

        Returns:
            Self for method chaining
        """
        self._callbacks = callbacks
        return self

    def with_name(self, name: str) -> "PipelineNode":
        """Configure node with specific name.

        Args:
            name: Display name

        Returns:
            Self for method chaining
        """
        self._name = name
        return self

    @property
    def root_args(self) -> tuple:
        """Get outer parameter names (after input mapping).

        Returns:
            Tuple of parameter names from outer pipeline's perspective
        """
        # Special case: map_over without input_mapping
        # In this mode, the outer interface exposes ONLY the map_over parameter(s),
        # and inner parameters are derived from transposed dict items at call time.
        if self.map_over and not self.input_mapping:
            return tuple(self.map_over)

        # Otherwise, start with inner pipeline's root args
        inner_params = set(self.pipeline.graph.root_args)

        # Apply reverse input mapping: inner -> outer
        reverse_mapping = {v: k for k, v in self.input_mapping.items()}

        outer_params = []
        for inner_param in inner_params:
            outer_param = reverse_mapping.get(inner_param, inner_param)
            outer_params.append(outer_param)

        # If map_over is specified (with input_mapping or matching names),
        # include those parameters as well to ensure varying params are accepted.
        if self.map_over:
            for map_param in self.map_over:
                if map_param not in outer_params:
                    outer_params.append(map_param)

        return tuple(outer_params)

    @property
    def output_name(self) -> Union[str, tuple]:
        """Get outer output names (after output mapping).

        Returns:
            Output name(s) from outer pipeline's perspective
        """
        # Get inner pipeline outputs
        inner_outputs = self.pipeline.output_name
        if not isinstance(inner_outputs, tuple):
            inner_outputs = (inner_outputs,)

        # Apply output mapping
        outer_outputs = []
        for inner_output in inner_outputs:
            outer_output = self.output_mapping.get(inner_output, inner_output)
            outer_outputs.append(outer_output)

        # Return tuple or single value
        if len(outer_outputs) == 1:
            return outer_outputs[0]
        return tuple(outer_outputs)

    @property
    def cache(self) -> bool:
        """Whether this node should be cached.

        Returns:
            True if caching enabled (delegates to pipeline)
        """
        return True  # Caching handled at inner pipeline level

    @property
    def func(self):
        """Get the wrapped pipeline for compatibility with Node interface.

        Returns:
            The wrapped pipeline
        """
        return self.pipeline

    def __call__(self, **kwargs) -> Union[Any, Dict[str, Any]]:
        """Execute the wrapped pipeline with input/output mapping.

        Args:
            **kwargs: Outer parameter names and values

        Returns:
            Mapped outputs (single value or dict depending on output_name)
        """
        # Check if we need to map over parameters
        if self.map_over:
            # Two modes of operation:
            # 1. With input_mapping: map parameter names and pass items directly
            # 2. Without input_mapping + dict items: transpose dict items into separate parameters

            if self.input_mapping:
                # Mode 1: Use input_mapping to rename parameters
                # Apply input mapping: outer -> inner
                inner_inputs = {}
                for outer_name, value in kwargs.items():
                    inner_name = self.input_mapping.get(outer_name, outer_name)
                    inner_inputs[inner_name] = value

                # Determine which inner parameters to map over
                inner_map_over = []
                for outer_param in self.map_over:
                    inner_param = self.input_mapping.get(outer_param, outer_param)
                    inner_map_over.append(inner_param)

                # Execute with map, passing context for nested progress tracking
                inner_results = self.pipeline.map(
                    inputs=inner_inputs,
                    map_over=inner_map_over,
                    map_mode=self.map_mode,
                    _ctx=exec_ctx,
                )
            else:
                # Mode 2: No input_mapping - check if items are dicts to transpose
                # Collect the list from the map_over parameters
                items_list = None
                for outer_param in self.map_over:
                    if outer_param in kwargs:
                        items_list = kwargs[outer_param]
                        break

                if items_list is None or not isinstance(items_list, list):
                    raise ValueError("map_over parameter must provide a list")

                # Check if items are dicts - if so, transpose them
                if items_list and isinstance(items_list[0], dict):
                    # Transpose: list of dicts -> dict of lists
                    inner_inputs = {}
                    for key in items_list[0].keys():
                        # Extract this key from all items
                        inner_inputs[key] = [item.get(key) for item in items_list]

                    # Map over all parameters from the transposed inputs
                    inner_map_over = list(inner_inputs.keys())
                else:
                    # Items are not dicts - use original approach
                    # This handles the case where map_over parameter name matches inner pipeline parameter
                    inner_inputs = {}
                    for outer_name, value in kwargs.items():
                        inner_inputs[outer_name] = value

                    inner_map_over = list(self.map_over)

                # Execute with map, passing context for nested progress tracking
                inner_results = self.pipeline.map(
                    inputs=inner_inputs,
                    map_over=inner_map_over,
                    map_mode=self.map_mode,
                    _ctx=exec_ctx,
                )
        else:
            # No mapping - apply input mapping: outer -> inner
            inner_inputs = {}
            for outer_name, value in kwargs.items():
                inner_name = self.input_mapping.get(outer_name, outer_name)
                inner_inputs[inner_name] = value

            # Execute normally, passing context for nested progress tracking
            inner_results = self.pipeline.run(inputs=inner_inputs, _ctx=exec_ctx)

        # Apply output mapping: inner -> outer
        outer_results = {}
        for inner_name, value in inner_results.items():
            outer_name = self.output_mapping.get(inner_name, inner_name)
            outer_results[outer_name] = value

        return outer_results

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PipelineNode({self.pipeline})"

    def __hash__(self) -> int:
        """Make PipelineNode hashable."""
        return hash(id(self))

    def __eq__(self, other) -> bool:
        """Check equality based on identity."""
        return self is other
