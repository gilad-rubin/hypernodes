"""Pipeline class for managing and executing DAGs of nodes."""

import itertools
import time
from typing import Any, Dict, List, Optional, Union

from .cache import Cache
from .callbacks import PipelineCallback
from .engine import Engine
from .graph_builder import NetworkXGraphBuilder
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
    def parameters(self) -> tuple:
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
        inner_params = set(self.pipeline.root_args)

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
        # Get execution context if available (set by engine)
        exec_ctx = getattr(self, "_exec_ctx", None)

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


class Pipeline:
    """Pipeline that manages DAG execution of nodes.

    A Pipeline constructs a directed acyclic graph (DAG) from a list of nodes,
    automatically resolving dependencies based on parameter and output names.
    It then executes the nodes in topological order.

    Pipelines can be nested: a Pipeline can itself be used as a node in another
    Pipeline, enabling hierarchical composition.

    Attributes:
        nodes: List of Node instances or nested Pipelines
        config: Pipeline configuration (engine, cache, callbacks, name)
        graph_builder: GraphBuilder for graph operations
        output_to_node: Mapping from output names to nodes
        graph: NetworkX DiGraph representing dependencies (cached)
        execution_order: Topologically sorted list of nodes (cached)
        root_args: External input parameters needed by the pipeline
    """

    def __init__(
        self,
        nodes: List[Union[Node, "Pipeline"]],
        engine: Optional[Engine] = None,
        cache: Optional[Cache] = None,
        callbacks: Optional[List[PipelineCallback]] = None,
        name: Optional[str] = None,
        # Deprecated parameters (kept for backward compatibility)
        parent: Optional["Pipeline"] = None,
    ):
        """Initialize a Pipeline from a list of nodes.

        Args:
            nodes: List of Node instances or Pipeline instances (which are auto-wrapped)
            engine: Engine for execution (default: HypernodesEngine())
            cache: Cache backend for result caching (default: None, no caching)
            callbacks: List of callbacks for lifecycle hooks (default: None)
            name: Human-readable name for the pipeline (used in visualization)
            parent: DEPRECATED - Parent pipeline for configuration inheritance (use ExecutionContext instead)

        Raises:
            CycleError: If a cycle is detected in the dependency graph
            DependencyError: If a dependency cannot be satisfied
        """
        # 1. Store configuration directly (no more PipelineConfiguration)
        self.engine = engine
        self.cache = cache
        self.callbacks = callbacks if callbacks is not None else []
        self.name = name
        
        # Store parent for backward compatibility (will be phased out)
        self._parent = parent
        
        # 2. Auto-wrap Pipeline instances in PipelineNode for consistency
        self.nodes = self._wrap_nodes(nodes)
        
        # 3. Build graph eagerly using graph builder (validates and computes all)
        graph_builder = NetworkXGraphBuilder()
        result = graph_builder.build_graph(self.nodes)
        
        # Store all graph-related properties from GraphResult
        self.graph = result.graph
        self.output_to_node = result.output_to_node
        self.execution_order = result.execution_order
        self.root_args = result.root_args
        
        # 4. Generate deterministic pipeline ID
        import builtins
        self.id = f"pipeline_{builtins.id(self)}"
    
    def _wrap_nodes(self, nodes: List[Union[Node, "Pipeline"]]) -> List[Node]:
        """Auto-wrap Pipeline instances in PipelineNode for consistency.
        
        Args:
            nodes: List of Node or Pipeline instances
            
        Returns:
            List of Node instances (Pipelines wrapped in PipelineNode)
        """
        wrapped_nodes = []
        for node in nodes:
            if isinstance(node, Pipeline):
                # Wrap pipeline in PipelineNode to maintain pipeline identity
                wrapped_nodes.append(PipelineNode(pipeline=node))
            else:
                wrapped_nodes.append(node)
        return wrapped_nodes

    @property
    def parameters(self) -> tuple:
        """Make Pipeline behave like a Node for nesting.

        Returns:
            Tuple of root argument names
        """
        return tuple(self.root_args)

    @property
    def output_names(self) -> tuple:
        """Get all output names from this pipeline.

        For nested pipelines, this allows parent pipelines to know
        what outputs are available.

        Returns:
            Tuple of all output names produced by this pipeline
        """
        outputs = []
        for node in self.nodes:
            if isinstance(node, (Pipeline, PipelineNode)):
                # Nested pipeline or PipelineNode
                node_outputs = node.output_name if hasattr(node, "output_name") else []
                if not isinstance(node_outputs, tuple):
                    node_outputs = (node_outputs,) if node_outputs else ()
                outputs.extend(node_outputs)
            else:
                outputs.append(node.output_name)
        return tuple(outputs)

    @property
    def output_name(self) -> tuple:
        """Deprecated alias for output_names. Use output_names instead.
        
        .. deprecated:: 
            Use :attr:`output_names` instead. This property is maintained for
            backward compatibility with the Node interface.

        Returns:
            Tuple of all output names produced by this pipeline
        """
        return self.output_names

    def _compute_required_nodes(
        self, output_names: Union[str, List[str], None]
    ) -> Optional[List[Node]]:
        """Compute minimal set of nodes needed to produce requested outputs.

        Delegates to graph builder for computation.

        Args:
            output_names: Output name(s) to compute, or None for all outputs

        Returns:
            List of nodes in execution order needed to produce outputs,
            or None if all nodes should be executed (output_names is None)

        Raises:
            ValueError: If any output_name is not found in the pipeline
        """
        graph_builder = NetworkXGraphBuilder()
        return graph_builder.compute_required_nodes(
            self.graph, self.execution_order, self.output_to_node, output_names
        )

    def run(
        self,
        inputs: Dict[str, Any],
        output_name: Union[str, List[str], None] = None,
        _ctx=None,  # @ideally this shouldn't be available here, but somehow differently. perhaps with a setter
    ) -> Dict[str, Any]:
        """Execute pipeline with given inputs.

        Delegates execution to the configured engine.

        Args:
            inputs: Dictionary mapping input parameter names to values
            output_name: Optional output name(s) to compute. Can be:
                - str: Single output name
                - List[str]: Multiple output names
                - None: Compute all outputs (default)
                Only the specified outputs will be returned, and only the
                minimal set of nodes needed will be executed.
            _ctx: Internal callback context (used for map operations)

        Returns:
            Dictionary containing only the requested outputs (or all outputs if None)

        Raises:
            ValueError: If output_name specifies a non-existent output
        """
        # Use effective engine to support inheritance
        engine = self.effective_engine
        return engine.run(self, inputs, output_name=output_name, _ctx=_ctx)

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

    # Fluent Builder Methods
    # ======================

    # Note: engine, cache, callbacks, and name are stored as direct attributes
    # They are set in __init__ and can be accessed/modified directly
    
    @property
    def backend(self) -> Optional[Engine]:
        """Deprecated alias for engine (kept for backward compatibility)."""
        return self.engine

    @backend.setter
    def backend(self, engine: Optional[Engine]) -> None:
        self.engine = engine

    # Fluent Builder Methods
    def with_engine(self, engine: "Engine") -> "Pipeline":
        """Configure pipeline with a specific engine.

        Returns the same pipeline instance for method chaining.

        Args:
            engine: Engine instance (HypernodesEngine, DaftEngine, etc.)

        Returns:
            Self for method chaining

        Example:
            >>> pipeline = Pipeline(nodes=[...]).with_engine(
            ...     HypernodesEngine(node_executor="threaded")
            ... )
        """
        self.engine = engine
        return self

    def with_backend(self, backend: "Engine") -> "Pipeline":
        """Deprecated alias for :meth:`with_engine`."""
        return self.with_engine(backend)

    def with_cache(self, cache: "Cache") -> "Pipeline":
        """Configure pipeline with a cache backend.

        Returns the same pipeline instance for method chaining.

        Args:
            cache: Cache instance (DiskCache, etc.)

        Returns:
            Self for method chaining

        Example:
            >>> from hypernodes import DiskCache
            >>> pipeline = Pipeline(nodes=[...]).with_cache(
            ...     DiskCache(path="./cache")
            ... )
        """
        self.cache = cache
        return self

    def with_callbacks(self, callbacks: List["PipelineCallback"]) -> "Pipeline":
        """Configure pipeline with lifecycle callbacks.

        Returns the same pipeline instance for method chaining.

        Args:
            callbacks: List of callback instances

        Returns:
            Self for method chaining

        Example:
            >>> from hypernodes.telemetry import ProgressCallback
            >>> pipeline = Pipeline(nodes=[...]).with_callbacks([
            ...     ProgressCallback()
            ... ])
        """
        self.callbacks = callbacks
        return self

    def with_name(self, name: str) -> "Pipeline":
        """Configure pipeline with a display name.

        Returns the same pipeline instance for method chaining.

        Args:
            name: Human-readable name for the pipeline (used in visualization)

        Returns:
            Self for method chaining

        Example:
            >>> pipeline = Pipeline(nodes=[...]).with_name("preprocessing")
        """
        self.name = name
        return self

    # Effective properties with inheritance support (for backward compatibility)
    @property
    def effective_engine(self):
        """Get effective engine (inherited from parent if not set).
        
        .. deprecated::
            Parent-based configuration inheritance is deprecated. In future versions,
            configuration will be managed through ExecutionContext. For now, if you need
            configuration inheritance, set the engine explicitly on each pipeline.
        """
        if self.engine is not None:
            return self.engine
        # Fallback to parent if available (deprecated)
        if self._parent is not None and hasattr(self._parent, "effective_engine"):
            import warnings
            warnings.warn(
                "Parent-based configuration inheritance is deprecated and will be removed in a future version. "
                "Set the engine explicitly on each pipeline instead.",
                DeprecationWarning,
                stacklevel=2
            )
            return self._parent.effective_engine
        # Default to HypernodesEngine
        from .engine import HypernodesEngine
        return HypernodesEngine()

    @property
    def effective_backend(self):
        """Deprecated alias for :meth:`effective_engine`."""
        return self.effective_engine

    @property
    def effective_cache(self):
        """Get effective cache (inherited from parent if not set).

        .. deprecated::
            Parent-based configuration inheritance is deprecated. Set the cache
            explicitly on each pipeline instead.

        Returns:
            Cache to use for caching
        """
        if self.cache is not None:
            return self.cache
        # Fallback to parent if available (deprecated)
        if self._parent is not None and hasattr(self._parent, "effective_cache"):
            import warnings
            warnings.warn(
                "Parent-based configuration inheritance is deprecated and will be removed in a future version. "
                "Set the cache explicitly on each pipeline instead.",
                DeprecationWarning,
                stacklevel=2
            )
            return self._parent.effective_cache
        return None

    @property
    def effective_callbacks(self):
        """Get effective callbacks (inherited from parent if not set).

        .. deprecated::
            Parent-based configuration inheritance is deprecated. Set the callbacks
            explicitly on each pipeline instead.

        Returns:
            List of callbacks to use
        """
        if self.callbacks:
            return self.callbacks
        # Fallback to parent if available (deprecated)
        if self._parent is not None and hasattr(self._parent, "effective_callbacks"):
            import warnings
            warnings.warn(
                "Parent-based configuration inheritance is deprecated and will be removed in a future version. "
                "Set the callbacks explicitly on each pipeline instead.",
                DeprecationWarning,
                stacklevel=2
            )
            return self._parent.effective_callbacks
        return []

    # Helper methods for map operations
    def _prepare_map_inputs( 
        self,
        inputs: Dict[str, Any],
        map_over: List[str],
        map_mode: str
    ) -> tuple[Dict[str, Any], Dict[str, Any], List[int]]:
        """Prepare inputs for map operation.
        
        Separates varying and fixed parameters, validates inputs.
        
        Args:
            inputs: All input parameters
            map_over: List of parameters that vary
            map_mode: "zip" or "product"
            
        Returns:
            Tuple of (varying_params, fixed_params, zip_lengths)
            
        Raises:
            ValueError: If validation fails
        """
        # Separate varying and fixed parameters
        varying_params = {}
        fixed_params = {}

        for key, value in inputs.items():
            if key in map_over:
                if not isinstance(value, list):
                    raise ValueError(
                        f"Parameter '{key}' is in map_over but value is not a list"
                    )
                varying_params[key] = value
            else:
                fixed_params[key] = value

        # Validate that all map_over parameters are present
        for param in map_over:
            if param not in varying_params:
                raise ValueError(f"Parameter '{param}' in map_over not found in inputs")
        
        # Validate zip mode lengths
        zip_lengths: List[int] = []
        if map_mode == "zip":
            zip_lengths = [len(lst) for lst in varying_params.values()]
            if zip_lengths and not all(length == zip_lengths[0] for length in zip_lengths):
                length_info = {k: len(v) for k, v in varying_params.items()}
                raise ValueError(
                    f"In zip mode, all lists must have the same length. "
                    f"Got lengths: {length_info}"
                )
        
        return varying_params, fixed_params, zip_lengths
    
    def _build_execution_plans(
        self,
        varying_params: Dict[str, Any],
        fixed_params: Dict[str, Any],
        map_mode: str,
        zip_lengths: List[int]
    ) -> List[Dict[str, Any]]:
        """Build list of execution plans based on map_mode.
        
        Args:
            varying_params: Parameters that vary across items
            fixed_params: Parameters that stay constant
            map_mode: "zip" or "product"
            zip_lengths: List lengths (for empty detection in zip mode)
            
        Returns:
            List of input dictionaries, one per execution
        """
        if map_mode == "zip":
            # Create execution plans by zipping
            if not varying_params or not zip_lengths or zip_lengths[0] == 0:
                # Empty case
                return []
            else:
                # Zip the varying parameters together
                param_names = list(varying_params.keys())
                param_lists = [varying_params[name] for name in param_names]
                return [
                    {**fixed_params, **{k: v for k, v in zip(param_names, values)}}
                    for values in zip(*param_lists)
                ]
        else:  # product mode
            # Create all combinations
            if not varying_params:
                return [fixed_params]
            else:
                param_names = list(varying_params.keys())
                param_lists = [varying_params[name] for name in param_names]
                return [
                    {**fixed_params, **{k: v for k, v in zip(param_names, values)}}
                    for values in itertools.product(*param_lists)
                ]

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

    # @this should create items from the map_over and map_mode using a helper function
    # @the just call engine.map(...) or something like that
    def map(
        self,
        inputs: Dict[str, Any],
        map_over: Union[str, List[str]],
        map_mode: str = "zip",
        output_name: Union[str, List[str], None] = None,
        return_format: str = "python",
        _ctx=None,
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

        # Use helper to prepare inputs
        varying_params, fixed_params, zip_lengths = self._prepare_map_inputs(
            inputs, map_over, map_mode
        )

        engine = self.effective_engine
        columnar_result = None
        columnar_error: Optional[Exception] = None
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
                    _ctx=_ctx,
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

            # Use helper to build execution plans
            execution_plans = self._build_execution_plans(
                varying_params, fixed_params, map_mode, zip_lengths
            )
        else:
            execution_plans = None

        # Execute pipeline for each plan with callbacks
        # Use provided context or create new one for map operation
        from .callbacks import CallbackContext

        if _ctx is None:
            ctx = CallbackContext()
            ctx.push_pipeline(self.id)
        else:
            # Use existing context (nested pipeline)
            ctx = _ctx
            ctx.push_pipeline(self.id)

        # Determine callbacks to use (support inheritance)
        callbacks = self.effective_callbacks

        # Set pipeline metadata so callbacks can access node information
        # Use same logic as engine._get_node_id() for consistency
        node_ids = []
        for n in self.execution_order:
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
                "total_nodes": len(self.execution_order),
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
                _ctx=ctx,
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
        if self.engine is not None and hasattr(self.engine, 'shutdown'):
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
