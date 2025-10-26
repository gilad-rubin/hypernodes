"""Pipeline class for managing and executing DAGs of nodes."""

import functools
import itertools
import time
from typing import Any, Dict, List, Optional, Set, Union

import networkx as nx

from .backend import Backend, LocalBackend
from .cache import Cache
from .callbacks import PipelineCallback
from .exceptions import CycleError, DependencyError
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
        name: Optional[str] = None,
    ):
        """Initialize a PipelineNode wrapper.

        Args:
            pipeline: Pipeline to wrap
            input_mapping: Maps {outer_name: inner_name} for inputs
            output_mapping: Maps {inner_name: outer_name} for outputs
            map_over: Parameter name(s) to map over (from outer perspective)
            name: Optional name for this node (displayed in visualizations)
        """
        # Don't call super().__init__() since we have a different initialization pattern
        # We'll override the necessary properties instead
        self.pipeline = pipeline
        self.input_mapping = input_mapping or {}
        self.output_mapping = output_mapping or {}
        self.map_over = map_over
        self.name = name

        # Make map_over a list if it's a string
        if isinstance(self.map_over, str):
            self.map_over = [self.map_over]

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
        # Get execution context if available (set by backend)
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
                    map_mode="zip",
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
                    map_mode="zip",
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
        backend: Backend for executing the pipeline (default: LocalBackend)
        output_to_node: Mapping from output names to nodes
        graph: NetworkX DiGraph representing dependencies (cached)
        execution_order: Topologically sorted list of nodes (cached)
        root_args: External input parameters needed by the pipeline
    """

    def __init__(
        self,
        nodes: List[Union[Node, "Pipeline"]],
        backend: Backend = None,
        cache: Optional[Cache] = None,
        callbacks: Optional[List[PipelineCallback]] = None,
        name: Optional[str] = None,
        parent: Optional["Pipeline"] = None,
    ):
        """Initialize a Pipeline from a list of nodes.

        Args:
            nodes: List of Node instances or Pipeline instances (which are auto-wrapped)
            backend: Backend for execution (default: LocalBackend())
            cache: Cache backend for result caching (default: None, no caching)
            callbacks: List of callbacks for lifecycle hooks (default: None)
            name: Human-readable name for the pipeline (used in visualization)
            parent: Parent pipeline for configuration inheritance (internal)

        Raises:
            CycleError: If a cycle is detected in the dependency graph
            DependencyError: If a dependency cannot be satisfied
        """
        # Auto-wrap Pipeline instances in PipelineNode for consistency
        wrapped_nodes = []
        for node in nodes:
            if isinstance(node, Pipeline):
                # Wrap pipeline in PipelineNode to maintain pipeline identity
                wrapped_nodes.append(PipelineNode(pipeline=node))
            else:
                wrapped_nodes.append(node)

        self.nodes = wrapped_nodes
        self.backend = backend
        self.cache = cache
        self.callbacks = callbacks
        self._parent = parent
        self.name = name

        # Generate deterministic pipeline ID based on object identity
        # Same pipeline instance always gets same ID for callback tracking
        import builtins

        self.id = f"pipeline_{builtins.id(self)}"

        # Build output_name -> Node mapping (inspired by pipefunc)
        self.output_to_node = {}
        for node in self.nodes:  # Use wrapped nodes (PipelineNode for nested pipelines)
            # Handle nested pipelines and PipelineNodes
            if isinstance(node, (Pipeline, PipelineNode)):
                # Get outputs from the node
                outputs = node.output_name if hasattr(node, "output_name") else []
                if not isinstance(outputs, tuple):
                    outputs = (outputs,)
                for output in outputs:
                    self.output_to_node[output] = node
            else:
                self.output_to_node[node.output_name] = node

        # Validate at construction
        self._validate()

    @functools.cached_property
    def graph(self) -> nx.DiGraph:
        """Build dependency graph using NetworkX.

        Constructs a directed graph where:
        - Nodes are functions or input parameters
        - Edges represent dependencies (parameter → function)

        Returns:
            NetworkX DiGraph representing the pipeline dependencies
        """
        g = nx.DiGraph()

        for node in self.nodes:
            g.add_node(node)

            # Get parameters for this node
            if isinstance(node, (Pipeline, PipelineNode)):
                # Nested pipeline or PipelineNode - use parameters property
                params = (
                    node.parameters if hasattr(node, "parameters") else node.root_args
                )
            else:
                params = node.parameters

            # Add edges based on dependencies
            for param in params:
                if param in self.output_to_node:
                    # Dependency: another node's output
                    producer = self.output_to_node[param]
                    g.add_edge(producer, node)
                else:
                    # Root argument: external input
                    # Add string node for input parameter
                    g.add_edge(param, node)

        return g

    @functools.cached_property
    def execution_order(self) -> List[Node]:
        """Get topological execution order using NetworkX.

        Uses topological sort to determine the order in which nodes
        should be executed to satisfy all dependencies.

        Returns:
            List of nodes in execution order

        Raises:
            CycleError: If a cycle is detected in the graph
        """
        try:
            # Use topological_sort from networkx
            sorted_nodes = list(nx.topological_sort(self.graph))
            # Filter to only Node/PipelineNode instances (not string inputs)
            # Note: We don't include raw Pipeline because they're auto-wrapped in PipelineNode
            return [n for n in sorted_nodes if isinstance(n, (Node, PipelineNode))]
        except nx.NetworkXError as e:
            raise CycleError(f"Cycle detected in pipeline: {e}") from e

    @property
    def root_args(self) -> List[str]:
        """Get external input parameters required by this pipeline.

        Root arguments are parameters that are not produced by any node
        in the pipeline and must be provided as inputs.

        Returns:
            List of parameter names that are external inputs
        """
        all_params: Set[str] = set()
        all_outputs: Set[str] = set()

        for node in self.nodes:
            if isinstance(node, (Pipeline, PipelineNode)):
                # Nested pipeline or PipelineNode
                all_params.update(
                    node.parameters if hasattr(node, "parameters") else node.root_args
                )
                # Get outputs
                outputs = node.output_name if hasattr(node, "output_name") else []
                if not isinstance(outputs, tuple):
                    outputs = (outputs,) if outputs else ()
                all_outputs.update(outputs)
            else:
                all_params.update(node.parameters)
                all_outputs.add(node.output_name)

        return list(all_params - all_outputs)

    @property
    def parameters(self) -> tuple:
        """Make Pipeline behave like a Node for nesting.

        Returns:
            Tuple of root argument names
        """
        return tuple(self.root_args)

    @property
    def output_name(self) -> tuple:
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

    def _validate(self) -> None:
        """Validate pipeline integrity.

        Checks:
        1. All dependencies can be satisfied
        2. No cycles exist in the dependency graph

        Raises:
            DependencyError: If a dependency cannot be satisfied
            CycleError: If a cycle is detected
        """
        root_args_set = set(self.root_args)

        # Check for missing dependencies
        for node in self.nodes:
            if isinstance(node, (Pipeline, PipelineNode)):
                params = (
                    node.parameters if hasattr(node, "parameters") else node.root_args
                )
            else:
                params = node.parameters

            for param in params:
                if param not in self.output_to_node and param not in root_args_set:
                    raise DependencyError(
                        f"Node {node} requires parameter '{param}' but it's not "
                        f"provided by any node or as an input"
                    )

        # Check for cycles (accessing execution_order will raise if cycle exists)
        _ = self.execution_order

    def _compute_required_nodes(
        self, output_names: Union[str, List[str], None]
    ) -> Optional[List[Node]]:
        """Compute minimal set of nodes needed to produce requested outputs.

        Args:
            output_names: Output name(s) to compute, or None for all outputs

        Returns:
            List of nodes in execution order needed to produce outputs,
            or None if all nodes should be executed (output_names is None)

        Raises:
            ValueError: If any output_name is not found in the pipeline
        """
        if output_names is None:
            return None

        # Normalize to list
        if isinstance(output_names, str):
            output_names = [output_names]

        # Validate all output names exist
        for output_name in output_names:
            if output_name not in self.output_to_node:
                available = ", ".join(sorted(self.output_to_node.keys()))
                raise ValueError(
                    f"Output '{output_name}' not found in pipeline. "
                    f"Available outputs: {available}"
                )

        # Find nodes that produce the requested outputs
        target_nodes = set()
        for output_name in output_names:
            target_nodes.add(self.output_to_node[output_name])

        # Use NetworkX to find all ancestors (dependencies) of target nodes
        required_nodes = set()
        for target_node in target_nodes:
            # Add the target node itself
            required_nodes.add(target_node)
            # Add all its ancestors (dependencies)
            ancestors = nx.ancestors(self.graph, target_node)
            # Filter to only Node/PipelineNode instances (not string inputs)
            for ancestor in ancestors:
                if isinstance(ancestor, (Node, PipelineNode)):
                    required_nodes.add(ancestor)

        # Return nodes in execution order
        return [n for n in self.execution_order if n in required_nodes]

    def run(
        self,
        inputs: Dict[str, Any],
        output_name: Union[str, List[str], None] = None,
        _ctx=None,
    ) -> Dict[str, Any]:
        """Execute pipeline with given inputs.

        Delegates execution to the configured backend.

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
        # Use effective backend to support inheritance
        backend = self.effective_backend
        return backend.run(self, inputs, _ctx, output_name=output_name)

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

    def with_backend(self, backend: "Backend") -> "Pipeline":
        """Configure pipeline with a specific backend.

        Returns the same pipeline instance for method chaining.

        Args:
            backend: Backend instance (LocalBackend, ModalBackend, etc.)

        Returns:
            Self for method chaining

        Example:
            >>> pipeline = Pipeline(nodes=[...]).with_backend(
            ...     ModalBackend(image=my_image, gpu="A100")
            ... )
        """
        self.backend = backend
        return self

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

    @property
    def effective_backend(self):
        """Get effective backend (inherited from parent if not set).

        Returns:
            Backend to use for execution
        """
        if self.backend is not None:
            return self.backend
        if self._parent is not None:
            return self._parent.effective_backend
        return LocalBackend()  # Default

    @property
    def effective_cache(self):
        """Get effective cache (inherited from parent if not set).

        Returns:
            Cache to use for caching
        """
        if self.cache is not None:
            return self.cache
        if self._parent is not None:
            return self._parent.effective_cache
        return None  # Default: no caching

    @property
    def effective_callbacks(self):
        """Get effective callbacks (inherited from parent if not set).

        Returns:
            List of callbacks to use
        """
        if self.callbacks is not None:
            return self.callbacks
        if self._parent is not None:
            return self._parent.effective_callbacks
        return []  # Default: no callbacks

    def as_node(
        self,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        map_over: Optional[Union[str, List[str]]] = None,
        name: Optional[str] = None,
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
            name: Optional name for this node (displayed in visualizations)

        Returns:
            PipelineNode that wraps this pipeline with the specified mapping

        Example:
            >>> inner = Pipeline(nodes=[clean_text])  # expects "passage"
            >>> adapted = inner.as_node(
            ...     input_mapping={"document": "passage"},  # outer -> inner
            ...     output_mapping={"cleaned": "processed"}  # inner -> outer
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
            name=name,
        )

    def map(
        self,
        inputs: Dict[str, Any],
        map_over: Union[str, List[str]],
        map_mode: str = "zip",
        output_name: Union[str, List[str], None] = None,
        _ctx=None,
    ) -> Dict[str, List[Any]]:
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
            output_name: Optional output name(s) to compute. Can be:
                - str: Single output name
                - List[str]: Multiple output names
                - None: Compute all outputs (default)
                Only the specified outputs will be returned, and only the
                minimal set of nodes needed will be executed.

        Returns:
            Dictionary where each output name maps to a list of results,
            one per execution. Only requested outputs are included.

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
        # Normalize map_over to list
        if isinstance(map_over, str):
            map_over = [map_over]

        # Validate map_mode
        if map_mode not in ("zip", "product"):
            raise ValueError(f"map_mode must be 'zip' or 'product', got '{map_mode}'")

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

        # Generate execution plans based on map_mode
        if map_mode == "zip":
            # Validate all lists have same length
            lengths = [len(lst) for lst in varying_params.values()]
            if lengths and not all(length == lengths[0] for length in lengths):
                raise ValueError(
                    f"In zip mode, all lists must have the same length. "
                    f"Got lengths: {dict(zip(varying_params.keys(), lengths))}"
                )

            # Create execution plans by zipping
            if not varying_params or not lengths or lengths[0] == 0:
                # Empty case
                execution_plans = []
            else:
                # Zip the varying parameters together
                param_names = list(varying_params.keys())
                param_lists = [varying_params[name] for name in param_names]
                execution_plans = [
                    {**fixed_params, **dict(zip(param_names, values))}
                    for values in zip(*param_lists)
                ]

        else:  # product mode
            # Create all combinations
            if not varying_params:
                execution_plans = [fixed_params]
            else:
                param_names = list(varying_params.keys())
                param_lists = [varying_params[name] for name in param_names]
                execution_plans = [
                    {**fixed_params, **dict(zip(param_names, values))}
                    for values in itertools.product(*param_lists)
                ]

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
        # Use same logic as backend._get_node_id() for consistency
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
        total_items = len(execution_plans)
        map_start_time = time.time()
        for callback in callbacks:
            callback.on_map_start(total_items, ctx)

        results_list = []
        for idx, plan in enumerate(execution_plans):
            # Mark that we're in a map operation (for cache callbacks)
            ctx.set("_in_map", True)
            ctx.set("_map_item_index", idx)
            ctx.set("_map_item_start_time", time.time())

            # Execute the pipeline for this item (pass context for callbacks)
            # The backend will fire on_map_item_start/end/cached as appropriate
            result = self.run(plan, output_name=output_name, _ctx=ctx)
            results_list.append(result)

            # Clear map context
            ctx.set("_in_map", False)
            ctx.set("_map_item_index", None)
            ctx.set("_map_item_start_time", None)

        # Trigger map end callbacks
        map_duration = time.time() - map_start_time
        for callback in callbacks:
            callback.on_map_end(map_duration, ctx)

        ctx.pop_pipeline()

        # Transpose results: from list of dicts to dict of lists
        if not results_list:
            # Empty case: return empty lists for each output
            # Determine output keys from pipeline structure
            output_keys = []
            for node in self.nodes:
                if isinstance(node, Pipeline):
                    # Nested pipeline - add all its outputs
                    for inner_node in node.nodes:
                        output_keys.append(inner_node.output_name)
                else:
                    output_keys.append(node.output_name)
            return {key: [] for key in output_keys}

        # Collect all output keys from first result
        output_keys = results_list[0].keys()

        # Transpose: dict of lists
        return {key: [result[key] for result in results_list] for key in output_keys}

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
