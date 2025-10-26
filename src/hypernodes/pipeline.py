"""Pipeline class for managing and executing DAGs of nodes."""
import functools
import itertools
import time
from typing import List, Dict, Any, Set, Union, Optional
import networkx as nx

from .node import Node
from .backend import LocalBackend
from .cache import Cache
from .callbacks import PipelineCallback
from .exceptions import CycleError, DependencyError


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
        nodes: List[Node],
        backend: LocalBackend = None,
        cache: Optional[Cache] = None,
        callbacks: Optional[List[PipelineCallback]] = None,
        id: Optional[str] = None
    ):
        """Initialize a Pipeline from a list of nodes.
        
        Args:
            nodes: List of Node instances (or nested Pipelines)
            backend: Backend for execution (default: LocalBackend())
            cache: Cache backend for result caching (default: None, no caching)
            callbacks: List of callbacks for lifecycle hooks (default: None)
            id: Pipeline identifier for callbacks (default: auto-generated)
            
        Raises:
            CycleError: If a cycle is detected in the dependency graph
            DependencyError: If a dependency cannot be satisfied
        """
        self.nodes = nodes
        self.backend = backend or LocalBackend()
        self.cache = cache
        self.callbacks = callbacks if callbacks is not None else None
        # Generate pipeline ID (handle shadowing of built-in id())
        import builtins
        self.id = id if id is not None else f"pipeline_{builtins.id(self)}"
        
        # Build output_name -> Node mapping (inspired by pipefunc)
        self.output_to_node = {}
        for node in nodes:
            # Handle nested pipelines
            if isinstance(node, Pipeline):
                # Nested pipeline has multiple outputs
                for inner_node in node.nodes:
                    self.output_to_node[inner_node.output_name] = node
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
            if isinstance(node, Pipeline):
                # Nested pipeline - use its root_args
                params = node.root_args
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
            # Filter to only Node/Pipeline instances (not string inputs)
            return [n for n in sorted_nodes if isinstance(n, (Node, Pipeline))]
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
            if isinstance(node, Pipeline):
                # Nested pipeline
                all_params.update(node.root_args)
                for inner_node in node.nodes:
                    all_outputs.add(inner_node.output_name)
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
            if isinstance(node, Pipeline):
                # Nested pipeline
                for inner_node in node.nodes:
                    outputs.append(inner_node.output_name)
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
            if isinstance(node, Pipeline):
                params = node.root_args
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
    
    def run(self, inputs: Dict[str, Any], _ctx=None) -> Dict[str, Any]:
        """Execute pipeline with given inputs.
        
        Delegates execution to the configured backend.
        
        Args:
            inputs: Dictionary mapping input parameter names to values
            _ctx: Internal callback context (used for map operations)
            
        Returns:
            Dictionary containing outputs from all nodes (not inputs)
        """
        return self.backend.run(self, inputs, _ctx)
    
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
        return f"Pipeline({len(self.nodes)} nodes)"
    
    def __hash__(self) -> int:
        """Make Pipeline hashable for use in networkx graphs."""
        return id(self)
    
    def __eq__(self, other) -> bool:
        """Check equality based on identity."""
        return self is other
    
    def map(
        self,
        inputs: Dict[str, Any],
        map_over: Union[str, List[str]],
        map_mode: str = "zip"
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
        
        Returns:
            Dictionary where each output name maps to a list of results,
            one per execution.
        
        Raises:
            ValueError: If map_mode is "zip" and list lengths don't match
        
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
                raise ValueError(
                    f"Parameter '{param}' in map_over not found in inputs"
                )
        
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
        # Create callback context for map operation
        from .callbacks import CallbackContext
        ctx = CallbackContext()
        ctx.push_pipeline(self.id)
        
        # Determine callbacks to use
        callbacks = self.callbacks if self.callbacks is not None else []
        
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
            result = self.run(plan, _ctx=ctx)
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
        return {
            key: [result[key] for result in results_list]
            for key in output_keys
        }
