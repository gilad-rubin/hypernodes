"""Pipeline class for managing and executing DAGs of nodes."""
import functools
from typing import List, Dict, Any, Set
import networkx as nx

from .node import Node
from .backend import LocalBackend
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
    
    def __init__(self, nodes: List[Node], backend: LocalBackend = None):
        """Initialize a Pipeline from a list of nodes.
        
        Args:
            nodes: List of Node instances (or nested Pipelines)
            backend: Backend for execution (default: LocalBackend())
            
        Raises:
            CycleError: If a cycle is detected in the dependency graph
            DependencyError: If a dependency cannot be satisfied
        """
        self.nodes = nodes
        self.backend = backend or LocalBackend()
        
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
    
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pipeline with given inputs.
        
        Delegates execution to the configured backend.
        
        Args:
            inputs: Dictionary mapping input parameter names to values
            
        Returns:
            Dictionary containing outputs from all nodes (not inputs)
        """
        return self.backend.run(self, inputs)
    
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
