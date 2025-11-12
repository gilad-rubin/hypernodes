"""Graph builder abstraction for pipeline dependency graphs.

This module provides the GraphBuilder abstraction and NetworkX implementation
for constructing and analyzing pipeline dependency graphs.

Key responsibilities:
- Build dependency graph from nodes
- Validate graph (cycles, missing dependencies)
- Compute topological execution order
- Compute root arguments (external inputs)
- Compute required nodes for selective execution
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

import networkx as nx

from .exceptions import CycleError, DependencyError

if TYPE_CHECKING:
    from .node import Node


@dataclass
class GraphResult:
    """Result of building a pipeline graph.
    
    Contains all computed information about the pipeline structure.
    
    Attributes:
        graph: NetworkX DiGraph representing dependencies
        output_to_node: Mapping from output names to nodes that produce them
        execution_order: Topologically sorted list of nodes
        root_args: External input parameters required by pipeline
    """
    graph: nx.DiGraph
    output_to_node: Dict[str, "Node"]
    execution_order: List["Node"]
    root_args: List[str]


class GraphBuilder(ABC):
    """Abstract base class for building and analyzing pipeline dependency graphs.
    
    This abstraction allows different graph implementations while maintaining
    a consistent interface for the Pipeline class.
    """

    @abstractmethod
    def build_graph(
        self,
        nodes: List["Node"]
    ) -> GraphResult:
        """Build dependency graph from nodes.
        
        Performs all graph construction, validation, and analysis in one operation.
        
        Args:
            nodes: List of Node instances (already wrapped, PipelineNodes for nested pipelines)
            
        Returns:
            GraphResult containing graph, output_to_node, execution_order, root_args
            
        Raises:
            CycleError: If cycle detected in graph
            DependencyError: If dependency cannot be satisfied
        """
        pass

    @abstractmethod
    def compute_required_nodes(
        self,
        graph: Any,
        execution_order: List["Node"],
        output_to_node: Dict[str, "Node"],
        output_names: Union[str, List[str], None]
    ) -> Optional[List["Node"]]:
        """Compute minimal set of nodes needed for requested outputs.
        
        Args:
            graph: Graph object from build_graph()
            execution_order: Full execution order from compute_execution_order()
            output_to_node: Mapping from output names to nodes
            output_names: Requested output name(s), or None for all
            
        Returns:
            List of required nodes in execution order, or None if all needed
            
        Raises:
            ValueError: If output_name not found in pipeline
        """
        pass


class NetworkXGraphBuilder(GraphBuilder):
    """NetworkX-based graph builder implementation.
    
    Uses NetworkX DiGraph for dependency representation and topological sorting.
    """

    def build_graph(
        self,
        nodes: List["Node"]
    ) -> GraphResult:
        """Build NetworkX dependency graph with full analysis.
        
        Performs all graph operations in one pass:
        1. Builds output_to_node mapping
        2. Constructs dependency graph
        3. Validates (cycles, missing dependencies)
        4. Computes execution order
        5. Computes root arguments
        
        Args:
            nodes: List of Node instances (already wrapped)
            
        Returns:
            GraphResult with all computed information
            
        Raises:
            CycleError: If cycle detected
            DependencyError: If dependency missing
        """
        # 1. Build output_to_node mapping
        output_to_node = self._build_output_mapping(nodes)
        
        # 2. Build graph
        g = nx.DiGraph()
        for node in nodes:
            g.add_node(node)
            params = self._get_node_parameters(node)
            
            for param in params:
                if param in output_to_node:
                    # Dependency: another node's output
                    producer = output_to_node[param]
                    g.add_edge(producer, node)
                else:
                    # Root argument: external input
                    g.add_edge(param, node)
        
        # 3. Compute execution order (validates cycles)
        execution_order = self._compute_execution_order(g)
        
        # 4. Compute root args
        root_args = self._compute_root_args(nodes, output_to_node)
        
        # 5. Validate dependencies
        self._validate_dependencies(g, nodes, root_args, output_to_node)
        
        return GraphResult(
            graph=g,
            output_to_node=output_to_node,
            execution_order=execution_order,
            root_args=root_args
        )
    
    def _build_output_mapping(self, nodes: List["Node"]) -> Dict[str, "Node"]:
        """Build output_name -> Node mapping.
        
        Args:
            nodes: List of Node instances (already wrapped)
            
        Returns:
            Dictionary mapping output names to nodes
        """
        output_to_node = {}
        for node in nodes:
            outputs = self._get_node_outputs(node)
            for output in outputs:
                output_to_node[output] = node
        return output_to_node

    def _compute_execution_order(self, graph: nx.DiGraph) -> List["Node"]:
        """Compute topological execution order using NetworkX.
        
        Uses topological sort to determine the order in which nodes
        should be executed to satisfy all dependencies.
        
        Args:
            graph: NetworkX DiGraph
            
        Returns:
            List of nodes in execution order
            
        Raises:
            CycleError: If a cycle is detected in the graph
        """
        try:
            # Use topological_sort from networkx
            sorted_nodes = list(nx.topological_sort(graph))
            # Filter to only Node instances (not string inputs)
            # Import here to avoid circular dependency
            from .node import Node
            return [n for n in sorted_nodes if isinstance(n, Node)]
        except nx.NetworkXError as e:
            raise CycleError(f"Cycle detected in pipeline: {e}") from e

    def _compute_root_args(
        self,
        nodes: List["Node"],
        output_to_node: Dict[str, "Node"]
    ) -> List[str]:
        """Compute external input parameters required by pipeline.
        
        Root arguments are parameters that are not produced by any node
        in the pipeline and must be provided as inputs.
        
        Args:
            nodes: List of Node instances
            output_to_node: Mapping from output names to nodes
            
        Returns:
            List of parameter names that are external inputs
        """
        all_params: Set[str] = set()
        all_outputs: Set[str] = set()

        for node in nodes:
            # Get parameters
            params = self._get_node_parameters(node)
            all_params.update(params)
            
            # Get outputs
            outputs = self._get_node_outputs(node)
            all_outputs.update(outputs)

        return list(all_params - all_outputs)

    def compute_required_nodes(
        self,
        graph: nx.DiGraph,
        execution_order: List["Node"],
        output_to_node: Dict[str, "Node"],
        output_names: Union[str, List[str], None]
    ) -> Optional[List["Node"]]:
        """Compute minimal set of nodes needed to produce requested outputs.
        
        Args:
            graph: NetworkX DiGraph from build_graph()
            execution_order: Full execution order
            output_to_node: Mapping from output names to nodes
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
            if output_name not in output_to_node:
                available = ", ".join(sorted(output_to_node.keys()))
                raise ValueError(
                    f"Output '{output_name}' not found in pipeline. "
                    f"Available outputs: {available}"
                )

        # Find nodes that produce the requested outputs
        target_nodes = set()
        for output_name in output_names:
            target_nodes.add(output_to_node[output_name])

        # Use NetworkX to find all ancestors (dependencies) of target nodes
        required_nodes = set()
        for target_node in target_nodes:
            # Add the target node itself
            required_nodes.add(target_node)
            # Add all its ancestors (dependencies)
            ancestors = nx.ancestors(graph, target_node)
            # Filter to only Node instances (not string inputs)
            from .node import Node
            for ancestor in ancestors:
                if isinstance(ancestor, Node):
                    required_nodes.add(ancestor)

        # Return nodes in execution order
        return [n for n in execution_order if n in required_nodes]

    def _validate_dependencies(
        self,
        graph: nx.DiGraph,
        nodes: List["Node"],
        root_args: List[str],
        output_to_node: Dict[str, "Node"]
    ) -> None:
        """Validate pipeline graph integrity.
        
        Checks:
        1. All dependencies can be satisfied
        2. No cycles exist in the dependency graph
        
        Args:
            graph: NetworkX DiGraph
            nodes: List of Node instances
            root_args: Root arguments
            output_to_node: Mapping from output names to nodes
            
        Raises:
            DependencyError: If a dependency cannot be satisfied
            CycleError: If a cycle is detected (via _compute_execution_order)
        """
        root_args_set = set(root_args)

        # Check for missing dependencies
        for node in nodes:
            params = self._get_node_parameters(node)
            for param in params:
                if param not in output_to_node and param not in root_args_set:
                    raise DependencyError(
                        f"Node {node} requires parameter '{param}' but it's not "
                        f"provided by any node or as an input"
                    )

        # Check for cycles (will raise CycleError if cycle exists)
        try:
            _ = self._compute_execution_order(graph)
        except CycleError:
            raise  # Re-raise with original message

    def _get_node_parameters(self, node: "Node") -> tuple:
        """Get parameters for a node (handles regular nodes and PipelineNodes)."""
        # Check if it's a PipelineNode or nested Pipeline
        if hasattr(node, "pipeline"):
            # PipelineNode - use its parameters property
            pipeline = getattr(node, "pipeline", None)
            if hasattr(node, "parameters"):
                return node.parameters
            elif pipeline and hasattr(pipeline, "root_args"):
                return tuple(pipeline.root_args)
            return ()
        elif hasattr(node, "parameters"):
            # Regular Node
            return node.parameters
        else:
            return ()

    def _get_node_outputs(self, node: "Node") -> List[str]:
        """Get output names for a node (handles regular nodes and PipelineNodes)."""
        if hasattr(node, "output_name"):
            outputs = node.output_name
            if not isinstance(outputs, tuple):
                outputs = (outputs,) if outputs else ()
            return list(outputs)
        return []
