"""Graph builder abstraction for pipeline dependency graphs.

This module provides the GraphBuilder abstraction and SimpleGraphBuilder implementation
for constructing and analyzing pipeline dependency graphs.

Key responsibilities:
- Build dependency graph from nodes
- Validate graph (cycles, missing dependencies)
- Compute topological execution order
- Compute root arguments (external inputs)
- Compute required nodes for selective execution
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

from .exceptions import CycleError, DependencyError

if TYPE_CHECKING:
    from .node import Node


@dataclass
class GraphResult:
    """Result of building a pipeline graph.

    Contains all computed information about the pipeline structure.

    Attributes:
        output_to_node: Mapping from output names to nodes that produce them
        execution_order: Topologically sorted list of nodes
        root_args: External input parameters required by pipeline
        available_output_names: All output names the pipeline can produce
        dependencies: Mapping from each node to its direct dependencies (other nodes)
        required_outputs: Mapping from PipelineNode to list of required output names.
                         This stores the optimization information without mutating nodes.
        branch_gates: Set of output names that are gate signals (from BranchNodes)
        gate_dependencies: Mapping from node to list of gate signals it depends on
        exclusive_producers: Mapping from output name to list of nodes that produce it
                            (only populated when multiple producers exist in exclusive branches)
    """

    output_to_node: Dict[str, "Node"]
    execution_order: List["Node"]
    root_args: List[str]
    available_output_names: List[str]
    dependencies: Dict["Node", List["Node"]]
    required_outputs: Dict["Node", Optional[List[str]]]
    branch_gates: Set[str] = field(default_factory=set)
    gate_dependencies: Dict["Node", List[str]] = field(default_factory=dict)
    exclusive_producers: Dict[str, List["Node"]] = field(default_factory=dict)

    def get_required_nodes(
        self, output_names: Union[str, List[str], None]
    ) -> Optional[List["Node"]]:
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
        # Include ALL exclusive producers for outputs in exclusive branches
        target_nodes = set()
        for output_name in output_names:
            target_nodes.add(self.output_to_node[output_name])
            # Also include all exclusive producers for this output
            if output_name in self.exclusive_producers:
                target_nodes.update(self.exclusive_producers[output_name])

        # Traverse dependencies to find all required nodes
        required_nodes = set()

        def add_dependencies(node: "Node"):
            """Recursively add node and all its dependencies."""
            if node in required_nodes:
                return
            required_nodes.add(node)
            for dep in self.dependencies.get(node, []):
                add_dependencies(dep)

        for target_node in target_nodes:
            add_dependencies(target_node)

        # Return nodes in execution order
        return [n for n in self.execution_order if n in required_nodes]


class GraphBuilder(ABC):
    """Abstract base class for building and analyzing pipeline dependency graphs.

    This abstraction allows different graph implementations while maintaining
    a consistent interface for the Pipeline class.
    """

    @abstractmethod
    def build_graph(self, nodes: List["Node"]) -> GraphResult:
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
        dependencies: Dict["Node", List["Node"]],
        execution_order: List["Node"],
        output_to_node: Dict[str, "Node"],
        output_names: Union[str, List[str], None],
    ) -> Optional[List["Node"]]:
        """Compute minimal set of nodes needed for requested outputs.

        Args:
            dependencies: Mapping from each node to its dependencies
            execution_order: Full execution order
            output_to_node: Mapping from output names to nodes
            output_names: Requested output name(s), or None for all

        Returns:
            List of required nodes in execution order, or None if all needed

        Raises:
            ValueError: If output_name not found in pipeline
        """
        pass


class SimpleGraphBuilder(GraphBuilder):
    """Simple graph builder without NetworkX dependency.

    Uses custom topological sort and dependency tracking.
    Handles nested pipelines (PipelineNode) as special nodes.
    """

    def build_graph(self, nodes: List["Node"]) -> GraphResult:
        """Build dependency graph with custom implementation.

        Performs:
        1. Identifies branch nodes and validates targets
        2. Builds output_to_node mapping (with exclusive branch handling)
        3. Computes dependencies for each node (including gate dependencies)
        4. Validates (cycles, missing dependencies)
        5. Computes topological execution order
        6. Computes root arguments

        Args:
            nodes: List of Node instances

        Returns:
            GraphResult with all computed information

        Raises:
            CycleError: If cycle detected
            DependencyError: If dependency missing or branch target missing
        """
        from .branch import BranchNode
        from .node import Node

        # 0. Identify branch nodes and build node name lookup
        branch_nodes: List[BranchNode] = []
        node_by_name: Dict[str, Node] = {}
        branch_gates: Set[str] = set()
        gate_dependencies: Dict[Node, List[str]] = {}
        exclusive_producers: Dict[str, List[Node]] = {}

        for node in nodes:
            node_by_name[node.name] = node
            if hasattr(node, "_is_branch") and node._is_branch:
                branch_nodes.append(node)
                # Register gate outputs
                branch_gates.add(node.true_gate)
                branch_gates.add(node.false_gate)

        # 1. Validate branch targets exist
        for branch in branch_nodes:
            true_target = branch.when_true_name
            false_target = branch.when_false_name
            if true_target not in node_by_name:
                raise DependencyError(
                    f"Branch '{branch.name}' targets '{true_target}' (when_true) "
                    f"which is not in the pipeline. Available nodes: {list(node_by_name.keys())}"
                )
            if false_target not in node_by_name:
                raise DependencyError(
                    f"Branch '{branch.name}' targets '{false_target}' (when_false) "
                    f"which is not in the pipeline. Available nodes: {list(node_by_name.keys())}"
                )

        # 2. Build gate dependencies: which gate signals does each node depend on?
        # Target nodes implicitly depend on the branch's gate output
        for branch in branch_nodes:
            true_target = node_by_name[branch.when_true_name]
            false_target = node_by_name[branch.when_false_name]

            # Add gate dependency for true target
            if true_target not in gate_dependencies:
                gate_dependencies[true_target] = []
            gate_dependencies[true_target].append(branch.true_gate)

            # Add gate dependency for false target
            if false_target not in gate_dependencies:
                gate_dependencies[false_target] = []
            gate_dependencies[false_target].append(branch.false_gate)

        # 3. Build output_to_node mapping (handle exclusive branches)
        output_to_node: Dict[str, Node] = {}
        for node in nodes:
            outputs = node.output_name
            if isinstance(outputs, str):
                outputs = (outputs,)
            for output in outputs:
                # Skip gate outputs in public output mapping (they're internal)
                if output in branch_gates:
                    continue

                if output in output_to_node:
                    existing_producer = output_to_node[output]
                    # Check if both producers are in exclusive branches
                    if self._are_exclusive(node, existing_producer, gate_dependencies):
                        # Allow multiple producers in exclusive branches
                        if output not in exclusive_producers:
                            exclusive_producers[output] = [existing_producer]
                        exclusive_producers[output].append(node)
                        # Keep the first producer in output_to_node for now
                        # (execution will pick the right one based on gate)
                    else:
                        raise DependencyError(
                            f"Multiple nodes produce output '{output}': "
                            f"{existing_producer.name} and {node.name}. "
                            f"This is only allowed when producers are in mutually exclusive branches."
                        )
                else:
                    output_to_node[output] = node

        # 4. Build dependencies: node -> [nodes it depends on]
        dependencies: Dict[Node, List[Node]] = {}
        for node in nodes:
            node_deps_set: Set[Node] = set()

            # Get parameters this node needs
            params = node.root_args

            for param in params:
                if param in output_to_node:
                    producer = output_to_node[param]
                    if producer != node:
                        node_deps_set.add(producer)
                # If param not in output_to_node, it's an external input (root_arg)

            # Add branch node as dependency if this node has gate dependencies
            if node in gate_dependencies:
                for gate in gate_dependencies[node]:
                    # Find the branch that produces this gate
                    for branch in branch_nodes:
                        if gate == branch.true_gate or gate == branch.false_gate:
                            if branch != node:
                                node_deps_set.add(branch)
                            break

            dependencies[node] = list(node_deps_set)

        # 5. Validate: check for missing dependencies and cycles
        self._validate_dependencies(nodes, dependencies, output_to_node)

        # 6. Compute topological execution order
        execution_order = self._topological_sort(nodes, dependencies)

        # 7. Compute root arguments (external inputs)
        all_params = set()
        for node in nodes:
            all_params.update(node.root_args)

        # Exclude gate outputs from root args
        root_args = sorted(all_params - set(output_to_node.keys()) - branch_gates)

        # 8. Compute required outputs for PipelineNodes (optimization)
        required_outputs = self._compute_required_outputs(
            nodes, dependencies, output_to_node
        )

        return GraphResult(
            output_to_node=output_to_node,
            execution_order=execution_order,
            root_args=root_args,
            available_output_names=sorted(
                k for k in output_to_node.keys() if k not in branch_gates
            ),
            dependencies=dependencies,
            required_outputs=required_outputs,
            branch_gates=branch_gates,
            gate_dependencies=gate_dependencies,
            exclusive_producers=exclusive_producers,
        )

    def _are_exclusive(
        self,
        node1: "Node",
        node2: "Node",
        gate_dependencies: Dict["Node", List[str]],
    ) -> bool:
        """Check if two nodes are in mutually exclusive branches.

        Two nodes are exclusive if they depend on opposite gates from the same branch.

        Args:
            node1: First node
            node2: Second node
            gate_dependencies: Mapping from node to gate signals it depends on

        Returns:
            True if nodes are in mutually exclusive branches
        """
        gates1 = set(gate_dependencies.get(node1, []))
        gates2 = set(gate_dependencies.get(node2, []))

        if not gates1 or not gates2:
            return False

        # Check for opposite gates from the same branch
        # Gate names are like "_gate_{branch_name}_true" and "_gate_{branch_name}_false"
        for g1 in gates1:
            if g1.endswith("_true"):
                opposite = g1[:-5] + "_false"
                if opposite in gates2:
                    return True
            elif g1.endswith("_false"):
                opposite = g1[:-6] + "_true"
                if opposite in gates2:
                    return True

        return False

    def _validate_dependencies(
        self,
        nodes: List["Node"],
        dependencies: Dict["Node", List["Node"]],
        output_to_node: Dict[str, "Node"],
    ) -> None:
        """Validate graph has no cycles and all dependencies exist."""
        # Check for cycles using DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node: WHITE for node in nodes}

        def visit(node: "Node", path: List["Node"]) -> None:
            if color[node] == GRAY:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycle_names = [n.name for n in cycle]
                raise CycleError(
                    f"Cycle detected in pipeline: {' -> '.join(cycle_names)}"
                )

            if color[node] == BLACK:
                return

            color[node] = GRAY
            path.append(node)

            for dep in dependencies[node]:
                visit(dep, path)

            path.pop()
            color[node] = BLACK

        for node in nodes:
            if color[node] == WHITE:
                visit(node, [])

    def _topological_sort(
        self, nodes: List["Node"], dependencies: Dict["Node", List["Node"]]
    ) -> List["Node"]:
        """Compute topological order using Kahn's algorithm."""
        # Calculate in-degree for each node (number of dependencies)
        in_degree = {node: len(dependencies[node]) for node in nodes}

        # Start with nodes that have no dependencies
        queue = [node for node in nodes if in_degree[node] == 0]
        result = []

        while queue:
            # Sort by node name for deterministic ordering
            # Use empty string for nodes without names (None)
            queue.sort(key=lambda n: n.name if n.name is not None else "")
            node = queue.pop(0)
            result.append(node)

            # For each node that depends on this one
            for other in nodes:
                if node in dependencies[other]:
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        queue.append(other)

        if len(result) != len(nodes):
            # This shouldn't happen if _validate_dependencies passed
            raise CycleError("Topological sort failed - possible cycle")

        return result

    def _compute_required_outputs(
        self,
        nodes: List["Node"],
        dependencies: Dict["Node", List["Node"]],
        output_to_node: Dict[str, "Node"],
    ) -> Dict["Node", Optional[List[str]]]:
        """Compute which outputs are required for each PipelineNode.

        For each PipelineNode, analyzes which of its outputs are used by downstream
        nodes. This information is stored in the graph result (not mutating the node).

        Args:
            nodes: List of all nodes in the pipeline
            dependencies: Dependency graph
            output_to_node: Mapping from output names to producing nodes

        Returns:
            Mapping from PipelineNode to required output names (or None for all)
        """
        # Import here to avoid circular dependency
        from .pipeline_node import PipelineNode

        required_outputs: Dict["Node", Optional[List[str]]] = {}

        for node in nodes:
            # Only analyze PipelineNodes
            if not isinstance(node, PipelineNode):
                continue

            # Get all outputs this PipelineNode produces (in outer naming)
            node_outputs = node.output_name
            if isinstance(node_outputs, str):
                node_outputs = [node_outputs]
            else:
                node_outputs = list(node_outputs)

            # Find which of these outputs are actually used
            used_outputs = set()

            # Check all other nodes to see if they depend on this node's outputs
            for other_node in nodes:
                if other_node == node:
                    continue

                # Check if other_node uses any of this PipelineNode's outputs
                for param in other_node.root_args:
                    if param in output_to_node and output_to_node[param] == node:
                        # This param is produced by our PipelineNode
                        # Find which output name it corresponds to
                        if param in node_outputs:
                            used_outputs.add(param)

            # Optimization logic:
            # - If SOME (but not all) outputs are used: store only those
            # - If ALL outputs are used OR no outputs are used: store None (return all)
            if used_outputs and len(used_outputs) < len(node_outputs):
                # Only some outputs are needed - optimize!
                required_outputs[node] = sorted(list(used_outputs))
            else:
                # Either all outputs are needed, or no outputs are needed
                # In both cases, return all outputs (None = no pruning)
                # This handles both:
                # - Terminal nodes (no downstream dependencies)
                # - Nodes where all outputs are used
                required_outputs[node] = None

        return required_outputs

    def compute_required_nodes(
        self,
        dependencies: Dict["Node", List["Node"]],
        execution_order: List["Node"],
        output_to_node: Dict[str, "Node"],
        output_names: Union[str, List[str], None],
    ) -> Optional[List["Node"]]:
        """Compute minimal set of nodes needed for requested outputs.

        Args:
            dependencies: Mapping from each node to its dependencies
            execution_order: Full execution order
            output_to_node: Mapping from output names to nodes
            output_names: Output name(s) to compute, or None for all outputs

        Returns:
            List of nodes in execution order needed to produce outputs,
            or None if all nodes should be executed

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

        # Traverse dependencies to find all required nodes
        required_nodes = set()

        def add_dependencies(node: "Node"):
            """Recursively add node and all its dependencies."""
            if node in required_nodes:
                return
            required_nodes.add(node)
            for dep in dependencies.get(node, []):
                add_dependencies(dep)

        for target_node in target_nodes:
            add_dependencies(target_node)

        # Return in execution order
        return [n for n in execution_order if n in required_nodes]
