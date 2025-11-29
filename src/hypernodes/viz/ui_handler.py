"""UI handler and backend for visualization frontends.

The UI handler owns visualization state and acts as the single backend for
both static (Graphviz) and interactive (React/ipywidgets) frontends. It is
responsible for:
- Accepting generic graph arguments (depth, grouping, etc.)
- Managing expansion/collapse state
- Delegating graph generation to GraphWalker
- Providing debugging/validation tools for graph structure
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Set

from ..pipeline import Pipeline
from ..pipeline_node import PipelineNode as HyperPipelineNode
from .graph_walker import GraphWalker
from .structures import (
    DataNode,
    DualNode,
    FunctionNode,
    GroupDataNode,
    PipelineNode,
    VisualizationGraph,
    VizNode,
)

_UNSET = object()


class UIHandler:
    """Backend controller for visualization state and events."""

    def __init__(
        self,
        pipeline: Pipeline,
        depth: Optional[int] = 1,
        group_inputs: bool = False,
        show_output_types: bool = False,
    ):
        self.pipeline = pipeline
        self.depth = depth
        self.group_inputs = group_inputs
        self.show_output_types = show_output_types

        # State
        self.expanded_nodes: Set[str] = set()
        
        # Initialize expansion based on depth
        self._initialize_expansion(pipeline, depth)

    def _initialize_expansion(self, pipeline: Pipeline, depth: Optional[int]):
        """Recursively set initial expansion state."""
        if depth is None:
            # Expand everything
            self._expand_all(pipeline)
        elif depth > 1:
            # Expand to specific depth
            self._expand_to_depth(pipeline, current_depth=1, max_depth=depth)

    def _expand_all(self, pipeline: Pipeline, prefix: str = ""):
        for node in pipeline.graph.execution_order:
            if isinstance(node, HyperPipelineNode):
                # Use human-readable label as the node ID (consistent with GraphWalker)
                label = self._get_pipeline_node_label(node)
                node_id = f"{prefix}{label}"
                self.expanded_nodes.add(node_id)
                self._expand_all(node.pipeline, prefix=f"{node_id}__")

    def _expand_to_depth(self, pipeline: Pipeline, current_depth: int, max_depth: int, prefix: str = ""):
        if current_depth >= max_depth:
            return
        
        for node in pipeline.graph.execution_order:
            if isinstance(node, HyperPipelineNode):
                # Use human-readable label as the node ID (consistent with GraphWalker)
                label = self._get_pipeline_node_label(node)
                node_id = f"{prefix}{label}"
                self.expanded_nodes.add(node_id)
                self._expand_to_depth(node.pipeline, current_depth + 1, max_depth, prefix=f"{node_id}__")

    def _get_pipeline_node_label(self, node: HyperPipelineNode) -> str:
        """Generate a human-readable label for a PipelineNode.
        
        Priority: node.name > node.pipeline.name > function name of first node > "Pipeline"
        """
        if node.name:
            return node.name
        elif hasattr(node.pipeline, "name") and node.pipeline.name:
            return node.pipeline.name
        else:
            # Try to get a descriptive name from the first function in the pipeline
            if node.pipeline.nodes:
                first_node = node.pipeline.nodes[0]
                if hasattr(first_node, "func") and hasattr(first_node.func, "__name__"):
                    return f"{first_node.func.__name__}_pipeline"
                else:
                    return "Pipeline"
            else:
                return "Pipeline"

    def apply_options(
        self,
        depth: Any = _UNSET,
        group_inputs: Optional[bool] = None,
        show_output_types: Optional[bool] = None,
    ) -> None:
        """Update generic options."""
        if depth is not _UNSET:
            self.depth = depth
            self.expanded_nodes.clear()
            self._initialize_expansion(self.pipeline, depth)

        if group_inputs is not None:
            self.group_inputs = group_inputs

        if show_output_types is not None:
            self.show_output_types = show_output_types

    def expand_node(self, node_id: str) -> None:
        """Mark a node as expanded."""
        self.expanded_nodes.add(node_id)

    def collapse_node(self, node_id: str) -> None:
        """Collapse a pipeline node and all its descendants."""
        if node_id in self.expanded_nodes:
            self.expanded_nodes.remove(node_id)
            # Also remove any descendants
            prefix = f"{node_id}__"
            to_remove = [nid for nid in self.expanded_nodes if nid.startswith(prefix)]
            for nid in to_remove:
                self.expanded_nodes.remove(nid)

    def toggle_node(self, node_id: str) -> None:
        """Toggle the expansion state of a node."""
        if node_id in self.expanded_nodes:
            self.collapse_node(node_id)
        else:
            self.expand_node(node_id)

    def get_visualization_data(self, traverse_collapsed: bool = True) -> VisualizationGraph:
        """Generate the flat visualization graph based on current state."""
        walker = GraphWalker(
            pipeline=self.pipeline,
            expanded_nodes=self.expanded_nodes,
            group_inputs=self.group_inputs,
            traverse_collapsed=traverse_collapsed
        )
        return walker.get_visualization_data()

    # =========================================================================
    # Debugging & Validation API
    # =========================================================================

    def validate_graph(self, traverse_collapsed: bool = True) -> List[str]:
        """Validate the visualization graph and return a list of errors.
        
        Checks for:
        - Orphan edges (edges referencing non-existent nodes)
        - Duplicate node IDs
        - Missing parent nodes
        - Self-loops
        
        Returns:
            List of error strings. Empty list means graph is valid.
        """
        graph = self.get_visualization_data(traverse_collapsed=traverse_collapsed)
        errors: List[str] = []
        
        node_ids = {n.id for n in graph.nodes}
        
        # Check for duplicate node IDs
        seen_ids: Set[str] = set()
        for node in graph.nodes:
            if node.id in seen_ids:
                errors.append(f"Duplicate node ID: '{node.id}'")
            seen_ids.add(node.id)
        
        # Check for orphan edges
        for edge in graph.edges:
            if edge.source not in node_ids:
                errors.append(f"Edge source '{edge.source}' not found (target: '{edge.target}')")
            if edge.target not in node_ids:
                errors.append(f"Edge target '{edge.target}' not found (source: '{edge.source}')")
            if edge.source == edge.target:
                errors.append(f"Self-loop detected: '{edge.source}'")
        
        # Check for missing parent nodes
        for node in graph.nodes:
            if node.parent_id and node.parent_id not in node_ids:
                errors.append(f"Node '{node.id}' has missing parent '{node.parent_id}'")
        
        return errors

    def debug_dump(self, traverse_collapsed: bool = True) -> Dict[str, Any]:
        """Return a complete state snapshot for debugging.
        
        Returns a dictionary with:
        - nodes: List of all nodes with their properties
        - edges: List of all edges
        - metadata: Computed metadata (boundary outputs, producer map, type hints)
        - validation: Results of validate_graph()
        - state: Current expansion state and options
        """
        graph = self.get_visualization_data(traverse_collapsed=traverse_collapsed)
        
        # Build node lookup
        node_by_id: Dict[str, VizNode] = {n.id: n for n in graph.nodes}
        
        # Compute metadata
        producer_map: Dict[str, str] = {}  # output_id -> producer_id
        boundary_outputs: List[str] = []
        input_type_hints: Dict[str, Optional[str]] = {}
        output_type_hints: Dict[str, Optional[str]] = {}
        
        for node in graph.nodes:
            if isinstance(node, (DataNode, GroupDataNode)):
                if node.source_id:
                    producer_map[node.id] = node.source_id
                    # Check if it's a boundary output (source is a PipelineNode)
                    source_node = node_by_id.get(node.source_id)
                    if isinstance(source_node, PipelineNode):
                        boundary_outputs.append(node.id)
                    # Record output type hint
                    if isinstance(node, DataNode) and node.type_hint:
                        output_type_hints[node.id] = node.type_hint
                else:
                    # Input node
                    if isinstance(node, DataNode) and node.type_hint:
                        input_type_hints[node.id] = node.type_hint
                    elif isinstance(node, GroupDataNode):
                        for inner in node.nodes:
                            if inner.type_hint:
                                input_type_hints[inner.id] = inner.type_hint
        
        # Serialize nodes
        def serialize_node(n: VizNode) -> Dict[str, Any]:
            result = {
                "id": n.id,
                "parent": n.parent_id,
            }
            if isinstance(n, FunctionNode):
                result["type"] = "DUAL" if isinstance(n, DualNode) else "FUNCTION"
                result["label"] = n.label
                result["function_name"] = n.function_name
            elif isinstance(n, PipelineNode):
                result["type"] = "PIPELINE"
                result["label"] = n.label
                result["is_expanded"] = n.is_expanded
            elif isinstance(n, DataNode):
                result["type"] = "DATA"
                result["name"] = n.name
                result["type_hint"] = n.type_hint
                result["is_bound"] = n.is_bound
                result["source_id"] = n.source_id
            elif isinstance(n, GroupDataNode):
                result["type"] = "GROUP_DATA"
                result["is_bound"] = n.is_bound
                result["source_id"] = n.source_id
                result["nodes"] = [serialize_node(inner) for inner in n.nodes]
            return result
        
        return {
            "nodes": [serialize_node(n) for n in graph.nodes],
            "edges": [{"source": e.source, "target": e.target, "label": e.label} for e in graph.edges],
            "metadata": {
                "boundary_outputs": boundary_outputs,
                "producer_map": producer_map,
                "input_type_hints": input_type_hints,
                "output_type_hints": output_type_hints,
            },
            "validation": {
                "errors": self.validate_graph(traverse_collapsed=traverse_collapsed),
            },
            "state": {
                "expanded_nodes": list(self.expanded_nodes),
                "depth": self.depth,
                "group_inputs": self.group_inputs,
            },
            "stats": {
                "total_nodes": len(graph.nodes),
                "total_edges": len(graph.edges),
                "node_types": dict(Counter(type(n).__name__ for n in graph.nodes)),
            },
        }

    def trace_node(self, node_id: str, traverse_collapsed: bool = True) -> Dict[str, Any]:
        """Trace detailed information about a specific node.
        
        Useful for debugging why a node looks wrong (missing type hint, wrong parent, etc.)
        
        Returns:
            Dictionary with node details, its connections, and diagnostic suggestions.
        """
        graph = self.get_visualization_data(traverse_collapsed=traverse_collapsed)
        node_by_id = {n.id: n for n in graph.nodes}
        
        node = node_by_id.get(node_id)
        if not node:
            # Try partial match
            matches = [n for n in graph.nodes if node_id in n.id]
            return {
                "status": "NOT_FOUND",
                "node_id": node_id,
                "partial_matches": [n.id for n in matches[:5]],
                "suggestion": f"Node '{node_id}' not found. Did you mean one of: {[n.id for n in matches[:3]]}?" if matches else "No similar nodes found.",
            }
        
        # Find incoming and outgoing edges
        incoming = [e for e in graph.edges if e.target == node_id]
        outgoing = [e for e in graph.edges if e.source == node_id]
        
        # Build trace info
        trace: Dict[str, Any] = {
            "status": "FOUND",
            "node_id": node_id,
            "node_type": type(node).__name__,
            "parent": node.parent_id,
            "incoming_edges": [{"from": e.source, "label": e.label} for e in incoming],
            "outgoing_edges": [{"to": e.target, "label": e.label} for e in outgoing],
        }
        
        # Type-specific info
        if isinstance(node, DataNode):
            trace["data_info"] = {
                "name": node.name,
                "type_hint": node.type_hint,
                "is_bound": node.is_bound,
                "source_id": node.source_id,
                "is_input": node.source_id is None,
                "is_output": node.source_id is not None,
            }
            # Diagnostic for missing type hint
            if not node.type_hint and not node.source_id:
                trace["diagnostics"] = {
                    "issue": "Missing type hint for input",
                    "suggestion": "Check if the consuming function has type annotations for this parameter.",
                }
        elif isinstance(node, (FunctionNode, DualNode)):
            trace["function_info"] = {
                "label": node.label,
                "function_name": node.function_name,
            }
            # Find outputs
            outputs = [n for n in graph.nodes if isinstance(n, DataNode) and n.source_id == node_id]
            trace["outputs"] = [{"id": o.id, "name": o.name, "type_hint": o.type_hint} for o in outputs]
        elif isinstance(node, PipelineNode):
            trace["pipeline_info"] = {
                "label": node.label,
                "is_expanded": node.is_expanded,
            }
            # Find children
            children = [n for n in graph.nodes if n.parent_id == node_id]
            trace["children"] = [n.id for n in children]
            # Find boundary outputs
            boundary = [n for n in graph.nodes if isinstance(n, DataNode) and n.source_id == node_id]
            trace["boundary_outputs"] = [{"id": b.id, "name": b.name} for b in boundary]
        
        return trace

    def trace_edge(self, source_id: str, target_id: str, traverse_collapsed: bool = True) -> Dict[str, Any]:
        """Trace information about a specific edge (or missing edge).
        
        Useful for debugging why an edge is missing or pointing to the wrong place.
        
        Returns:
            Dictionary with edge details, both nodes' info, and diagnostic analysis.
        """
        graph = self.get_visualization_data(traverse_collapsed=traverse_collapsed)
        node_by_id = {n.id: n for n in graph.nodes}
        
        # Find the edge
        edge = next((e for e in graph.edges if e.source == source_id and e.target == target_id), None)
        
        source_node = node_by_id.get(source_id)
        target_node = node_by_id.get(target_id)
        
        result: Dict[str, Any] = {
            "edge_query": f"{source_id} → {target_id}",
            "edge_found": edge is not None,
        }
        
        if edge:
            result["edge"] = {"source": edge.source, "target": edge.target, "label": edge.label}
        
        # Analyze source
        if source_node:
            result["source"] = {
                "found": True,
                "type": type(source_node).__name__,
                "parent": source_node.parent_id,
            }
            if isinstance(source_node, DataNode):
                result["source"]["source_id"] = source_node.source_id
        else:
            result["source"] = {"found": False}
            # Try to find similar
            matches = [n.id for n in graph.nodes if source_id in n.id][:3]
            if matches:
                result["source"]["similar_ids"] = matches
        
        # Analyze target
        if target_node:
            result["target"] = {
                "found": True,
                "type": type(target_node).__name__,
                "parent": target_node.parent_id,
            }
        else:
            result["target"] = {"found": False}
            matches = [n.id for n in graph.nodes if target_id in n.id][:3]
            if matches:
                result["target"]["similar_ids"] = matches
        
        # Find related edges if edge not found
        if not edge:
            # Edges FROM source
            from_source = [e for e in graph.edges if e.source == source_id]
            # Edges TO target
            to_target = [e for e in graph.edges if e.target == target_id]
            
            result["analysis"] = {
                "edges_from_source": [{"to": e.target} for e in from_source],
                "edges_to_target": [{"from": e.source} for e in to_target],
            }
            
            # Try to find the path
            if source_node and target_node:
                # Check if there's an intermediate node
                for e1 in from_source:
                    intermediate = node_by_id.get(e1.target)
                    if intermediate:
                        for e2 in graph.edges:
                            if e2.source == e1.target and e2.target == target_id:
                                result["analysis"]["indirect_path"] = [
                                    {"from": source_id, "to": e1.target},
                                    {"from": e1.target, "to": target_id},
                                ]
                                break
                
                # Check if it's a boundary output situation
                if isinstance(source_node, (FunctionNode, DualNode)):
                    outputs = [n for n in graph.nodes if isinstance(n, DataNode) and n.source_id == source_id]
                    if outputs:
                        result["analysis"]["source_outputs"] = [o.id for o in outputs]
                        result["analysis"]["suggestion"] = (
                            f"Edge might go through output node. Check: {outputs[0].id} → {target_id}"
                        )
        
        return result

    def find_issues(self, traverse_collapsed: bool = True) -> Dict[str, Any]:
        """Run comprehensive diagnostics and return all found issues.
        
        This is the "one-stop" debugging method for AI agents.
        """
        graph = self.get_visualization_data(traverse_collapsed=traverse_collapsed)
        node_by_id = {n.id: n for n in graph.nodes}
        
        issues: Dict[str, List[str]] = {
            "validation_errors": self.validate_graph(traverse_collapsed),
            "missing_type_hints": [],
            "disconnected_nodes": [],
            "suspicious_edges": [],
        }
        
        # Find all node IDs that are edge endpoints
        connected_nodes = set()
        for e in graph.edges:
            connected_nodes.add(e.source)
            connected_nodes.add(e.target)
        
        for node in graph.nodes:
            # Check for missing type hints on inputs
            if isinstance(node, DataNode) and not node.source_id and not node.type_hint:
                issues["missing_type_hints"].append(node.id)
            
            # Check for disconnected nodes (not in any edge, not a parent)
            children = [n for n in graph.nodes if n.parent_id == node.id]
            if node.id not in connected_nodes and not children:
                issues["disconnected_nodes"].append(node.id)
        
        # Check for suspicious edge patterns
        edge_pairs = set()
        for e in graph.edges:
            pair = (e.source, e.target)
            if pair in edge_pairs:
                issues["suspicious_edges"].append(f"Duplicate edge: {e.source} → {e.target}")
            edge_pairs.add(pair)
            
            # Check for edges crossing boundaries unexpectedly
            source_node = node_by_id.get(e.source)
            target_node = node_by_id.get(e.target)
            if source_node and target_node:
                if source_node.parent_id != target_node.parent_id:
                    # Cross-boundary edge - not necessarily an issue but worth noting
                    pass
        
        # Filter out empty lists
        return {k: v for k, v in issues.items() if v}
