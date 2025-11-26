from functools import singledispatchmethod
from typing import Any, Dict, List, Optional, Set, get_type_hints

from ..node import Node
from ..pipeline import Pipeline
from ..pipeline_node import PipelineNode as HyperPipelineNode
from .structures import (
    DataNode,
    DualNode,
    FunctionNode,
    GroupDataNode,
    PipelineNode,
    VisualizationGraph,
    VizEdge,
    VizNode,
)


class GraphWalker:
    """Traverses the pipeline graph to generate a flat visualization structure."""

    def __init__(self, pipeline: Pipeline, expanded_nodes: Set[str], group_inputs: bool = True, traverse_collapsed: bool = False):
        self.pipeline = pipeline
        self.expanded_nodes = expanded_nodes
        self.group_inputs = group_inputs
        self.traverse_collapsed = traverse_collapsed

    def get_visualization_data(self) -> VisualizationGraph:
        """Generate the flat visualization graph based on current state."""
        nodes: List[VizNode] = []
        edges: List[VizEdge] = []
        
        # Scope tracking: Map output_name -> DataNode ID
        scope: Dict[str, str] = {}
        
        self._process_pipeline(
            pipeline=self.pipeline,
            parent_id=None,
            prefix="",
            nodes_out=nodes,
            edges_out=edges,
            scope=scope
        )
        
        if self.group_inputs:
            nodes, edges = self._group_data_nodes(nodes, edges)
            
        return VisualizationGraph(nodes=nodes, edges=edges)

    def _process_pipeline(
        self,
        pipeline: Pipeline,
        parent_id: Optional[str],
        prefix: str,
        nodes_out: List[VizNode],
        edges_out: List[VizEdge],
        scope: Dict[str, str],
        parent_pipeline: Optional[Pipeline] = None,
    ):
        """Recursively process pipeline to generate nodes and edges."""
        graph = pipeline.graph
        
        # 1. Register bound inputs as DataNodes in the current scope
        for param_name, value in pipeline.bound_inputs.items():
            # Extract type hint from any node in the pipeline that uses this parameter
            type_hint = None
            for node in graph.execution_order:
                if hasattr(node, "func"):
                    try:
                        hints = get_type_hints(node.func)
                        if param_name in hints:
                            type_str = str(hints[param_name]).replace("typing.", "").replace("<class '", "").replace("'>", "")
                            type_hint = self._simplify_type_string(type_str)
                            break
                    except Exception:
                        pass
            
            # Use human-readable param name (with prefix for nested scopes)
            data_node_id = f"{prefix}{param_name}" if prefix else param_name
            data_node = DataNode(
                id=data_node_id,
                parent_id=parent_id,
                name=param_name,
                is_bound=True,
                source_id=None, # Bound value, no source node
                type_hint=type_hint
            )
            nodes_out.append(data_node)
            scope[param_name] = data_node_id

        # 2. Process nodes
        for node in graph.execution_order:
            self._visit_node(node, parent_id, prefix, nodes_out, edges_out, scope, parent_pipeline=pipeline)

    @singledispatchmethod
    def _visit_node(
        self,
        node: Any,
        parent_id: Optional[str],
        prefix: str,
        nodes_out: List[VizNode],
        edges_out: List[VizEdge],
        scope: Dict[str, str],
        parent_pipeline: Optional[Pipeline] = None,
    ):
        """Default visitor for unknown node types."""
        pass

    @_visit_node.register
    def _(
        self,
        node: HyperPipelineNode,
        parent_id: Optional[str],
        prefix: str,
        nodes_out: List[VizNode],
        edges_out: List[VizEdge],
        scope: Dict[str, str],
        parent_pipeline: Optional[Pipeline] = None,
    ):
        # Generate a human-readable label
        label = self._get_pipeline_node_label(node)
        
        # Use human-readable label as the node ID
        node_id = f"{prefix}{label}"
        is_expanded = node_id in self.expanded_nodes
        
        viz_node = PipelineNode(
            id=node_id,
            parent_id=parent_id,
            label=label,
            is_expanded=is_expanded
        )
        nodes_out.append(viz_node)
        
        if is_expanded or self.traverse_collapsed:
            # For expanded PipelineNode (or if traversing collapsed nodes for interactive viz):
            # 1. Expand the internal structure (creates boundary output DataNodes)
            self._expand_pipeline_node(node, node_id, prefix, nodes_out, edges_out, scope)
            
            # Note: When expanded, input connections go directly to inner nodes (via _expand_pipeline_node logic),
            # or we rely on the fact that outer DataNodes are in scope.
            # We do NOT create edges to the container PipelineNode itself to avoid double edges.
            
        else:
            # For collapsed PipelineNode: handle both inputs and outputs normally
            self._handle_connections(node, node_id, prefix, nodes_out, edges_out, scope, parent_pipeline)

    @_visit_node.register
    def _(
        self,
        node: Node,
        parent_id: Optional[str],
        prefix: str,
        nodes_out: List[VizNode],
        edges_out: List[VizEdge],
        scope: Dict[str, str],
        parent_pipeline: Optional[Pipeline] = None,
    ):
        func_name = self._get_func_name(node)
        # Use human-readable function name as the node ID
        node_id = f"{prefix}{func_name}"
        label = self._get_node_label(node)
        
        # Check for dual node property safely
        is_dual = getattr(node, "is_dual_node", False)
        
        if is_dual:
            viz_node = DualNode(
                id=node_id,
                parent_id=parent_id,
                label=label,
                function_name=func_name
            )
        else:
            viz_node = FunctionNode(
                id=node_id,
                parent_id=parent_id,
                label=label,
                function_name=func_name
            )
        nodes_out.append(viz_node)
        
        self._handle_connections(node, node_id, prefix, nodes_out, edges_out, scope, parent_pipeline)

    def _expand_pipeline_node(
        self,
        node: HyperPipelineNode,
        node_id: str,
        prefix: str,
        nodes_out: List[VizNode],
        edges_out: List[VizEdge],
        scope: Dict[str, str],
    ):
        """Handle expansion of a pipeline node."""
        # Create a new scope for the nested pipeline
        nested_scope = {}
        
        # Create reverse mapping for inputs: Inner -> Outer
        reverse_input_mapping = {v: k for k, v in node.input_mapping.items()} if node.input_mapping else {}

        # Map inputs for UNFULFILLED args only (bound inputs will be handled by _process_pipeline)
        # Use node.unfulfilled_args (outer names) instead of pipeline.graph.root_args
        for outer_arg in node.unfulfilled_args:
            # Map to inner name
            inner_arg = node.input_mapping.get(outer_arg, outer_arg)
            
            if outer_arg in scope:
                nested_scope[inner_arg] = scope[outer_arg]
            else:
                # Not found in outer scope - create input DataNode
                # Use just the argument name (with prefix for nested scopes)
                input_node_id = f"{prefix}{outer_arg}" if prefix else outer_arg
                if input_node_id not in [n.id for n in nodes_out]:
                    input_node = DataNode(
                        id=input_node_id,
                        parent_id=nodes_out[-1].parent_id if nodes_out else None,
                        name=outer_arg,
                        is_bound=False
                    )
                    nodes_out.append(input_node)
                    scope[outer_arg] = input_node_id
                
                nested_scope[inner_arg] = input_node_id

        # Recurse - this will handle bound inputs via _process_pipeline
        self._process_pipeline(
            pipeline=node.pipeline,
            parent_id=node_id,
            prefix=f"{node_id}__",
            nodes_out=nodes_out,
            edges_out=edges_out,
            scope=nested_scope
        )
        
        # Map outputs: Create boundary DataNodes for mapped outputs if names differ
        # Get the PipelineNode's parent_id
        pipeline_node = next(n for n in nodes_out if n.id == node_id)
        outer_parent_id = pipeline_node.parent_id
        
        if node.output_mapping:
            for inner_out, outer_out in node.output_mapping.items():
                if inner_out in nested_scope:
                    inner_data_node_id = nested_scope[inner_out]
                    if inner_out != outer_out:
                        # Create boundary DataNode for the mapped output name
                        outer_data_node_id = f"{prefix}{outer_out}" if prefix else outer_out
                        outer_data_node = DataNode(
                            id=outer_data_node_id,
                            parent_id=outer_parent_id,
                            name=outer_out,
                            is_bound=False,
                            source_id=inner_data_node_id  # Links to inner output
                        )
                        nodes_out.append(outer_data_node)
                        # Create edge from inner output to outer boundary
                        edges_out.append(VizEdge(source=inner_data_node_id, target=outer_data_node_id))
                        scope[outer_out] = outer_data_node_id
                    else:
                        # Same name - just register in outer scope
                        scope[outer_out] = inner_data_node_id
        else:
            # Default mapping (same names) - register inner DataNodes directly in outer scope
            # No need to create boundary DataNodes since the pipeline is expanded
            for out_name, _ in node.pipeline.graph.output_to_node.items():
                if out_name in nested_scope:
                    # Simply register the inner DataNode in the outer scope
                    scope[out_name] = nested_scope[out_name]

    def _handle_input_connections(
        self,
        node: Any,
        node_id: str,
        prefix: str,
        nodes_out: List[VizNode],
        edges_out: List[VizEdge],
        scope: Dict[str, str],
    ):
        """Handle input connections to a node."""
        if hasattr(node, "root_args"):
            for arg in node.root_args:
                # arg is already the correct name (outer name for PipelineNode)
                actual_arg = arg

                if actual_arg in scope:
                    data_node_id = scope[actual_arg]
                    edges_out.append(VizEdge(source=data_node_id, target=node_id))
                else:
                    # Create new Input DataNode
                    # Use just the argument name (with prefix for nested scopes)
                    input_node_id = f"{prefix}{actual_arg}" if prefix else actual_arg
                    exists = False
                    for n in nodes_out:
                        if n.id == input_node_id:
                            exists = True
                            break
                    
                    if not exists:
                        # Extract type hint for input
                        type_hint = None
                        if hasattr(node, "func"):
                            try:
                                hints = get_type_hints(node.func)
                                if actual_arg in hints:
                                    type_str = str(hints[actual_arg]).replace("typing.", "").replace("<class '", "").replace("'>", "")
                                    type_hint = self._simplify_type_string(type_str)
                            except Exception:
                                pass

                        input_node = DataNode(
                            id=input_node_id,
                            parent_id=nodes_out[-1].parent_id if nodes_out else None,
                            name=actual_arg,
                            is_bound=False,
                            type_hint=type_hint
                        )
                        nodes_out.append(input_node)
                        scope[actual_arg] = input_node_id
                    
                    edges_out.append(VizEdge(source=input_node_id, target=node_id))

    def _handle_output_connections(
        self,
        node: Any,
        node_id: str,
        prefix: str,
        nodes_out: List[VizNode],
        edges_out: List[VizEdge],
        scope: Dict[str, str],
        parent_pipeline: Optional[Pipeline] = None,
    ):
        """Handle output connections from a node."""
        if hasattr(node, "output_name"):
            output_names = node.output_name
            if not isinstance(output_names, tuple):
                output_names = (output_names,)
            
            # For collapsed PipelineNodes, filter to only required outputs
            if isinstance(node, HyperPipelineNode) and parent_pipeline is not None:
                # Get required outputs for this node from the parent pipeline
                required_outputs = parent_pipeline.graph.required_outputs.get(node)
                if required_outputs is not None:
                    # Filter output_names to only include required ones
                    if isinstance(required_outputs, str):
                        required_outputs = [required_outputs]
                    output_names = tuple(out for out in output_names if out in required_outputs)
            
            # output_names are already correct (outer names for PipelineNode)
            for out_name in output_names:
                final_out_name = out_name
                
                # Use human-readable output name (with prefix from parent scope)
                data_node_id = f"{prefix}{final_out_name}" if prefix else final_out_name
                type_hint = self._extract_return_type(node, final_out_name)
                
                data_node = DataNode(
                    id=data_node_id,
                    parent_id=nodes_out[-1].parent_id if nodes_out else None,
                    name=final_out_name,
                    type_hint=type_hint,
                    source_id=node_id
                )
                nodes_out.append(data_node)
                
                # Register in scope
                scope[final_out_name] = data_node_id
                
                # Create Edge: FunctionNode -> DataNode
                edges_out.append(VizEdge(source=node_id, target=data_node_id))

    def _handle_connections(
        self, 
        node: Any, 
        node_id: str, 
        prefix: str,
        nodes_out: List[VizNode], 
        edges_out: List[VizEdge], 
        scope: Dict[str, str],
        parent_pipeline: Optional[Pipeline] = None,
    ):
        """Handle input/output connections for any node type."""
        self._handle_input_connections(node, node_id, prefix, nodes_out, edges_out, scope)
        self._handle_output_connections(node, node_id, prefix, nodes_out, edges_out, scope, parent_pipeline)

    def _group_data_nodes(self, nodes: List[VizNode], edges: List[VizEdge]) -> tuple[List[VizNode], List[VizEdge]]:
        """Group DataNodes that share source, target set, and bound state."""
        data_nodes: Dict[str, DataNode] = {
            n.id: n for n in nodes if isinstance(n, DataNode)
        }

        # Build outgoing target sets for each data node
        targets_map: Dict[str, set[str]] = {n_id: set() for n_id in data_nodes}
        for edge in edges:
            if edge.source in data_nodes:
                targets_map[edge.source].add(edge.target)

        # Bucket by (source_id, targets, is_bound)
        groups: Dict[tuple[Optional[str], tuple[str, ...], bool], List[DataNode]] = {}
        for dn_id, dn in data_nodes.items():
            key = (
                dn.source_id,
                tuple(sorted(targets_map.get(dn_id, []))),
                dn.is_bound,
            )
            groups.setdefault(key, []).append(dn)

        new_nodes = list(nodes)
        new_edges = list(edges)

        for (source_id, targets, is_bound), group in groups.items():
            if len(group) < 2:
                continue

            group_id = f"group_{source_id or 'input'}_{abs(hash((source_id, targets, is_bound))) % 100000}"
            group_node = GroupDataNode(
                id=group_id,
                parent_id=group[0].parent_id,
                nodes=group,
                is_bound=is_bound,
                source_id=source_id,
            )

            grouped_ids = {g.id for g in group}
            new_nodes = [n for n in new_nodes if getattr(n, "id", None) not in grouped_ids]
            new_nodes.append(group_node)

            # Remove edges touching grouped data nodes
            new_edges = [
                e
                for e in new_edges
                if e.source not in grouped_ids and e.target not in grouped_ids
            ]

            # Source -> group
            if source_id:
                new_edges.append(VizEdge(source=source_id, target=group_id))

            # Group -> targets
            for tgt in targets:
                new_edges.append(VizEdge(source=group_id, target=tgt))

        return new_nodes, new_edges

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

    def _get_node_label(self, node: Node) -> str:
        if hasattr(node, "func") and hasattr(node.func, "__name__"):
            return node.func.__name__
        return str(node)

    def _get_func_name(self, node: Node) -> str:
        if hasattr(node, "func"):
            if hasattr(node.func, "__name__"):
                return node.func.__name__
            return str(node.func)
        return "unknown"

    def _extract_return_type(self, node: Node, output_name: Optional[str] = None) -> Optional[str]:
        """Extract return type hint from node function or inner pipeline."""
        if isinstance(node, HyperPipelineNode):
            if output_name is None:
                return None
                
            # Map outer output name to inner output name
            inner_out_name = output_name
            if node.output_mapping:
                # output_mapping is {inner: outer}
                for inner, outer in node.output_mapping.items():
                    if outer == output_name:
                        inner_out_name = inner
                        break
            
            # Find inner node producing this output
            if hasattr(node.pipeline, "graph") and hasattr(node.pipeline.graph, "output_to_node"):
                if inner_out_name in node.pipeline.graph.output_to_node:
                    inner_node = node.pipeline.graph.output_to_node[inner_out_name]
                    return self._extract_return_type(inner_node, inner_out_name)
            return None

        if not hasattr(node, "func"):
            return None
        try:
            hints = get_type_hints(node.func)
            if "return" in hints:
                type_str = str(hints["return"]).replace("typing.", "").replace("<class '", "").replace("'>", "")
                return self._simplify_type_string(type_str)
        except Exception:
            pass
        return None
    
    def _simplify_type_string(self, type_str: str) -> str:
        """Simplify type strings by extracting only the final part of dotted names.
        
        Examples:
            "List[rag_hypernodes.models.Document]" -> "List[Document]"
            "Dict[str, foo.bar.Baz]" -> "Dict[str, Baz]"
        """
        import re
        # Pattern to match dotted names (e.g., a.b.c.ClassName)
        # This matches sequences like word.word.word where the words contain letters, numbers, and underscores
        pattern = r'([a-zA-Z_][a-zA-Z0-9_]*\.)+([a-zA-Z_][a-zA-Z0-9_]*)'
        
        def replace_with_final(match):
            # Get the full match and extract the final part after the last dot
            full_match = match.group(0)
            parts = full_match.split('.')
            return parts[-1]
        
        return re.sub(pattern, replace_with_final, type_str)
