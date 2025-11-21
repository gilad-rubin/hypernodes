"""Frontend-agnostic graph serialization with complete semantic information.

This module provides comprehensive graph serialization that computes ALL relationships
and hierarchy information, eliminating the need for frontend calculations.
"""

import inspect
from typing import Any, Dict, List, Optional, Set, Union, get_type_hints

from ..pipeline import Pipeline
from ..pipeline_node import PipelineNode
from ..node import Node


class GraphSerializer:
    """Serializes a Pipeline into a frontend-agnostic semantic structure.
    
    The serializer computes all relationships, per-level hierarchy analysis, and
    semantic metadata. Frontends only need to render based on the provided data.
    
    Key features:
    - Per-level hierarchy analysis (unfulfilled inputs, bound inputs, mappings)
    - Complete type hint extraction
    - Node type detection (STANDARD, DUAL, PIPELINE)
    - Edge type classification (data_flow, parameter_flow)
    - No styling information - frontends decide based on semantic flags
    """

    def __init__(self, pipeline: Pipeline):
        """Initialize serializer with a pipeline.
        
        Args:
            pipeline: Pipeline to serialize
        """
        self.pipeline = pipeline
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self.levels: List[Dict[str, Any]] = []
        self._visited_nodes: Set[str] = set()
        self._node_id_to_level: Dict[str, str] = {}
        self._output_to_node_id: Dict[str, str] = {}  # Maps output names to node IDs (global)
        self._node_id_to_node: Dict[str, Any] = {}  # Maps node ID to actual node object
        self._pipeline_node_mappings: Dict[str, Dict[str, Any]] = {}  # Track mappings per PipelineNode

    def serialize(self, depth: Optional[int] = 1) -> Dict[str, Any]:
        """Serialize the pipeline into a frontend-agnostic structure.
        
        Args:
            depth: Expansion depth for nested pipelines
                   1 = collapsed nested pipelines
                   None = fully expand all nesting
        
        Returns:
            Dictionary with complete semantic graph data:
            {
                "levels": [...],  # Per-level hierarchy analysis
                "nodes": [...],   # Node semantic data
                "edges": [...]    # Edge semantic data
            }
        """
        self.nodes = []
        self.edges = []
        self.levels = []
        self._visited_nodes = set()
        self._node_id_to_level = {}
        self._output_to_node_id = {}
        self._node_id_to_node = {}
        self._pipeline_node_mappings = {}
        
        # Process the root pipeline
        self._process_pipeline(
            pipeline=self.pipeline,
            level_id="root",
            parent_level_id=None,
            current_depth=1,
            max_depth=depth,
            prefix=""
        )
        
        return {
            "levels": self.levels,
            "nodes": self.nodes,
            "edges": self.edges,
            "input_levels": self._compute_input_levels()
        }

    def _process_pipeline(
        self,
        pipeline: Pipeline,
        level_id: str,
        parent_level_id: Optional[str],
        current_depth: int,
        max_depth: Optional[int],
        prefix: str,
        parent_pipeline_node: Optional[PipelineNode] = None,
        parent_pipeline_node_id: Optional[str] = None
    ):
        """Recursively process a pipeline and compute per-level information.
        
        Args:
            pipeline: Pipeline to process
            level_id: Unique identifier for this level
            parent_level_id: Parent level ID (None for root)
            current_depth: Current nesting depth
            max_depth: Maximum depth to expand (None = unlimited)
            prefix: ID prefix for uniqueness
            parent_pipeline_node: The PipelineNode that wraps this pipeline (if nested)
        """
        if not hasattr(pipeline, "graph"):
            return

        graph_result = pipeline.graph
        
        # Compute per-level hierarchy information
        level_info = self._compute_level_info(
            pipeline=pipeline,
            level_id=level_id,
            parent_level_id=parent_level_id,
            parent_pipeline_node_id=parent_pipeline_node_id
        )
        self.levels.append(level_info)
        
        # Process each node in the execution order
        for node in graph_result.execution_order:
            node_id = f"{prefix}{id(node)}"
            
            if node_id in self._visited_nodes:
                continue
            self._visited_nodes.add(node_id)
            self._node_id_to_level[node_id] = level_id
            self._node_id_to_node[node_id] = node
            
            # Check if this is a nested pipeline
            is_nested = isinstance(node, PipelineNode)
            should_expand = max_depth is None or current_depth < max_depth
            
            if is_nested and should_expand:
                # Expanded nested pipeline
                nested_level_id = f"{level_id}__nested_{id(node)}"
                nested_prefix = f"{node_id}__"
                
                # Add the pipeline node itself
                node_data = self._create_pipeline_node_data(
                    node=node,
                    node_id=node_id,
                    level_id=level_id,
                    is_expanded=True
                )
                self.nodes.append(node_data)
                
                # Store mapping information for this PipelineNode
                self._pipeline_node_mappings[node_id] = {
                    "input_mapping": node.input_mapping if node.input_mapping else {},
                    "output_mapping": node.output_mapping if node.output_mapping else {},
                    "reverse_input_mapping": {inner: outer for outer, inner in (node.input_mapping or {}).items()},
                    "level_id": level_id,
                    "nested_level_id": nested_level_id
                }
                
                # Recursively process the nested pipeline
                self._process_pipeline(
                    pipeline=node.pipeline,
                    level_id=nested_level_id,
                    parent_level_id=level_id,
                    current_depth=current_depth + 1,
                    max_depth=max_depth,
                    prefix=nested_prefix,
                    parent_pipeline_node=node,
                    parent_pipeline_node_id=node_id
                )
                
                # After processing nested pipeline, update global output_to_node
                # with the actual inner nodes that produce outputs (with output mapping applied)
                self._update_output_mappings_for_nested(node, node_id, nested_prefix)
                
            elif is_nested and not should_expand:
                # Collapsed nested pipeline - treat as single node
                node_data = self._create_pipeline_node_data(
                    node=node,
                    node_id=node_id,
                    level_id=level_id,
                    is_expanded=False
                )
                self.nodes.append(node_data)
                
                # Track outputs
                if hasattr(node, 'output_name'):
                    output_names = node.output_name if isinstance(node.output_name, tuple) else (node.output_name,)
                    for out_name in output_names:
                        self._output_to_node_id[out_name] = node_id
                
            else:
                # Regular node (not a pipeline)
                node_data = self._create_node_data(
                    node=node,
                    node_id=node_id,
                    level_id=level_id,
                    pipeline=pipeline
                )
                self.nodes.append(node_data)
                
                # Track outputs
                if hasattr(node, 'output_name'):
                    output_names = node.output_name if isinstance(node.output_name, tuple) else (node.output_name,)
                    for out_name in output_names:
                        self._output_to_node_id[out_name] = node_id
        
        # Create edges for this level
        self._create_edges_for_level(
            graph_result=graph_result,
            level_id=level_id,
            prefix=prefix,
            pipeline=pipeline,
            parent_pipeline_node=parent_pipeline_node
        )

    def _compute_level_info(
        self,
        pipeline: Pipeline,
        level_id: str,
        parent_level_id: Optional[str],
        parent_pipeline_node_id: Optional[str]
    ) -> Dict[str, Any]:
        """Compute complete hierarchy information for a level.
        
        Args:
            pipeline: Pipeline at this level
            level_id: This level's ID
            parent_level_id: Parent level ID (None for root)
        
        Returns:
            Dictionary with per-level semantic data
        """
        graph_result = pipeline.graph
        
        # Get bound inputs at THIS specific level
        bound_inputs_at_this_level = list(pipeline.bound_inputs.keys())
        
        # Get unfulfilled inputs (not bound, not produced by siblings)
        unfulfilled_inputs = list(graph_result.root_args)
        
        # Inherited inputs from parent (for nested pipelines)
        inherited_inputs = []
        local_input_mapping = {}
        local_output_mapping = {}
        
        # If this is a nested pipeline (has parent), get mapping info
        # We'll populate this when processing PipelineNode
        
        return {
            "level_id": level_id,
            "parent_level_id": parent_level_id,
            "unfulfilled_inputs": unfulfilled_inputs,
            "bound_inputs_at_this_level": bound_inputs_at_this_level,
            "inherited_inputs": inherited_inputs,
            "local_output_mapping": local_output_mapping,
            "local_input_mapping": local_input_mapping,
            "parent_pipeline_node_id": parent_pipeline_node_id
        }

    def _create_node_data(
        self,
        node: Node,
        node_id: str,
        level_id: str,
        pipeline: Pipeline
    ) -> Dict[str, Any]:
        """Create semantic data for a regular node.
        
        Args:
            node: Node to serialize
            node_id: Unique node ID
            level_id: Level this node belongs to
            pipeline: Parent pipeline (for bound input checking)
        
        Returns:
            Dictionary with node semantic data
        """
        # Determine node type
        node_type = "STANDARD"
        if hasattr(node, "is_dual_node") and node.is_dual_node:
            node_type = "DUAL"
        
        # Get function name and label
        if hasattr(node, "func"):
            function_name = node.func.__name__ if hasattr(node.func, "__name__") else str(node.func)
            label = function_name
        else:
            function_name = "unknown"
            label = str(node)
        
        # Get output names
        if hasattr(node, "output_name"):
            output_name = node.output_name
            if isinstance(output_name, tuple):
                output_names = list(output_name)
            else:
                output_names = [output_name]
        else:
            output_names = []
        
        # Process inputs
        inputs = self._extract_input_info(node, pipeline)
        
        return {
            "id": node_id,
            "level_id": level_id,
            "node_type": node_type,
            "label": label,
            "function_name": function_name,
            "output_names": output_names,
            "inputs": inputs
        }

    def _create_pipeline_node_data(
        self,
        node: PipelineNode,
        node_id: str,
        level_id: str,
        is_expanded: bool
    ) -> Dict[str, Any]:
        """Create semantic data for a pipeline node.
        
        Args:
            node: PipelineNode to serialize
            node_id: Unique node ID
            level_id: Level this node belongs to
            is_expanded: Whether this pipeline is expanded or collapsed
        
        Returns:
            Dictionary with pipeline node semantic data
        """
        # Get label
        if node.name:
            label = node.name
        elif hasattr(node.pipeline, "name") and node.pipeline.name:
            label = node.pipeline.name
        else:
            label = "pipeline"
        
        # Get output names (after output mapping)
        if hasattr(node, "output_name"):
            output_name = node.output_name
            if isinstance(output_name, tuple):
                output_names = list(output_name)
            else:
                output_names = [output_name]
        else:
            output_names = []
        
        # Get inputs (from outer perspective)
        inputs = []
        for param_name in node.root_args:
            input_info = {
                "name": param_name,
                "type_hint": None,
                "default_value": None,
                "is_bound": param_name in node.pipeline.bound_inputs,
                "is_fulfilled_by_sibling": False  # Will be computed when creating edges
            }
            inputs.append(input_info)
        
        # Add mapping information
        input_mapping = node.input_mapping if node.input_mapping else {}
        output_mapping = node.output_mapping if node.output_mapping else {}
        
        return {
                "id": node_id,
            "level_id": level_id,
            "node_type": "PIPELINE",
                    "label": label,
            "function_name": label,
            "output_names": output_names,
                    "inputs": inputs,
            "is_expanded": is_expanded,
            "input_mapping": input_mapping,
            "output_mapping": output_mapping
        }

    def _extract_input_info(
        self,
        node: Node,
        pipeline: Pipeline
    ) -> List[Dict[str, Any]]:
        """Extract detailed input information from a node.
        
        Args:
            node: Node to extract from
            pipeline: Parent pipeline (for bound input checking)
        
        Returns:
            List of input dictionaries with semantic data
        """
        inputs = []
        bound_inputs = pipeline.bound_inputs
        
        if not hasattr(node, "root_args"):
            return inputs
        
        for param_name in node.root_args:
            input_info = {
                "name": param_name,
                "type_hint": self._extract_type_hint(node, param_name),
                "default_value": self._extract_default_value(node, param_name),
                "is_bound": param_name in bound_inputs,
                "is_fulfilled_by_sibling": False  # Will be computed when creating edges
            }
            inputs.append(input_info)
        
        return inputs

    def _extract_type_hint(self, node: Node, param_name: str) -> Optional[str]:
        """Extract type hint for a parameter.
        
        Args:
            node: Node to extract from
            param_name: Parameter name
        
        Returns:
            Type hint as string, or None if not available
        """
        if not hasattr(node, "func"):
            return None
        
        try:
            hints = get_type_hints(node.func)
            if param_name in hints:
                type_obj = hints[param_name]
                type_str = str(type_obj)
                # Clean up typing module prefix
                type_str = type_str.replace("typing.", "")
                # Clean up class representation
                type_str = type_str.replace("<class '", "").replace("'>", "")
                # Remove module prefixes
                import re
                type_str = re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\.", "", type_str)
                return type_str
        except Exception:
            pass
        
        return None

    def _extract_default_value(self, node: Node, param_name: str) -> Optional[str]:
        """Extract default value for a parameter.
        
        Args:
            node: Node to extract from
            param_name: Parameter name
        
        Returns:
            Default value as string, or None if not available
        """
        if not hasattr(node, "func"):
            return None
        
        try:
            sig = inspect.signature(node.func)
            param = sig.parameters[param_name]
            if param.default != inspect.Parameter.empty:
                default_str = repr(param.default)
                if len(default_str) > 20:
                    default_str = default_str[:17] + "..."
                return default_str
        except Exception:
            pass
        
        return None

    def _update_output_mappings_for_nested(
        self,
        pipeline_node: PipelineNode,
        pipeline_node_id: str,
        nested_prefix: str
    ):
        """Update global output_to_node_id with inner nodes from nested pipeline.
        
        Args:
            pipeline_node: The PipelineNode that was expanded
            pipeline_node_id: ID of the PipelineNode
            nested_prefix: Prefix used for nested node IDs
        """
        # Get the nested pipeline's graph
        nested_graph = pipeline_node.pipeline.graph
        
        # Get output mapping (inner_name -> outer_name)
        output_mapping = pipeline_node.output_mapping if pipeline_node.output_mapping else {}
        
        # For each node in the nested pipeline that produces an output
        for inner_output_name, inner_node in nested_graph.output_to_node.items():
            # Check if this output is remapped
            outer_output_name = output_mapping.get(inner_output_name, inner_output_name)
            
            # Find the node ID for this inner node
            inner_node_id = f"{nested_prefix}{id(inner_node)}"
            
            # Update global mapping with the actual inner node that produces this output
            self._output_to_node_id[outer_output_name] = inner_node_id

    def _create_edges_for_level(
        self,
        graph_result: Any,
        level_id: str,
        prefix: str,
        pipeline: Pipeline,
        parent_pipeline_node: Optional[PipelineNode] = None
    ):
        """Create edges for nodes at this level.
        
        Args:
            graph_result: GraphResult from pipeline
            level_id: Current level ID
            prefix: ID prefix for node IDs
            pipeline: Pipeline at this level
            parent_pipeline_node: The PipelineNode that wraps this pipeline (if nested)
        """
        # Get reverse input mapping if this is a nested pipeline
        reverse_input_mapping = {}
        if parent_pipeline_node and parent_pipeline_node.input_mapping:
            reverse_input_mapping = {
                inner: outer 
                for outer, inner in parent_pipeline_node.input_mapping.items()
            }
        
        # Build output_to_node mapping for this level
        output_to_node = graph_result.output_to_node
        
        # Track which nodes are expanded PipelineNodes at this level
        expanded_pipeline_node_ids = set()
        for node in graph_result.execution_order:
            node_id = f"{prefix}{id(node)}"
            if isinstance(node, PipelineNode) and node_id in self._pipeline_node_mappings:
                # This PipelineNode was expanded
                if self._pipeline_node_mappings[node_id].get("level_id") == level_id:
                    expanded_pipeline_node_ids.add(node_id)
        
        for node in graph_result.execution_order:
            node_id = f"{prefix}{id(node)}"
            target_level_id = self._node_id_to_level.get(node_id, level_id)

            # Skip container edges for expanded pipelines; their internals already wire inputs/outputs
            if node_id in expanded_pipeline_node_ids:
                continue
            
            # Add edges for dependencies (node -> node)
            for dep_node in graph_result.dependencies.get(node, []):
                dep_id = f"{prefix}{id(dep_node)}"
                
                # Check if the dependency is an expanded PipelineNode
                if dep_id in expanded_pipeline_node_ids:
                    # This is an expanded pipeline - find the actual inner nodes that produce needed outputs
                    # For each parameter this node needs, find the inner producer
                    for param in node.root_args:
                        # Check if this param is produced by the expanded pipeline
                        if param in self._output_to_node_id:
                            inner_producer_id = self._output_to_node_id[param]
                            # Check if the producer is within the nested pipeline
                            if inner_producer_id.startswith(f"{dep_id}__"):
                                # This is from the nested pipeline
                                source_level_id = self._node_id_to_level.get(inner_producer_id, level_id)
                                
                                # Check for output mapping label
                                edge_label = None
                                pipeline_node_obj = self._node_id_to_node.get(dep_id)
                                if pipeline_node_obj and hasattr(pipeline_node_obj, 'output_mapping') and pipeline_node_obj.output_mapping:
                                    # Find if param was renamed
                                    reverse_output_mapping = {outer: inner for inner, outer in pipeline_node_obj.output_mapping.items()}
                                    if param in reverse_output_mapping:
                                        inner_name = reverse_output_mapping[param]
                                        edge_label = f"{inner_name} → {param}"
                                
                                edge_id = f"e_{inner_producer_id}_{node_id}_{param}"
                                self.edges.append({
                                    "id": edge_id,
                                    "source": inner_producer_id,
                                    "target": node_id,
                                    "edge_type": "data_flow",
                                    "mapping_label": edge_label,
                                    "source_level_id": source_level_id,
                                    "target_level_id": target_level_id
                                })
                else:
                    # Regular dependency - not an expanded PipelineNode
                    source_level_id = self._node_id_to_level.get(dep_id, level_id)
                    
                    edge_id = f"e_{dep_id}_{node_id}"
                    self.edges.append({
                        "id": edge_id,
                        "source": dep_id,
                        "target": node_id,
                        "edge_type": "data_flow",
                        "mapping_label": None,
                        "source_level_id": source_level_id,
                        "target_level_id": target_level_id
                    })
            
            # Add edges for root args (parameter -> node)
            # Skip expanded PipelineNodes: their inputs are handled by inner nodes
            if node_id in expanded_pipeline_node_ids:
                continue

            for param_name in node.root_args:
                # First check if this parameter is produced by a node at this level
                if param_name in output_to_node:
                    # It's produced locally, edge already handled by dependencies
                    continue
                
                # Apply reverse input mapping (inner param -> outer param)
                outer_param_name = reverse_input_mapping.get(param_name, param_name)
                
                # Check if the outer parameter is produced by a node in an outer scope
                if outer_param_name in self._output_to_node_id:
                    # It's produced by a node (possibly in outer scope)
                    producer_node_id = self._output_to_node_id[outer_param_name]
                    source_level_id = self._node_id_to_level.get(producer_node_id, level_id)
                    
                    # Create node->node edge with mapping label if names differ
                    edge_label = None
                    if outer_param_name != param_name:
                        edge_label = f"{outer_param_name} → {param_name}"
                    
                    edge_id = f"e_{producer_node_id}_{node_id}_{param_name}"
                    self.edges.append({
                        "id": edge_id,
                        "source": producer_node_id,
                        "target": node_id,
                        "edge_type": "data_flow",
                        "mapping_label": edge_label,
                        "source_level_id": source_level_id,
                        "target_level_id": target_level_id
                    })
                else:
                    # It's an external input parameter (use outer name for display)
                    edge_id = f"e_input_{outer_param_name}_{node_id}"
                    
                    # Add mapping label if inner and outer names differ
                    edge_label = None
                    if outer_param_name != param_name:
                        edge_label = f"{outer_param_name} → {param_name}"
                    
                    self.edges.append({
                        "id": edge_id,
                        "source": f"input_{outer_param_name}",
                        "target": node_id,
                        "edge_type": "parameter_flow",
                        "mapping_label": edge_label,
                        "source_level_id": level_id,
                        "target_level_id": target_level_id
                    })

    def _compute_input_levels(self) -> Dict[str, str]:
        """Determine the most specific level for each external input."""
        input_levels: Dict[str, str] = {}

        # parent map for levels
        parent_map = {
            level["level_id"]: level.get("parent_level_id")
            for level in self.levels
        }
        # track where an input is declared as unfulfilled (external)
        declared_levels: Dict[str, str] = {}
        for level in self.levels:
            for param in level.get("unfulfilled_inputs", []):
                # prefer the first time we see it (outermost declaration)
                declared_levels.setdefault(param, level["level_id"])

        def ancestors(level_id: str) -> List[str]:
            chain = []
            current = level_id
            while current is not None:
                chain.append(current)
                current = parent_map.get(current)
            return chain

        node_level_map = dict(self._node_id_to_level)

        # Collect target levels for each input from parameter_flow edges
        input_targets: Dict[str, Set[str]] = {}
        for edge in self.edges:
            source = edge["source"]
            if isinstance(source, str) and source.startswith("input_"):
                param_name = source.replace("input_", "")
                target_level = edge.get("target_level_id")
                if not target_level:
                    target_node_id = edge["target"]
                    target_level = node_level_map.get(target_node_id, "root")
                input_targets.setdefault(param_name, set()).add(target_level)

        for param, target_levels in input_targets.items():
            # If declared as unfulfilled in a level, prefer that level
            if param in declared_levels:
                input_levels[param] = declared_levels[param]
                continue

            if not target_levels:
                input_levels[param] = "root"
                continue
            if len(target_levels) == 1:
                input_levels[param] = next(iter(target_levels))
                continue

            ancestor_chains = [ancestors(lvl) for lvl in target_levels]
            common = set(ancestor_chains[0])
            for chain in ancestor_chains[1:]:
                common &= set(chain)
            if common:
                depth_order = {lvl: ancestor_chains[0].index(lvl) for lvl in common}
                input_levels[param] = max(depth_order, key=depth_order.get)
            else:
                input_levels[param] = "root"

        return input_levels
