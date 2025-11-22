"""UI handler and backend for visualization frontends.

The UI handler owns visualization state and acts as the single backend for
both static (Graphviz) and interactive (React/ipywidgets) frontends. It is
responsible for:
- Accepting generic graph arguments (depth, grouping, etc.)
- Managing expansion/collapse state
- Computing visible nodes/edges and routing them to parents when collapsed
- Handling events and returning minimal diffs for frontends
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, get_type_hints

from ..node import Node
from ..pipeline import Pipeline
from ..pipeline_node import PipelineNode

_UNSET = object()


@dataclass
class GraphUpdate:
    """Represents a minimal update after an interaction."""

    view: Dict[str, Any]
    added_nodes: List[Dict[str, Any]]
    removed_nodes: List[str]
    updated_nodes: List[Dict[str, Any]]
    added_edges: List[Dict[str, Any]]
    removed_edges: List[str]
    updated_edges: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return {
            "view": self.view,
            "added_nodes": self.added_nodes,
            "removed_nodes": self.removed_nodes,
            "updated_nodes": self.updated_nodes,
            "added_edges": self.added_edges,
            "removed_edges": self.removed_edges,
            "updated_edges": self.updated_edges,
        }


class _GraphSerializer:
    """Internal graph serializer used by UIHandler."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self.levels: List[Dict[str, Any]] = []
        self._visited_nodes: Set[str] = set()
        self._node_id_to_level: Dict[str, str] = {}
        self._output_to_node_id: Dict[str, str] = {}
        self._node_id_to_node: Dict[str, Any] = {}
        self._pipeline_node_mappings: Dict[str, Dict[str, Any]] = {}

    def serialize(self, depth: Optional[int] = 1) -> Dict[str, Any]:
        """Serialize the pipeline into a frontend-agnostic structure."""
        self.nodes = []
        self.edges = []
        self.levels = []
        self._visited_nodes = set()
        self._node_id_to_level = {}
        self._output_to_node_id = {}
        self._node_id_to_node = {}
        self._pipeline_node_mappings = {}

        self._process_pipeline(
            pipeline=self.pipeline,
            level_id="root",
            parent_level_id=None,
            current_depth=1,
            max_depth=depth,
            prefix="",
        )

        input_levels = self._compute_input_levels()

        for edge in self.edges:
            if edge.get("edge_type") == "parameter_flow":
                source = edge.get("source", "")
                if isinstance(source, str) and source.startswith("input_"):
                    param_name = source.replace("input_", "")
                    if param_name in input_levels:
                        edge["source_level_id"] = input_levels[param_name]

        grouped_inputs = self._identify_grouped_inputs()

        return {
            "levels": self.levels,
            "nodes": self.nodes,
            "edges": self.edges,
            "input_levels": input_levels,
            "grouped_inputs": grouped_inputs,
        }

    def get_node_lookup(self) -> Dict[str, Any]:
        """Expose mapping of node_id -> underlying HyperNode instance."""
        return dict(self._node_id_to_node)

    def _process_pipeline(
        self,
        pipeline: Pipeline,
        level_id: str,
        parent_level_id: Optional[str],
        current_depth: int,
        max_depth: Optional[int],
        prefix: str,
        parent_pipeline_node: Optional[PipelineNode] = None,
        parent_pipeline_node_id: Optional[str] = None,
    ):
        """Recursively process a pipeline and compute per-level information."""
        if not hasattr(pipeline, "graph"):
            return

        graph_result = pipeline.graph

        level_info = self._compute_level_info(
            pipeline=pipeline,
            level_id=level_id,
            parent_level_id=parent_level_id,
            parent_pipeline_node_id=parent_pipeline_node_id,
        )
        self.levels.append(level_info)

        for node in graph_result.execution_order:
            node_id = f"{prefix}{id(node)}"

            if node_id in self._visited_nodes:
                continue
            self._visited_nodes.add(node_id)
            self._node_id_to_level[node_id] = level_id
            self._node_id_to_node[node_id] = node

            is_nested = isinstance(node, PipelineNode)
            should_expand = max_depth is None or current_depth < max_depth

            if is_nested and should_expand:
                nested_level_id = f"{level_id}__nested_{id(node)}"
                nested_prefix = f"{node_id}__"

                node_data = self._create_pipeline_node_data(
                    node=node,
                    node_id=node_id,
                    level_id=level_id,
                    is_expanded=True,
                )
                self.nodes.append(node_data)

                self._pipeline_node_mappings[node_id] = {
                    "input_mapping": node.input_mapping if node.input_mapping else {},
                    "output_mapping": node.output_mapping if node.output_mapping else {},
                    "reverse_input_mapping": {
                        inner: outer for outer, inner in (node.input_mapping or {}).items()
                    },
                    "level_id": level_id,
                    "nested_level_id": nested_level_id,
                }

                self._process_pipeline(
                    pipeline=node.pipeline,
                    level_id=nested_level_id,
                    parent_level_id=level_id,
                    current_depth=current_depth + 1,
                    max_depth=max_depth,
                    prefix=nested_prefix,
                    parent_pipeline_node=node,
                    parent_pipeline_node_id=node_id,
                )

                self._update_output_mappings_for_nested(node, node_id, nested_prefix)

            elif is_nested and not should_expand:
                node_data = self._create_pipeline_node_data(
                    node=node,
                    node_id=node_id,
                    level_id=level_id,
                    is_expanded=False,
                )
                self.nodes.append(node_data)

                if hasattr(node, "output_name"):
                    output_names = (
                        node.output_name
                        if isinstance(node.output_name, tuple)
                        else (node.output_name,)
                    )
                    for out_name in output_names:
                        self._output_to_node_id[out_name] = node_id

            else:
                node_data = self._create_node_data(
                    node=node, node_id=node_id, level_id=level_id, pipeline=pipeline
                )
                self.nodes.append(node_data)

                if hasattr(node, "output_name"):
                    output_names = (
                        node.output_name
                        if isinstance(node.output_name, tuple)
                        else (node.output_name,)
                    )
                    for out_name in output_names:
                        self._output_to_node_id[out_name] = node_id

        self._create_edges_for_level(
            graph_result=graph_result,
            level_id=level_id,
            prefix=prefix,
            pipeline=pipeline,
            parent_pipeline_node=parent_pipeline_node,
        )

    def _compute_level_info(
        self,
        pipeline: Pipeline,
        level_id: str,
        parent_level_id: Optional[str],
        parent_pipeline_node_id: Optional[str],
    ) -> Dict[str, Any]:
        """Compute complete hierarchy information for a level."""
        graph_result = pipeline.graph

        bound_inputs_at_this_level = list(pipeline.bound_inputs.keys())
        unfulfilled_inputs = list(graph_result.root_args)

        inherited_inputs: List[str] = []
        local_input_mapping: Dict[str, str] = {}
        local_output_mapping: Dict[str, str] = {}

        return {
            "level_id": level_id,
            "parent_level_id": parent_level_id,
            "unfulfilled_inputs": unfulfilled_inputs,
            "bound_inputs_at_this_level": bound_inputs_at_this_level,
            "inherited_inputs": inherited_inputs,
            "local_output_mapping": local_output_mapping,
            "local_input_mapping": local_input_mapping,
            "parent_pipeline_node_id": parent_pipeline_node_id,
        }

    def _create_node_data(
        self, node: Node, node_id: str, level_id: str, pipeline: Pipeline
    ) -> Dict[str, Any]:
        """Create semantic data for a regular node."""
        node_type = "STANDARD"
        if hasattr(node, "is_dual_node") and node.is_dual_node:
            node_type = "DUAL"

        if hasattr(node, "func"):
            function_name = node.func.__name__ if hasattr(node.func, "__name__") else str(node.func)
            label = function_name
        else:
            function_name = "unknown"
            label = str(node)

        if hasattr(node, "output_name"):
            output_name = node.output_name
            if isinstance(output_name, tuple):
                output_names = list(output_name)
            else:
                output_names = [output_name]
        else:
            output_names = []

        inputs = self._extract_input_info(node, pipeline)

        return {
            "id": node_id,
            "level_id": level_id,
            "node_type": node_type,
            "label": label,
            "function_name": function_name,
            "output_names": output_names,
            "inputs": inputs,
        }

    def _create_pipeline_node_data(
        self, node: PipelineNode, node_id: str, level_id: str, is_expanded: bool
    ) -> Dict[str, Any]:
        """Create semantic data for a pipeline node."""
        if node.name:
            label = node.name
        elif hasattr(node.pipeline, "name") and node.pipeline.name:
            label = node.pipeline.name
        else:
            label = "pipeline"

        if hasattr(node, "output_name"):
            output_name = node.output_name
            if isinstance(output_name, tuple):
                output_names = list(output_name)
            else:
                output_names = [output_name]
        else:
            output_names = []

        inputs = []
        for param_name in node.root_args:
            input_info = {
                "name": param_name,
                "type_hint": None,
                "default_value": None,
                "is_bound": param_name in node.pipeline.bound_inputs,
                "is_fulfilled_by_sibling": False,
            }
            inputs.append(input_info)

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
            "output_mapping": output_mapping,
        }

    def _extract_input_info(self, node: Node, pipeline: Pipeline) -> List[Dict[str, Any]]:
        """Extract detailed input information from a node."""
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
                "is_fulfilled_by_sibling": False,
            }
            inputs.append(input_info)

        return inputs

    def _extract_type_hint(self, node: Node, param_name: str) -> Optional[str]:
        """Extract type hint for a parameter."""
        if not hasattr(node, "func"):
            return None

        try:
            hints = get_type_hints(node.func)
            if param_name in hints:
                type_obj = hints[param_name]
                type_str = str(type_obj)
                type_str = type_str.replace("typing.", "")
                type_str = type_str.replace("<class '", "").replace("'>", "")
                import re

                type_str = re.sub(r"\\b[a-zA-Z_][a-zA-Z0-9_]*\\.", "", type_str)
                return type_str.split(".")[-1]
        except Exception:
            pass

        return None

    def _extract_default_value(self, node: Node, param_name: str) -> Optional[str]:
        """Extract default value for a parameter."""
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
        self, pipeline_node: PipelineNode, pipeline_node_id: str, nested_prefix: str
    ):
        """Update global output_to_node_id with inner nodes from nested pipeline."""
        nested_graph = pipeline_node.pipeline.graph

        output_mapping = pipeline_node.output_mapping if pipeline_node.output_mapping else {}

        for inner_output_name, inner_node in nested_graph.output_to_node.items():
            outer_output_name = output_mapping.get(inner_output_name, inner_output_name)

            inner_node_id = f"{nested_prefix}{id(inner_node)}"
            self._output_to_node_id[outer_output_name] = inner_node_id

    def _create_edges_for_level(
        self,
        graph_result: Any,
        level_id: str,
        prefix: str,
        pipeline: Pipeline,
        parent_pipeline_node: Optional[PipelineNode] = None,
    ):
        """Create edges for nodes at this level."""
        reverse_input_mapping = {}
        if parent_pipeline_node and parent_pipeline_node.input_mapping:
            reverse_input_mapping = {
                inner: outer for outer, inner in parent_pipeline_node.input_mapping.items()
            }

        output_to_node = graph_result.output_to_node

        expanded_pipeline_node_ids = set()
        for node in graph_result.execution_order:
            node_id = f"{prefix}{id(node)}"
            if isinstance(node, PipelineNode) and node_id in self._pipeline_node_mappings:
                if self._pipeline_node_mappings[node_id].get("level_id") == level_id:
                    expanded_pipeline_node_ids.add(node_id)

        for node in graph_result.execution_order:
            node_id = f"{prefix}{id(node)}"
            target_level_id = self._node_id_to_level.get(node_id, level_id)

            if node_id in expanded_pipeline_node_ids:
                continue

            for dep_node in graph_result.dependencies.get(node, []):
                dep_id = f"{prefix}{id(dep_node)}"

                if dep_id in expanded_pipeline_node_ids:
                    for param in node.root_args:
                        if param in self._output_to_node_id:
                            inner_producer_id = self._output_to_node_id[param]
                            if inner_producer_id.startswith(f"{dep_id}__"):
                                source_level_id = self._node_id_to_level.get(inner_producer_id, level_id)

                                edge_label = None
                                pipeline_node_obj = self._node_id_to_node.get(dep_id)
                                if (
                                    pipeline_node_obj
                                    and hasattr(pipeline_node_obj, "output_mapping")
                                    and pipeline_node_obj.output_mapping
                                ):
                                    reverse_output_mapping = {
                                        outer: inner for inner, outer in pipeline_node_obj.output_mapping.items()
                                    }
                                    if param in reverse_output_mapping:
                                        inner_name = reverse_output_mapping[param]
                                        edge_label = f"{inner_name} → {param}"

                                edge_id = f"e_{inner_producer_id}_{node_id}_{param}"
                                self.edges.append(
                                    {
                                        "id": edge_id,
                                        "source": inner_producer_id,
                                        "target": node_id,
                                        "edge_type": "data_flow",
                                        "mapping_label": edge_label,
                                        "source_level_id": source_level_id,
                                        "target_level_id": target_level_id,
                                    }
                                )
                else:
                    source_level_id = self._node_id_to_level.get(dep_id, level_id)

                    edge_id = f"e_{dep_id}_{node_id}"
                    self.edges.append(
                        {
                            "id": edge_id,
                            "source": dep_id,
                            "target": node_id,
                            "edge_type": "data_flow",
                            "mapping_label": None,
                            "source_level_id": source_level_id,
                            "target_level_id": target_level_id,
                        }
                    )

            if node_id in expanded_pipeline_node_ids:
                continue

            for param_name in node.root_args:
                if param_name in output_to_node:
                    continue

                outer_param_name = reverse_input_mapping.get(param_name, param_name)

                if outer_param_name in self._output_to_node_id:
                    producer_node_id = self._output_to_node_id[outer_param_name]
                    source_level_id = self._node_id_to_level.get(producer_node_id, level_id)

                    edge_label = None
                    if outer_param_name != param_name:
                        edge_label = f"{outer_param_name} → {param_name}"

                    edge_id = f"e_{producer_node_id}_{node_id}_{param_name}"
                    self.edges.append(
                        {
                            "id": edge_id,
                            "source": producer_node_id,
                            "target": node_id,
                            "edge_type": "data_flow",
                            "mapping_label": edge_label,
                            "source_level_id": source_level_id,
                            "target_level_id": target_level_id,
                        }
                    )
                else:
                    edge_id = f"e_input_{outer_param_name}_{node_id}"

                    edge_label = None
                    if outer_param_name != param_name:
                        edge_label = f"{outer_param_name} → {param_name}"

                    self.edges.append(
                        {
                            "id": edge_id,
                            "source": f"input_{outer_param_name}",
                            "target": node_id,
                            "edge_type": "parameter_flow",
                            "mapping_label": edge_label,
                            "source_level_id": level_id,
                            "target_level_id": target_level_id,
                        }
                    )

    def _compute_input_levels(self) -> Dict[str, str]:
        """Determine the most specific level for each external input."""
        input_levels: Dict[str, str] = {}

        parent_map = {level["level_id"]: level.get("parent_level_id") for level in self.levels}
        declared_levels: Dict[str, str] = {}
        for level in self.levels:
            for param in level.get("unfulfilled_inputs", []):
                declared_levels.setdefault(param, level["level_id"])

        def ancestors(level_id: str) -> List[str]:
            chain = []
            current = level_id
            while current is not None:
                chain.append(current)
                current = parent_map.get(current)
            return chain

        node_level_map = dict(self._node_id_to_level)

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

    def _identify_grouped_inputs(self) -> Dict[str, Dict[str, List[str]]]:
        """Identify grouped inputs for each node."""
        input_consumers: Dict[str, Set[str]] = {}
        for edge in self.edges:
            if edge.get("edge_type") == "parameter_flow":
                source = edge.get("source")
                target = edge.get("target")
                if isinstance(source, str) and source.startswith("input_"):
                    param_name = source.replace("input_", "")
                    if param_name not in input_consumers:
                        input_consumers[param_name] = set()
                    input_consumers[param_name].add(target)

        bound_inputs_set: Set[str] = set()
        for level in self.levels:
            bound_inputs_set.update(level.get("bound_inputs_at_this_level", []))

        grouped_inputs_by_consumer: Dict[str, Dict[str, List[str]]] = {}
        for param_name, consumers in input_consumers.items():
            if len(consumers) == 1:
                consumer_id = next(iter(consumers))
                is_bound = param_name in bound_inputs_set
                group_type = "bound" if is_bound else "unbound"
                if consumer_id not in grouped_inputs_by_consumer:
                    grouped_inputs_by_consumer[consumer_id] = {"bound": [], "unbound": []}
                grouped_inputs_by_consumer[consumer_id][group_type].append(param_name)

        final_groups: Dict[str, Dict[str, List[str]]] = {}
        for consumer_id, groups in grouped_inputs_by_consumer.items():
            has_groups = False
            filtered_groups = {"bound": [], "unbound": []}

            for group_type in ["bound", "unbound"]:
                params = groups[group_type]
                if len(params) >= 2:
                    filtered_groups[group_type] = sorted(params)
                    has_groups = True

            if has_groups:
                final_groups[consumer_id] = filtered_groups

        for level in self.levels:
            bound_at_level = level.get("bound_inputs_at_this_level", [])
            parent_pnode_id = level.get("parent_pipeline_node_id")
            if not bound_at_level or not parent_pnode_id:
                continue
            bound_params = sorted(bound_at_level)
            if len(bound_params) >= 2:
                existing = final_groups.get(parent_pnode_id, {"bound": [], "unbound": []})
                merged_bound = sorted(set(existing.get("bound", [])).union(bound_params))
                existing["bound"] = merged_bound
                final_groups[parent_pnode_id] = existing

        return final_groups


class UIHandler:
    """Backend controller for visualization state and events."""

    GENERIC_OPTIONS = {"depth", "group_inputs", "show_output_types"}

    def __init__(
        self,
        pipeline: Pipeline,
        depth: Optional[int] = 1,
        group_inputs: bool = True,
        show_output_types: bool = False,
    ):
        self.pipeline = pipeline
        self.serializer = _GraphSerializer(pipeline)

        # Cache the fully expanded semantic graph once
        self.full_graph = self.serializer.serialize(depth=None)
        self._node_lookup = self.serializer.get_node_lookup()

        # Options that affect the view (shared across frontends)
        self.depth = depth
        self.group_inputs = group_inputs
        self.show_output_types = show_output_types

        # State
        self.expanded_nodes: Set[str] = set()
        self.hidden_nodes: Set[str] = set()

        # Cache for node lookups
        self._node_map: Dict[str, Dict[str, Any]] = {
            node["id"]: node for node in self.full_graph["nodes"]
        }

        # Map level_id to parent node ID
        self._level_parent_map: Dict[str, str] = {}
        for level in self.full_graph["levels"]:
            if level.get("parent_pipeline_node_id"):
                self._level_parent_map[level["level_id"]] = level["parent_pipeline_node_id"]
        self._level_depth: Dict[str, int] = self._compute_level_depths()

        # Pre-compute grouped inputs based on options
        self.grouped_inputs = self._filter_grouped_inputs(
            self.full_graph.get("grouped_inputs", {}),
            group_inputs=self.group_inputs,
        )

        # Initialize view cache
        self._current_view: Optional[Dict[str, Any]] = None
        self.set_initial_depth(depth)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def apply_options(
        self,
        depth: Any = _UNSET,
        group_inputs: Optional[bool] = None,
        show_output_types: Optional[bool] = None,
    ) -> None:
        """Update generic options and recompute grouped inputs if needed."""
        options_changed = False

        if depth is not _UNSET and depth != self.depth:
            self.depth = depth
            self.set_initial_depth(depth)
            options_changed = True

        if group_inputs is not None and group_inputs != self.group_inputs:
            self.group_inputs = group_inputs
            options_changed = True

        if show_output_types is not None and show_output_types != self.show_output_types:
            self.show_output_types = show_output_types
            options_changed = True

        if options_changed:
            self.grouped_inputs = self._filter_grouped_inputs(
                self.full_graph.get("grouped_inputs", {}),
                group_inputs=self.group_inputs,
            )
            # clear cached view so it is recomputed on next request
            self._current_view = None

    def expand_node(self, node_id: str) -> None:
        """Mark a node as expanded."""
        if node_id in self._node_map:
            self.expanded_nodes.add(node_id)
            self._current_view = None

    def collapse_node(self, node_id: str) -> None:
        """Collapse a pipeline node and all its descendants."""
        if node_id in self.expanded_nodes:
            self.expanded_nodes.remove(node_id)

            # Recursively collapse all descendants
            prefix = f"{node_id}__"
            to_remove = [eid for eid in self.expanded_nodes if eid.startswith(prefix)]
            for rid in to_remove:
                self.expanded_nodes.remove(rid)

            self._current_view = None

    def toggle_node(self, node_id: str) -> None:
        """Toggle the expansion state of a node."""
        if node_id in self.expanded_nodes:
            self.collapse_node(node_id)
        else:
            self.expand_node(node_id)

    def set_initial_depth(self, depth: Optional[int]) -> None:
        """Set the initial expansion state based on depth."""
        self.expanded_nodes.clear()

        if depth is None:
            # Fully expand all pipeline nodes
            for node in self.full_graph["nodes"]:
                if node.get("node_type") == "PIPELINE":
                    self.expanded_nodes.add(node["id"])
            self._current_view = None
            return

        if depth <= 1:
            self._current_view = None
            return

        for node in self.full_graph["nodes"]:
            if node.get("node_type") == "PIPELINE":
                level_id = node.get("level_id", "root")
                current_depth = self._level_depth.get(level_id, 1)

                if current_depth < depth:
                    self.expanded_nodes.add(node["id"])

        self._current_view = None

    def get_view_data(self) -> Dict[str, Any]:
        """Compute the visible graph based on current state."""
        if self._current_view is not None:
            return self._current_view

        visible_nodes = []
        visible_node_ids = set()

        for node in self.full_graph["nodes"]:
            if self._is_node_visible(node):
                view_node = node.copy()
                view_node["is_expanded"] = node["id"] in self.expanded_nodes
                self._attach_hypernode_metadata(view_node)
                visible_nodes.append(view_node)
                visible_node_ids.add(node["id"])

        visible_edges = []
        for edge in self.full_graph["edges"]:
            source = edge["source"]
            target = edge["target"]

            visible_source = self._find_visible_ancestor(source, visible_node_ids)
            visible_target = self._find_visible_ancestor(target, visible_node_ids)

            if visible_source and visible_target and visible_source != visible_target:
                view_edge = edge.copy()
                view_edge["source"] = visible_source
                view_edge["target"] = visible_target

                if visible_source != source or visible_target != target:
                    view_edge["id"] = f"e_{visible_source}_{visible_target}_{edge.get('id', '')}"

                visible_edges.append(view_edge)

        unique_edges = []
        seen_edges = set()
        for edge in visible_edges:
            edge_key = (edge["source"], edge["target"], edge.get("mapping_label"))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                unique_edges.append(edge)

        self._current_view = self._json_safe({
            "nodes": visible_nodes,
            "edges": unique_edges,
            "levels": self.full_graph.get("levels", []),
            "input_levels": self.full_graph.get("input_levels", {}),
            "grouped_inputs": self.grouped_inputs,
            "applied_options": {
                "depth": self.depth,
                "group_inputs": self.group_inputs,
                "show_output_types": self.show_output_types,
            },
            **self._build_render_graph(
                visible_nodes=visible_nodes,
                visible_edges=unique_edges,
                levels=self.full_graph.get("levels", []),
                input_levels=self.full_graph.get("input_levels", {}),
                grouped_inputs=self.grouped_inputs,
            ),
        })
        return self._current_view

    def get_full_graph_with_state(self, include_events: bool = False) -> Dict[str, Any]:
        """Get the full graph with current expansion state applied."""
        nodes = []
        for node in self.full_graph["nodes"]:
            view_node = node.copy()
            view_node["is_expanded"] = node["id"] in self.expanded_nodes
            self._attach_hypernode_metadata(view_node)
            nodes.append(view_node)

        result = self._json_safe({
            "nodes": nodes,
            "edges": self.full_graph["edges"],
            "levels": self.full_graph.get("levels", []),
            "input_levels": self.full_graph.get("input_levels", {}),
            "grouped_inputs": self.grouped_inputs,
            "applied_options": {
                "depth": self.depth,
                "group_inputs": self.group_inputs,
                "show_output_types": self.show_output_types,
            },
            **self._build_render_graph(
                visible_nodes=nodes,
                visible_edges=self.full_graph["edges"],
                levels=self.full_graph.get("levels", []),
                input_levels=self.full_graph.get("input_levels", {}),
                grouped_inputs=self.grouped_inputs,
            ),
        })
        if include_events:
            result["event_index"] = self.build_event_index()
        return result

    def prepare_for_engine(
        self,
        engine_name: str,
        **options: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Split generic vs frontend-specific options and return graph + engine opts."""
        generic_kwargs = {k: v for k, v in options.items() if k in self.GENERIC_OPTIONS}
        # Graphviz passes show_types; map it to show_output_types for metadata extraction
        if "show_types" in options and "show_output_types" not in generic_kwargs:
            generic_kwargs["show_output_types"] = options["show_types"]
        frontend_kwargs = {k: v for k, v in options.items() if k not in self.GENERIC_OPTIONS}

        # Apply generic options first
        self.apply_options(
            depth=generic_kwargs.get("depth", _UNSET),
            group_inputs=generic_kwargs.get("group_inputs", self.group_inputs),
            show_output_types=generic_kwargs.get("show_output_types", self.show_output_types),
        )

        if engine_name == "ipywidget":
            graph_data = self.get_full_graph_with_state()
        else:
            graph_data = self.get_view_data()

        return graph_data, frontend_kwargs

    def handle_event(self, event: Dict[str, Any]) -> GraphUpdate:
        """Handle interactive events and return minimal diffs.

        Supported events:
            {"type": "toggle", "node_id": "..."}
            {"type": "expand", "node_id": "..."}
            {"type": "collapse", "node_id": "..."}
            {"type": "set_depth", "depth": int}
        """
        before = self.get_view_data()
        event_type = event.get("type")

        if event_type == "toggle":
            self.toggle_node(event["node_id"])
        elif event_type == "expand":
            self.expand_node(event["node_id"])
        elif event_type == "collapse":
            self.collapse_node(event["node_id"])
        elif event_type == "set_depth":
            new_depth = event.get("depth")
            if new_depth is not None:
                self.apply_options(depth=new_depth)
        else:
            # Unknown event: no-op but still return view
            return GraphUpdate(
                view=before,
                added_nodes=[],
                removed_nodes=[],
                updated_nodes=[],
                added_edges=[],
                removed_edges=[],
                updated_edges=[],
            )

        after = self.get_view_data()
        return self._compute_diff(before, after)

    def simulate_event(self, event: Dict[str, Any]) -> GraphUpdate:
        """Compute an event update without mutating handler state."""
        snapshot_expanded = set(self.expanded_nodes)
        snapshot_view = self._current_view
        update = self.handle_event(event)
        # Restore state/cache
        self.expanded_nodes = snapshot_expanded
        self._current_view = snapshot_view
        return update

    def build_event_index(self) -> Dict[str, Dict[str, Any]]:
        """Precompute expand/collapse updates for each pipeline node."""
        index: Dict[str, Dict[str, Any]] = {}
        for node in self.full_graph["nodes"]:
            if node.get("node_type") != "PIPELINE":
                continue
            node_id = node["id"]
            index[node_id] = {}
            # Expand patch
            expand_update = self.simulate_event({"type": "expand", "node_id": node_id})
            index[node_id]["expand"] = expand_update.to_dict()
            # Collapse patch
            collapse_update = self.simulate_event({"type": "collapse", "node_id": node_id})
            index[node_id]["collapse"] = collapse_update.to_dict()
        return index

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _is_node_visible(self, node: Dict[str, Any]) -> bool:
        level_id = node.get("level_id")

        if level_id == "root":
            return True

        parent_node_id = self._level_parent_map.get(level_id)
        if parent_node_id:
            if parent_node_id not in self.expanded_nodes:
                return False

            parent_node = self._node_map.get(parent_node_id)
            if parent_node:
                return self._is_node_visible(parent_node)

        return True

    def _find_visible_ancestor(self, node_id: str, visible_node_ids: Set[str]) -> Optional[str]:
        if node_id in visible_node_ids:
            return node_id

        if isinstance(node_id, str) and node_id.startswith("input_"):
            return node_id

        current_id = node_id
        while current_id:
            node = self._node_map.get(current_id)
            if not node:
                return None

            level_id = node.get("level_id")
            parent_node_id = self._level_parent_map.get(level_id)

            if parent_node_id:
                if parent_node_id in visible_node_ids:
                    return parent_node_id
                current_id = parent_node_id
            else:
                return None

        return None

    def _find_visible_level(self, level_id: str) -> str:
        """Find the nearest visible ancestor level for a given level."""
        if level_id == "root":
            return "root"

        # If the level itself is visible (parent is expanded), return it
        parent_node_id = self._level_parent_map.get(level_id)
        if not parent_node_id:
            return "root"

        if parent_node_id in self.expanded_nodes:
            return level_id

        # Parent is collapsed, so this level is hidden.
        # Recurse up to find the visible level of the parent node.
        parent_node = self._node_map.get(parent_node_id)
        if parent_node:
            return self._find_visible_level(parent_node.get("level_id", "root"))

        return "root"

    def _compute_level_depths(self) -> Dict[str, int]:
        """Compute depth per level using parent relationships (root=1)."""
        depth: Dict[str, int] = {"root": 1}
        levels = {lvl["level_id"]: lvl for lvl in self.full_graph.get("levels", [])}

        changed = True
        while changed:
            changed = False
            for level_id, level in levels.items():
                if level_id in depth:
                    continue
                parent = level.get("parent_level_id")
                if parent is None:
                    depth[level_id] = 1
                    changed = True
                elif parent in depth:
                    depth[level_id] = depth[parent] + 1
                    changed = True
        return depth

    def _attach_hypernode_metadata(self, view_node: Dict[str, Any]) -> None:
        """Enrich view node with metadata derived from the underlying HyperNode."""
        node_obj = self._node_lookup.get(view_node["id"])
        if not node_obj:
            return

        if hasattr(node_obj, "root_args"):
            try:
                view_node["root_args"] = tuple(node_obj.root_args)
            except Exception:
                pass

        if hasattr(node_obj, "output_name"):
            try:
                view_node["output_name"] = node_obj.output_name
                if self.show_output_types:
                    view_node["output_types"] = self._extract_output_types(
                        node_obj, view_node.get("output_names", [])
                    )
            except Exception:
                pass

        if hasattr(node_obj, "unfulfilled_args"):
            try:
                view_node["unfulfilled_args"] = tuple(node_obj.unfulfilled_args)
            except Exception:
                pass

        if hasattr(node_obj, "bound_inputs"):
            try:
                bound_inputs = node_obj.bound_inputs
                if isinstance(bound_inputs, dict):
                    view_node["bound_inputs"] = sorted(bound_inputs.keys())
                else:
                    view_node["bound_inputs"] = list(bound_inputs) if bound_inputs else []
            except Exception:
                pass

        # If this is a PipelineNode, expose inner pipeline details for debugging/visualization
        if hasattr(node_obj, "pipeline"):
            try:
                inner_pipeline = node_obj.pipeline
                view_node["inner_root_args"] = tuple(getattr(inner_pipeline.graph, "root_args", ()))
                inner_bound_inputs = getattr(inner_pipeline, "bound_inputs", {})
                if callable(inner_bound_inputs):
                    inner_bound_inputs = inner_bound_inputs()
                if isinstance(inner_bound_inputs, dict):
                    view_node["inner_bound_inputs"] = sorted(inner_bound_inputs.keys())
                else:
                    view_node["inner_bound_inputs"] = (
                        list(inner_bound_inputs) if inner_bound_inputs else []
                    )
            except Exception:
                pass

    def _extract_output_types(self, node_obj: Any, output_names: List[str]) -> List[str]:
        """Best-effort extraction of output types from annotations."""
        try:
            import inspect
            from typing import get_type_hints
            if hasattr(node_obj, "func"):
                func = getattr(node_obj, "func")
                sig = inspect.signature(func)
                ret = sig.return_annotation
                if ret is inspect._empty:
                    hints = get_type_hints(func)
                    ret = hints.get("return", inspect._empty)
            else:
                ret = None
        except Exception:
            ret = None

        def _fmt(t):
            try:
                name = getattr(t, "__name__", str(t))
            except Exception:
                name = str(t)
            if not isinstance(name, str):
                name = str(name)
            name = name.replace("typing.", "").replace("<class '", "").replace("'>", "")
            if "." in name:
                name = name.split(".")[-1]
            return name

        if not output_names:
            output_names = []
        if ret in (None, inspect._empty):
            return []
        if isinstance(ret, tuple) or isinstance(ret, list):
            return [_fmt(t) for t in ret]
        if hasattr(ret, "__iter__") and not isinstance(ret, (str, bytes)):
            try:
                return [_fmt(t) for t in list(ret)]
            except Exception:
                pass
        # Single return type
        if len(output_names) <= 1:
            return [_fmt(ret)]
        return [_fmt(ret) for _ in output_names]

    def _json_safe(self, obj: Any) -> Any:
        """Best-effort sanitization to keep graph data JSON serializable."""
        primitive = (str, int, float, bool, type(None))
        if isinstance(obj, primitive):
            return obj
        if isinstance(obj, dict):
            return {str(k): self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._json_safe(v) for v in obj]
        try:
            return obj.__class__.__name__
        except Exception:
            return str(obj)

    def _filter_grouped_inputs(
        self,
        grouped_inputs: Dict[str, Dict[str, List[str]]],
        group_inputs: bool,
    ) -> Dict[str, Dict[str, List[str]]]:
        """Apply grouping options on top of serializer output."""
        if not group_inputs:
            return {}
        
        min_size = 2

        filtered: Dict[str, Dict[str, List[str]]] = {}
        for consumer_id, groups in grouped_inputs.items():
            bound_candidates = groups.get("bound", [])
            unbound_candidates = groups.get("unbound", [])

            bound = [p for p in bound_candidates if len(bound_candidates) >= min_size]
            unbound = [p for p in unbound_candidates if len(unbound_candidates) >= min_size]

            if bound or unbound:
                filtered[consumer_id] = {
                    "bound": sorted(bound),
                    "unbound": sorted(unbound),
                }
        return filtered

    def _compute_diff(self, before: Dict[str, Any], after: Dict[str, Any]) -> GraphUpdate:
        """Compute minimal diff between two views."""
        before_nodes = {n["id"]: n for n in before.get("nodes", [])}
        after_nodes = {n["id"]: n for n in after.get("nodes", [])}

        before_edges = {e["id"]: e for e in before.get("edges", [])}
        after_edges = {e["id"]: e for e in after.get("edges", [])}

        added_nodes = [after_nodes[nid] for nid in after_nodes.keys() - before_nodes.keys()]
        removed_nodes = list(before_nodes.keys() - after_nodes.keys())
        updated_nodes = [
            after_nodes[nid]
            for nid in after_nodes.keys() & before_nodes.keys()
            if after_nodes[nid] != before_nodes[nid]
        ]

        added_edges = [after_edges[eid] for eid in after_edges.keys() - before_edges.keys()]
        removed_edges = list(before_edges.keys() - after_edges.keys())
        updated_edges = [
            after_edges[eid]
            for eid in after_edges.keys() & before_edges.keys()
            if after_edges[eid] != before_edges[eid]
        ]

        return GraphUpdate(
            view=after,
            added_nodes=added_nodes,
            removed_nodes=removed_nodes,
            updated_nodes=updated_nodes,
            added_edges=added_edges,
            removed_edges=removed_edges,
            updated_edges=updated_edges,
        )

    def _build_render_graph(
        self,
        visible_nodes: List[Dict[str, Any]],
        visible_edges: List[Dict[str, Any]],
        levels: List[Dict[str, Any]],
        input_levels: Dict[str, str],
        grouped_inputs: Dict[str, Dict[str, List[str]]],
    ) -> Dict[str, Any]:
        """Prepare render-ready nodes/edges so frontends can stay dumb."""
        level_map = {lvl["level_id"]: lvl for lvl in levels}
        nodes_by_id = {n["id"]: n for n in visible_nodes}

        # Build input metadata lookup for labels
        input_meta: Dict[str, Dict[str, Any]] = {}
        for node in visible_nodes:
            for meta in node.get("inputs", []):
                name = meta.get("name")
                if name and name not in input_meta:
                    input_meta[name] = meta

        grouped_param_names = {
            param
            for buckets in grouped_inputs.values()
            for params in buckets.values()
            for param in params
        }

        inputs_by_level: Dict[str, List[str]] = {}
        for name, lvl in input_levels.items():
            inputs_by_level.setdefault(lvl, []).append(name)

        bound_inputs_set = set().union(
            *[set(level.get("bound_inputs_at_this_level", [])) for level in levels]
        )

        render_nodes: List[Dict[str, Any]] = []
        render_edges: List[Dict[str, Any]] = []

        def build_input_label(param: str) -> str:
            meta = input_meta.get(param, {})
            type_hint = meta.get("type_hint")
            label = param
            if type_hint:
                label = f"{param} : {type_hint}"
            return label

        # Add function/pipeline nodes as-is
        for node in visible_nodes:
            render_nodes.append(node.copy())

        # Add per-level inputs (skipping grouped ones)
        for level_id, params in inputs_by_level.items():
            # Ensure we place the input in a visible level
            visible_level_id = self._find_visible_level(level_id)
            
            for param_name in params:
                if param_name in grouped_param_names:
                    continue
                is_bound = param_name in bound_inputs_set
                render_nodes.append(
                    {
                        "id": f"input_{param_name}",
                        "label": build_input_label(param_name),
                        "node_type": "INPUT",
                        "level_id": visible_level_id,
                        "is_bound": is_bound,
                    }
                )

        # Add grouped input nodes and edges to consumers
        for consumer_id, buckets in grouped_inputs.items():
            consumer_node = nodes_by_id.get(consumer_id)
            if consumer_node and consumer_node.get("node_type") == "PIPELINE":
                # Skip wrapper-level grouping; use actual consumer nodes
                continue
            consumer_level = nodes_by_id.get(consumer_id, {}).get("level_id", "root")
            for group_type, params in buckets.items():
                if not params:
                    continue
                group_id = f"group_{consumer_id}_{group_type}"
                is_bound = group_type == "bound"
                label_lines = []
                for p in sorted(params):
                    meta = input_meta.get(p, {})
                    type_hint = meta.get("type_hint")
                    line = p
                    if type_hint:
                        line = f"{line} : {type_hint}"
                    label_lines.append(line)
                render_nodes.append(
                    {
                        "id": group_id,
                        "label": "\\n".join(label_lines),
                        "node_type": "INPUT_GROUP",
                        "level_id": consumer_level,
                        "is_bound": is_bound,
                        "params": params,
                    }
                )
                render_edges.append(
                    {
                        "id": f"e_{group_id}_{consumer_id}",
                        "source": group_id,
                        "target": consumer_id,
                        "edge_type": "parameter_flow",
                    }
                )

        # Add edges, skipping those covered by grouped params
        for edge in visible_edges:
            source = edge.get("source")
            if isinstance(source, str) and source.startswith("input_"):
                param_name = source.replace("input_", "")
                if param_name in grouped_param_names:
                    consumers = grouped_inputs.get(edge.get("target"), {})
                    if consumers:
                        continue
            render_edges.append(edge.copy())

        return {
            "render_nodes": render_nodes,
            "render_edges": render_edges,
        }
