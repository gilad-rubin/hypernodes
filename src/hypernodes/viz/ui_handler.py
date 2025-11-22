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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from ..pipeline import Pipeline
from .graph_serializer import GraphSerializer


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


class UIHandler:
    """Backend controller for visualization state and events."""

    GENERIC_OPTIONS = {"depth", "group_inputs", "min_arg_group_size"}

    def __init__(
        self,
        pipeline: Pipeline,
        depth: Optional[int] = 1,
        group_inputs: bool = True,
        min_arg_group_size: Optional[int] = 2,
    ):
        self.pipeline = pipeline
        self.serializer = GraphSerializer(pipeline)

        # Cache the fully expanded semantic graph once
        self.full_graph = self.serializer.serialize(depth=None)
        self._node_lookup = self.serializer.get_node_lookup()

        # Options that affect the view (shared across frontends)
        self.depth = depth
        self.group_inputs = group_inputs
        self.min_arg_group_size = min_arg_group_size

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
            min_size=self.min_arg_group_size,
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
        min_arg_group_size: Optional[int] = None,
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

        if min_arg_group_size is not None and min_arg_group_size != self.min_arg_group_size:
            self.min_arg_group_size = min_arg_group_size
            options_changed = True

        if options_changed:
            self.grouped_inputs = self._filter_grouped_inputs(
                self.full_graph.get("grouped_inputs", {}),
                group_inputs=self.group_inputs,
                min_size=self.min_arg_group_size,
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

        self._current_view = {
            "nodes": visible_nodes,
            "edges": unique_edges,
            "levels": self.full_graph.get("levels", []),
            "input_levels": self.full_graph.get("input_levels", {}),
            "grouped_inputs": self.grouped_inputs,
            "applied_options": {
                "depth": self.depth,
                "group_inputs": self.group_inputs,
                "min_arg_group_size": self.min_arg_group_size,
            },
        }
        return self._current_view

    def get_full_graph_with_state(self, include_events: bool = False) -> Dict[str, Any]:
        """Get the full graph with current expansion state applied."""
        nodes = []
        for node in self.full_graph["nodes"]:
            view_node = node.copy()
            view_node["is_expanded"] = node["id"] in self.expanded_nodes
            self._attach_hypernode_metadata(view_node)
            nodes.append(view_node)

        result = {
            "nodes": nodes,
            "edges": self.full_graph["edges"],
            "levels": self.full_graph.get("levels", []),
            "input_levels": self.full_graph.get("input_levels", {}),
            "grouped_inputs": self.grouped_inputs,
            "applied_options": {
                "depth": self.depth,
                "group_inputs": self.group_inputs,
                "min_arg_group_size": self.min_arg_group_size,
            },
        }
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
        frontend_kwargs = {k: v for k, v in options.items() if k not in self.GENERIC_OPTIONS}

        # Apply generic options first
        self.apply_options(
            depth=generic_kwargs.get("depth", _UNSET),
            group_inputs=generic_kwargs.get("group_inputs", self.group_inputs),
            min_arg_group_size=generic_kwargs.get("min_arg_group_size", self.min_arg_group_size),
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
                view_node["bound_inputs"] = bound_inputs.copy() if isinstance(bound_inputs, dict) else bound_inputs
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
                view_node["inner_bound_inputs"] = (
                    inner_bound_inputs.copy() if isinstance(inner_bound_inputs, dict) else inner_bound_inputs
                )
            except Exception:
                pass

    def _filter_grouped_inputs(
        self,
        grouped_inputs: Dict[str, Dict[str, List[str]]],
        group_inputs: bool,
        min_size: Optional[int],
    ) -> Dict[str, Dict[str, List[str]]]:
        """Apply grouping options on top of serializer output."""
        if not group_inputs or min_size is None:
            return {}

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
