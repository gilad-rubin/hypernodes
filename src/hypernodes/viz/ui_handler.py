"""UI handler and backend for visualization frontends.

The UI handler owns visualization state and acts as the single backend for
both static (Graphviz) and interactive (React/ipywidgets) frontends. It is
responsible for:
- Accepting generic graph arguments (depth, grouping, etc.)
- Managing expansion/collapse state
- Delegating graph generation to GraphWalker
"""

from __future__ import annotations

from typing import Any, Optional, Set

from ..pipeline import Pipeline
from ..pipeline_node import PipelineNode as HyperPipelineNode
from .graph_walker import GraphWalker
from .structures import VisualizationGraph

_UNSET = object()


class UIHandler:
    """Backend controller for visualization state and events."""

    def __init__(
        self,
        pipeline: Pipeline,
        depth: Optional[int] = 1,
        group_inputs: bool = True,
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
                node_id = f"{prefix}{id(node)}"
                self.expanded_nodes.add(node_id)
                self._expand_all(node.pipeline, prefix=f"{node_id}__")

    def _expand_to_depth(self, pipeline: Pipeline, current_depth: int, max_depth: int, prefix: str = ""):
        if current_depth >= max_depth:
            return
        
        for node in pipeline.graph.execution_order:
            if isinstance(node, HyperPipelineNode):
                node_id = f"{prefix}{id(node)}"
                self.expanded_nodes.add(node_id)
                self._expand_to_depth(node.pipeline, current_depth + 1, max_depth, prefix=f"{node_id}__")

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

    def get_visualization_data(self) -> VisualizationGraph:
        """Generate the flat visualization graph based on current state."""
        walker = GraphWalker(
            pipeline=self.pipeline,
            expanded_nodes=self.expanded_nodes,
            group_inputs=self.group_inputs
        )
        return walker.get_visualization_data()
