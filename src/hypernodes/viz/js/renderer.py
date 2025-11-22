from typing import Any, Dict, List

from ..structures import (
    DataNode,
    DualNode,
    FunctionNode,
    GroupDataNode,
    PipelineNode,
    VisualizationGraph,
)


class JSRenderer:
    """Transforms VisualizationGraph to React Flow node/edge structures.
    
    Layout is performed client-side using ELK (Eclipse Layout Kernel) with
    the Sugiyama-style layered algorithm, which arranges nodes in hierarchical
    layers and minimizes edge crossings for clean DAG visualization.
    """

    def render(
        self,
        graph_data: VisualizationGraph,
        theme: str = "CYBERPUNK",
        initial_depth: int = 1,
        theme_debug: bool = False,
        pan_on_scroll: bool = False,
    ) -> Dict[str, Any]:
        """Transform graph data to React Flow format."""
        nodes = []
        edges = []

        viz_nodes = graph_data.nodes
        viz_edges = graph_data.edges

        for node in viz_nodes:
            rf_node = {
                "id": node.id,
                "position": {"x": 0, "y": 0},
                "data": {"theme": theme},
            }

            if node.parent_id:
                rf_node["parentNode"] = node.parent_id
                rf_node["extent"] = "parent"

            # Map specific node types
            if isinstance(node, FunctionNode):
                rf_node["type"] = "custom"
                rf_node["data"].update({
                    "label": node.label,
                    "nodeType": "FUNCTION",
                    "functionName": node.function_name,
                })
                if isinstance(node, DualNode):
                    rf_node["data"]["nodeType"] = "DUAL"
            
            elif isinstance(node, PipelineNode):
                if node.is_expanded:
                    rf_node["type"] = "pipelineGroup"
                    rf_node["style"] = {"width": 600, "height": 400} # Initial size, ELK Sugiyama layout will optimize
                    rf_node["data"].update({
                        "label": node.label,
                        "nodeType": "PIPELINE",
                        "isExpanded": True,
                    })
                else:
                    rf_node["type"] = "custom"
                    rf_node["data"].update({
                        "label": node.label,
                        "nodeType": "PIPELINE",
                        "isExpanded": False,
                    })

            elif isinstance(node, DataNode):
                rf_node["type"] = "custom"
                rf_node["data"].update({
                    "label": node.name,
                    "nodeType": "DATA",
                    "typeHint": node.type_hint,
                    "isBound": node.is_bound,
                })
                
            elif isinstance(node, GroupDataNode):
                rf_node["type"] = "custom"
                rf_node["data"].update({
                    "label": "Inputs",
                    "nodeType": "INPUT_GROUP",
                    "params": [n.name for n in node.nodes],
                    "isBound": node.is_bound,
                })

            nodes.append(rf_node)

        for edge in viz_edges:
            rf_edge = {
                "id": f"e_{edge.source}_{edge.target}",
                "source": edge.source,
                "target": edge.target,
                "animated": False,
                "style": {"stroke": "#64748b", "strokeWidth": 2},
                "data": {},
            }
            
            if edge.label:
                rf_edge["data"]["label"] = edge.label
                
            edges.append(rf_edge)

        return {
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "theme_preference": theme,
                "initial_depth": initial_depth,
                "theme_debug": theme_debug,
                "pan_on_scroll": pan_on_scroll,
            },
        }
