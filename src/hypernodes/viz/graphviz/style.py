from dataclasses import dataclass, field
from typing import Dict


@dataclass
class NodeConfig:
    """Configuration for a specific node type."""

    shape: str = "box"
    style: str = "filled,rounded"
    is_bold: bool = False
    margin: str = "0.1"


@dataclass
class GraphvizStyle:
    # Colors
    arg_node_color: str
    arg_outline_color: str
    arg_text_color: str
    func_node_color: str
    func_outline_color: str
    func_text_color: str
    nested_func_node_color: str
    nested_func_outline_color: str
    nested_func_text_color: str
    bound_node_color: str
    bound_outline_color: str
    bound_text_color: str
    group_node_color: str
    group_outline_color: str
    group_text_color: str

    # Edges
    edge_color: str

    # Font
    font_name: str
    font_size: int
    edge_font_size: int

    # Graph
    graph_attr: Dict[str, str]
    node_attr: Dict[str, str]
    edge_attr: Dict[str, str]

    # Node Type Mappings
    node_configs: Dict[str, NodeConfig] = field(default_factory=dict)


# Category20 colors - corrected palette
# Pairs of (Dark, Light)
CATEGORY20 = [
    "#1f77b4",
    "#aec7e8",  # 0,1: Blue
    "#ff7f0e",
    "#ffbb78",  # 2,3: Orange
    "#2ca02c",
    "#98df8a",  # 4,5: Green
    "#d62728",
    "#ff9896",  # 6,7: Red
    "#9467bd",
    "#c5b0d5",  # 8,9: Purple
    "#8c564b",
    "#c49c94",  # 10,11: Brown
    "#e377c2",
    "#f7b6d2",  # 12,13: Pink
    "#7f7f7f",
    "#c7c7c7",  # 14,15: Grey
    "#bcbd22",
    "#dbdb8d",  # 16,17: Olive
    "#17becf",
    "#dbdb8d",  # 18,19: Cyan
]

# Default node configurations
DEFAULT_NODE_CONFIGS = {
    "function": NodeConfig(shape="box", style="filled,rounded", is_bold=True),
    "pipeline": NodeConfig(shape="box", style="filled,rounded", is_bold=True),
    "dual": NodeConfig(shape="box", style="filled,rounded", is_bold=True),
    "data": NodeConfig(shape="box", style="filled,rounded", is_bold=False),
    "bound_data": NodeConfig(shape="box", style="filled,rounded,dashed", is_bold=False),
    "group": NodeConfig(shape="box", style="filled,rounded", is_bold=False),
    "bound_group": NodeConfig(
        shape="box", style="filled,rounded,dashed", is_bold=False
    ),
}

# Mapping to semantic roles using Light variants for background, Dark for outline
DEFAULT_STYLE = GraphvizStyle(
    arg_node_color=CATEGORY20[5],  # Light Green
    arg_outline_color=CATEGORY20[4],  # Dark Green
    arg_text_color="black",
    func_node_color=CATEGORY20[1],  # Light Blue
    func_outline_color=CATEGORY20[0],  # Dark Blue
    func_text_color="black",
    nested_func_node_color=CATEGORY20[9],  # Light Purple
    nested_func_outline_color=CATEGORY20[8],  # Dark Purple
    nested_func_text_color="black",
    bound_node_color=CATEGORY20[7],  # Light Red
    bound_outline_color=CATEGORY20[6],  # Dark Red
    bound_text_color="black",
    group_node_color=CATEGORY20[5],  # Light Green (same as arg)
    group_outline_color=CATEGORY20[4],  # Dark Green
    group_text_color="black",
    edge_color="#555555",
    font_name="Helvetica",
    font_size=12,
    edge_font_size=10,
    graph_attr={
        "rankdir": "TB",
        "ranksep": "1.3",  # More vertical space
        "nodesep": "1.05",  # More horizontal space
        # "splines": "true",  # Smoother lines
        "pad": "0.55",
        "overlap": "false",
    },
    node_attr={
        "shape": "plain",
        "style": "filled,rounded",
        "fontname": "Helvetica",
        "fontsize": "12",
        "margin": "0.05",
    },
    edge_attr={
        "fontname": "Helvetica",
        "fontsize": "10",
        "penwidth": "1.01",
        "color": "#555555",
        "arrowsize": "0.8",
    },
    node_configs=DEFAULT_NODE_CONFIGS,
)

DESIGN_STYLES = {"default": DEFAULT_STYLE}
