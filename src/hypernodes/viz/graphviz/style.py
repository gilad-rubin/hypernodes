from dataclasses import dataclass, field
from typing import Dict


@dataclass
class NodeConfig:
    """Configuration for a specific node type."""

    shape: str = "box"
    style: str = "filled,rounded"
    is_bold: bool = False
    margin: str = "0.1"
    fill_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    text_color: str = "#000000"


@dataclass
class GraphvizStyle:
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
    "#9edae5",  # 18,19: Cyan
]

# Default node configurations
DEFAULT_NODE_CONFIGS = {
    "function": NodeConfig(
        shape="box",
        style="filled,rounded",
        is_bold=True,
        fill_color=CATEGORY20[1],  # Light Blue
        outline_color=CATEGORY20[0],  # Dark Blue
        text_color="black",
    ),
    "pipeline": NodeConfig(
        shape="box",
        style="filled,rounded",
        is_bold=True,
        fill_color=CATEGORY20[9],  # Light Purple
        outline_color=CATEGORY20[8],  # Dark Purple
        text_color="black",
    ),
    "dual": NodeConfig(
        shape="box",
        style="filled,rounded",
        is_bold=True,
        fill_color=CATEGORY20[1],  # Light Blue (Same as function for now)
        outline_color=CATEGORY20[0],  # Dark Blue
        text_color="black",
    ),
    "data": NodeConfig(
        shape="box",
        style="filled,rounded",
        is_bold=False,
        fill_color=CATEGORY20[5],  # Light Green
        outline_color=CATEGORY20[4],  # Dark Green
        text_color="black",
    ),
    "bound_data": NodeConfig(
        shape="box",
        style="filled,rounded,dashed",
        is_bold=False,
        fill_color=CATEGORY20[7],  # Light Red
        outline_color=CATEGORY20[6],  # Dark Red
        text_color="black",
    ),
    "group": NodeConfig(
        shape="box",
        style="filled,rounded",
        is_bold=False,
        fill_color=CATEGORY20[5],  # Light Green
        outline_color=CATEGORY20[4],  # Dark Green
        text_color="black",
    ),
    "bound_group": NodeConfig(
        shape="box",
        style="filled,rounded,dashed",
        is_bold=False,
        fill_color=CATEGORY20[7],  # Light Red
        outline_color=CATEGORY20[6],  # Dark Red
        text_color="black",
    ),
}

# Mapping to semantic roles using Light variants for background, Dark for outline
DEFAULT_STYLE = GraphvizStyle(
    edge_color="#555555",
    font_name="Helvetica",
    font_size=12,
    edge_font_size=10,
    graph_attr={
        # "rankdir": "TB",
        "ranksep": "0.8",  # More vertical space
        "nodesep": "0.5",  # More horizontal space
        "bgcolor": "transparent",  # Restore transparent background
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
        "penwidth": "1.2",  # Match previous penwidth
        "color": "#555555",
        "arrowsize": "0.7",  # Match previous arrowsize
        "arrowhead": "vee",  # Match previous arrowhead
    },
    node_configs=DEFAULT_NODE_CONFIGS,
)

DESIGN_STYLES = {"default": DEFAULT_STYLE}
