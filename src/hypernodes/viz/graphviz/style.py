from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class ColorPair:
    """Represents a color scheme for a node (fill, outline, text)."""

    fill: str
    outline: str
    text: str = "#000000"


class Palettes:
    """Standard color palettes (based on d3.category20)."""

    # Pairs are (Light Fill, Dark Outline)
    BLUE = ColorPair(fill="#aec7e8", outline="#1f77b4")
    ORANGE = ColorPair(fill="#ffbb78", outline="#ff7f0e")
    GREEN = ColorPair(fill="#98df8a", outline="#2ca02c")
    RED = ColorPair(fill="#ff9896", outline="#d62728")
    PURPLE = ColorPair(fill="#c5b0d5", outline="#9467bd")
    BROWN = ColorPair(fill="#c49c94", outline="#8c564b")
    PINK = ColorPair(fill="#f7b6d2", outline="#e377c2")
    GREY = ColorPair(fill="#c7c7c7", outline="#7f7f7f")
    OLIVE = ColorPair(fill="#dbdb8d", outline="#bcbd22")
    CYAN = ColorPair(fill="#9edae5", outline="#17becf")


@dataclass
class NodeStyle:
    """Visual configuration for a specific node type."""

    shape: str = "box"
    style: str = "filled,rounded"
    is_bold: bool = False
    margin: str = "0.1"
    color: ColorPair = field(default_factory=lambda: Palettes.GREY)

    @property
    def graphviz_attrs(self) -> Dict[str, str]:
        """Convert to Graphviz node attributes."""
        return {
            "shape": self.shape,
            "style": self.style,
            "fillcolor": self.color.fill,
            "color": self.color.outline,
            "fontcolor": self.color.text,
            "margin": self.margin,
        }


@dataclass
class GraphvizTheme:
    """Complete visual theme for Graphviz rendering."""

    # Global settings
    font_name: str = "Helvetica"
    base_font_size: int = 12
    edge_font_size: int = 10
    edge_color: str = "#555555"

    # Graphviz Attributes
    graph_attr: Dict[str, str] = field(default_factory=dict)
    node_attr_defaults: Dict[str, str] = field(default_factory=dict)
    edge_attr: Dict[str, str] = field(default_factory=dict)

    # Node Type Specific Styles
    node_styles: Dict[str, NodeStyle] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize defaults if not provided
        self._init_graph_attrs()
        self._init_node_defaults()
        self._init_edge_attrs()

    def _init_graph_attrs(self):
        if not self.graph_attr:
            self.graph_attr = {
                "ranksep": "0.7",
                "nodesep": "0.5",
                "bgcolor": "transparent",
            }

    def _init_node_defaults(self):
        if not self.node_attr_defaults:
            self.node_attr_defaults = {
                "shape": "plain",
                "style": "filled,rounded",
                "fontname": self.font_name,
                "fontsize": str(self.base_font_size),
                "margin": "0.05",
            }

    def _init_edge_attrs(self):
        if not self.edge_attr:
            self.edge_attr = {
                "fontname": self.font_name,
                "fontsize": str(self.edge_font_size),
                "penwidth": "1.2",
                "color": self.edge_color,
                "arrowsize": "0.7",
                "arrowhead": "vee",
            }


def create_default_theme() -> GraphvizTheme:
    """Factory for the default HyperNodes theme."""
    return GraphvizTheme(
        node_styles={
            "function": NodeStyle(is_bold=True, color=Palettes.BLUE),
            "pipeline": NodeStyle(is_bold=True, color=Palettes.PURPLE),
            "dual": NodeStyle(is_bold=True, color=Palettes.BLUE),
            "data": NodeStyle(is_bold=False, color=Palettes.GREEN),
            "bound_data": NodeStyle(
                style="filled,rounded,dashed", color=Palettes.GREEN
            ),
            "group": NodeStyle(is_bold=False, color=Palettes.GREEN),
            "bound_group": NodeStyle(
                style="filled,rounded,dashed", color=Palettes.GREEN
            ),
        }
    )


# Global registry
DEFAULT_THEME = create_default_theme()
DESIGN_STYLES = {"default": DEFAULT_THEME}
