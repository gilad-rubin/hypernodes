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
    # All outlines standardized to black
    BLUE = ColorPair(fill="#aec7e8", outline="#000000")
    ORANGE = ColorPair(fill="#ffbb78", outline="#000000")
    GREEN = ColorPair(fill="#98df8a", outline="#000000")
    RED = ColorPair(fill="#ff9896", outline="#000000")
    PURPLE = ColorPair(fill="#c5b0d5", outline="#000000")
    BROWN = ColorPair(fill="#c49c94", outline="#000000")
    PINK = ColorPair(fill="#f7b6d2", outline="#000000")
    GREY = ColorPair(fill="#c7c7c7", outline="#000000")
    OLIVE = ColorPair(fill="#dbdb8d", outline="#000000")
    CYAN = ColorPair(fill="#9edae5", outline="#000000")


class PastelPalettes:
    """Legacy pastel color combinations (kept for backwards compatibility)."""

    # All outlines standardized to black

    # 1. Original Pastel Bright (The favorite)
    ORIGINAL_BLUE = ColorPair(fill="#87CEEB", outline="#000000")
    ORIGINAL_PLUM = ColorPair(fill="#DDA0DD", outline="#000000")
    ORIGINAL_GREEN = ColorPair(fill="#98FB98", outline="#000000")

    # 2. Cotton Candy
    CANDY_PINK = ColorPair(fill="#FFB7B2", outline="#000000")
    CANDY_BLUE = ColorPair(fill="#A0C4FF", outline="#000000")
    CANDY_YELLOW = ColorPair(fill="#FFFFB5", outline="#000000")

    # 3. Minty Fresh
    MINT_MAIN = ColorPair(fill="#B5EAD7", outline="#000000")
    MINT_SEC = ColorPair(fill="#C7CEEA", outline="#000000")
    MINT_ACCENT = ColorPair(fill="#E2F0CB", outline="#000000")

    # 4. Sunset Haze
    HAZE_PEACH = ColorPair(fill="#FFDAC1", outline="#000000")
    HAZE_LILAC = ColorPair(fill="#E0BBE4", outline="#000000")
    HAZE_GOLD = ColorPair(fill="#FDFD96", outline="#000000")

    # 5. Ocean Breeze
    BREEZE_TEAL = ColorPair(fill="#9bf6ff", outline="#000000")
    BREEZE_BLUE = ColorPair(fill="#a0c4ff", outline="#000000")
    BREEZE_SAND = ColorPair(fill="#fff1e6", outline="#000000")

    # 6. Lavender Fields
    LAV_MAIN = ColorPair(fill="#E6E6FA", outline="#000000")
    LAV_SEC = ColorPair(fill="#D8BFD8", outline="#000000")
    LAV_ACCENT = ColorPair(fill="#F0FFF0", outline="#000000")

    # 7. Rose Garden
    ROSE_MAIN = ColorPair(fill="#ffc6ff", outline="#000000")
    ROSE_SEC = ColorPair(fill="#bdb2ff", outline="#000000")
    ROSE_ACCENT = ColorPair(fill="#caffbf", outline="#000000")

    # 8. Peaches & Cream
    PEACH_MAIN = ColorPair(fill="#FFDAB9", outline="#000000")
    PEACH_SEC = ColorPair(fill="#FFE4B5", outline="#000000")
    PEACH_ACCENT = ColorPair(fill="#FFFACD", outline="#000000")

    # 9. Baby Blues
    BABY_MAIN = ColorPair(fill="#B0E0E6", outline="#000000")
    BABY_SEC = ColorPair(fill="#ADD8E6", outline="#000000")
    BABY_ACCENT = ColorPair(fill="#E0FFFF", outline="#000000")

    # 10. Spring Meadow
    MEADOW_MAIN = ColorPair(fill="#90EE90", outline="#000000")
    MEADOW_SEC = ColorPair(fill="#87CEFA", outline="#000000")
    MEADOW_ACCENT = ColorPair(fill="#FFFFE0", outline="#000000")

    # 11. Coral Reef
    CORAL_MAIN = ColorPair(fill="#F08080", outline="#000000")
    CORAL_SEC = ColorPair(fill="#20B2AA", outline="#000000")
    CORAL_ACCENT = ColorPair(fill="#E0FFFF", outline="#000000")

    # 12. Nordic Frost
    NORDIC_MAIN = ColorPair(fill="#D3D3D3", outline="#000000")
    NORDIC_SEC = ColorPair(fill="#B0C4DE", outline="#000000")
    NORDIC_ACCENT = ColorPair(fill="#F5F5F5", outline="#000000")

    # 13. Lemonade
    LEMON_MAIN = ColorPair(fill="#FFFACD", outline="#000000")
    LEMON_SEC = ColorPair(fill="#FF69B4", outline="#000000")
    LEMON_ACCENT = ColorPair(fill="#E0FFFF", outline="#000000")

    # 14. Vintage Paper
    VINTAGE_MAIN = ColorPair(fill="#FAEBD7", outline="#000000")
    VINTAGE_SEC = ColorPair(fill="#D2B48C", outline="#000000")
    VINTAGE_ACCENT = ColorPair(fill="#F5DEB3", outline="#000000")

    # 15. Soft Berry
    BERRY_MAIN = ColorPair(fill="#D8BFD8", outline="#000000")
    BERRY_SEC = ColorPair(fill="#DDA0DD", outline="#000000")
    BERRY_ACCENT = ColorPair(fill="#FFE4E1", outline="#000000")

    # 16. Glacier
    GLACIER_MAIN = ColorPair(fill="#AFEEEE", outline="#000000")
    GLACIER_SEC = ColorPair(fill="#E0FFFF", outline="#000000")
    GLACIER_ACCENT = ColorPair(fill="#F0F8FF", outline="#000000")

    # 17. Sandstone
    SAND_MAIN = ColorPair(fill="#F4A460", outline="#000000")
    SAND_SEC = ColorPair(fill="#DEB887", outline="#000000")
    SAND_ACCENT = ColorPair(fill="#FFF8DC", outline="#000000")

    # 18. Periwinkle
    PERI_MAIN = ColorPair(fill="#CCCCFF", outline="#000000")
    PERI_SEC = ColorPair(fill="#B0C4DE", outline="#000000")
    PERI_ACCENT = ColorPair(fill="#E6E6FA", outline="#000000")

    # 19. Eucalyptus
    EUC_MAIN = ColorPair(fill="#8FBC8F", outline="#000000")
    EUC_SEC = ColorPair(fill="#66CDAA", outline="#000000")
    EUC_ACCENT = ColorPair(fill="#F5FFFA", outline="#000000")

    # 20. Mauve
    MAUVE_MAIN = ColorPair(fill="#E0B0FF", outline="#000000")
    MAUVE_SEC = ColorPair(fill="#D8BFD8", outline="#000000")
    MAUVE_ACCENT = ColorPair(fill="#FFF0F5", outline="#000000")


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


def create_custom_theme(
    func_color: ColorPair, pipe_color: ColorPair, input_color: ColorPair
) -> GraphvizTheme:
    """Factory for custom colored themes."""
    return GraphvizTheme(
        node_styles={
            "function": NodeStyle(is_bold=True, color=func_color),
            "pipeline": NodeStyle(is_bold=True, color=pipe_color),
            "dual": NodeStyle(is_bold=True, color=func_color),
            "data": NodeStyle(is_bold=False, color=input_color),
            "bound_data": NodeStyle(style="filled,rounded,dashed", color=input_color),
            "group": NodeStyle(is_bold=False, color=input_color),
            "bound_group": NodeStyle(style="filled,rounded,dashed", color=input_color),
        }
    )


class SelectedColors:
    """User selected custom colors (soft Tailwind-inspired)."""

    # All outlines standardized to black (#000000)

    # Function nodes: soft blue
    FUNCTION = ColorPair(
        fill="#87D3F9",  # Sky Blue
        outline="#000000",
    )

    # Pipeline nodes: soft violet
    PIPELINE = ColorPair(
        fill="#D5BCFE",  # Plum
        outline="#000000",
    )

    # Input / data nodes: soft amber
    INPUT = ColorPair(
        fill="#FFC661",  # Mint Green
        outline="#000000",
    )


class DagPalettes:
    """
    Curated DAG-friendly palettes from modern UI/data-viz themes.

    Each triple is (FUNCTION, PIPELINE, INPUT).
    """

    # 1. Nord Soft — muted professional
    NORD_FUNC = ColorPair(fill="#88C0D0", outline="#000000")
    NORD_PIPE = ColorPair(fill="#A3BE8C", outline="#000000")
    NORD_INPUT = ColorPair(fill="#E5E9F0", outline="#000000")

    # 2. Soft Candy — blue/pink/lilac pastels
    CANDY_FUNC = ColorPair(fill="#BDE0FE", outline="#000000")
    CANDY_PIPE = ColorPair(fill="#FFC8DD", outline="#000000")
    CANDY_INPUT = ColorPair(fill="#CDB4DB", outline="#000000")

    # 3. Warm Flow — sunset oranges & corals
    WARM_FUNC = ColorPair(fill="#F8AD9D", outline="#000000")
    WARM_PIPE = ColorPair(fill="#FFDAB9", outline="#000000")
    WARM_INPUT = ColorPair(fill="#FBC4AB", outline="#000000")

    # 4. Tropical Ocean — teal + coral
    TROPIC_FUNC = ColorPair(fill="#A5FFD6", outline="#000000")
    TROPIC_PIPE = ColorPair(fill="#84DCC6", outline="#000000")
    TROPIC_INPUT = ColorPair(fill="#FFA69E", outline="#000000")

    # 5. Sunrise — blue + apricot + lemon
    SUNRISE_FUNC = ColorPair(fill="#79ADDC", outline="#000000")
    SUNRISE_PIPE = ColorPair(fill="#FFC09F", outline="#000000")
    SUNRISE_INPUT = ColorPair(fill="#FFEE93", outline="#000000")

    # 6. Tailwind Soft — matches SelectedColors (for consistency)
    TW_FUNC = SelectedColors.FUNCTION
    TW_PIPE = SelectedColors.PIPELINE
    TW_INPUT = SelectedColors.INPUT


def create_default_theme() -> GraphvizTheme:
    """Factory for the default HyperNodes theme (Tailwind-soft)."""
    return create_custom_theme(
        SelectedColors.FUNCTION, SelectedColors.PIPELINE, SelectedColors.INPUT
    )


# Global registry
DEFAULT_THEME = create_default_theme()

# Theme Registry using curated DAG palettes
DESIGN_STYLES: Dict[str, GraphvizTheme] = {
    # Default uses the Tailwind-soft SelectedColors
    "default": DEFAULT_THEME,
    # 1. Tailwind Soft – light SaaS/dashboard vibe
    "tailwind_soft": create_custom_theme(
        DagPalettes.TW_FUNC,
        DagPalettes.TW_PIPE,
        DagPalettes.TW_INPUT,
    ),
    # 2. Nord Soft – muted, serious, great on light or slightly tinted backgrounds
    "nord_soft": create_custom_theme(
        DagPalettes.NORD_FUNC,
        DagPalettes.NORD_PIPE,
        DagPalettes.NORD_INPUT,
    ),
    # 3. Soft Candy – playful, good for exploratory / dev DAGs
    "soft_candy": create_custom_theme(
        DagPalettes.CANDY_FUNC,
        DagPalettes.CANDY_PIPE,
        DagPalettes.CANDY_INPUT,
    ),
    # 4. Warm Flow – warm, high-energy pipelines (alerts, experiments, etc.)
    "warm_flow": create_custom_theme(
        DagPalettes.WARM_FUNC,
        DagPalettes.WARM_PIPE,
        DagPalettes.WARM_INPUT,
    ),
    # 5. Tropical Ocean – teal/coral contrast for visually busy graphs
    "tropical_ocean": create_custom_theme(
        DagPalettes.TROPIC_FUNC,
        DagPalettes.TROPIC_PIPE,
        DagPalettes.TROPIC_INPUT,
    ),
    # 6. Sunrise – optimistic, good for presentations
    "sunrise": create_custom_theme(
        DagPalettes.SUNRISE_FUNC,
        DagPalettes.SUNRISE_PIPE,
        DagPalettes.SUNRISE_INPUT,
    ),
}
