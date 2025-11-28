from dataclasses import dataclass, field
from typing import Dict, Optional

# Universal outline/edge color - visible in both light and dark modes
UNIVERSAL_OUTLINE = "#64748b"  # Tailwind slate-500 - balanced for both modes

# Edges for specific modes
EDGE_LIGHT = "#1e293b"  # Slate 800 - Almost black for light mode
EDGE_DARK = "#e2e8f0"  # Slate 200 - Contrast for dark mode


def calculate_outline(hex_color: str, factor: float = 0.7) -> str:
    """Calculates a darker outline color by multiplying RGB values by a factor (default 0.7)."""
    # Remove # if present
    c = hex_color.lstrip("#")

    # Handle short hex (e.g. #ABC -> #AABBCC)
    if len(c) == 3:
        c = "".join([x * 2 for x in c])

    if len(c) != 6:
        # Fallback for invalid hex
        return hex_color

    try:
        # Parse RGB
        r = int(c[0:2], 16)
        g = int(c[2:4], 16)
        b = int(c[4:6], 16)

        # Apply factor (darken)
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)

        # Clamp
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))

        return f"#{r:02x}{g:02x}{b:02x}"
    except ValueError:
        return hex_color


@dataclass(frozen=True)
class ColorPair:
    """Represents a color scheme for a node (fill, outline, text)."""

    fill: str
    outline: Optional[str] = None
    text: str = "#000000"  # Always black for max contrast

    def __post_init__(self):
        # Automatic outline calculation if not provided
        if self.outline is None:
            calculated = calculate_outline(self.fill)
            object.__setattr__(self, "outline", calculated)


class Palettes:
    """Standard color palettes (based on d3.category20)."""

    # Pairs are (Light Fill, Auto Outline)
    BLUE = ColorPair(fill="#aec7e8")
    ORANGE = ColorPair(fill="#ffbb78")
    GREEN = ColorPair(fill="#98df8a")
    RED = ColorPair(fill="#ff9896")
    PURPLE = ColorPair(fill="#c5b0d5")
    BROWN = ColorPair(fill="#c49c94")
    PINK = ColorPair(fill="#f7b6d2")
    GREY = ColorPair(fill="#c7c7c7")
    OLIVE = ColorPair(fill="#dbdb8d")
    CYAN = ColorPair(fill="#9edae5")


class PastelPalettes:
    """Legacy pastel color combinations (kept for backwards compatibility)."""

    # All outlines use auto calculation

    # 1. Original Pastel Bright (The favorite)
    ORIGINAL_BLUE = ColorPair(fill="#87CEEB")
    ORIGINAL_PLUM = ColorPair(fill="#DDA0DD")
    ORIGINAL_GREEN = ColorPair(fill="#98FB98")

    # 2. Cotton Candy
    CANDY_PINK = ColorPair(fill="#FFB7B2")
    CANDY_BLUE = ColorPair(fill="#A0C4FF")
    CANDY_YELLOW = ColorPair(fill="#FFFFB5")

    # 3. Minty Fresh
    MINT_MAIN = ColorPair(fill="#B5EAD7")
    MINT_SEC = ColorPair(fill="#C7CEEA")
    MINT_ACCENT = ColorPair(fill="#E2F0CB")

    # 4. Sunset Haze
    HAZE_PEACH = ColorPair(fill="#FFDAC1")
    HAZE_LILAC = ColorPair(fill="#E0BBE4")
    HAZE_GOLD = ColorPair(fill="#FDFD96")

    # 5. Ocean Breeze
    BREEZE_TEAL = ColorPair(fill="#9bf6ff")
    BREEZE_BLUE = ColorPair(fill="#a0c4ff")
    BREEZE_SAND = ColorPair(fill="#fff1e6")

    # 6. Lavender Fields
    LAV_MAIN = ColorPair(fill="#E6E6FA")
    LAV_SEC = ColorPair(fill="#D8BFD8")
    LAV_ACCENT = ColorPair(fill="#F0FFF0")

    # 7. Rose Garden
    ROSE_MAIN = ColorPair(fill="#ffc6ff")
    ROSE_SEC = ColorPair(fill="#bdb2ff")
    ROSE_ACCENT = ColorPair(fill="#caffbf")

    # 8. Peaches & Cream
    PEACH_MAIN = ColorPair(fill="#FFDAB9")
    PEACH_SEC = ColorPair(fill="#FFE4B5")
    PEACH_ACCENT = ColorPair(fill="#FFFACD")

    # 9. Baby Blues
    BABY_MAIN = ColorPair(fill="#B0E0E6")
    BABY_SEC = ColorPair(fill="#ADD8E6")
    BABY_ACCENT = ColorPair(fill="#E0FFFF")

    # 10. Spring Meadow
    MEADOW_MAIN = ColorPair(fill="#90EE90")
    MEADOW_SEC = ColorPair(fill="#87CEFA")
    MEADOW_ACCENT = ColorPair(fill="#FFFFE0")

    # 11. Coral Reef
    CORAL_MAIN = ColorPair(fill="#F08080")
    CORAL_SEC = ColorPair(fill="#20B2AA")
    CORAL_ACCENT = ColorPair(fill="#E0FFFF")

    # 12. Nordic Frost
    NORDIC_MAIN = ColorPair(fill="#D3D3D3")
    NORDIC_SEC = ColorPair(fill="#B0C4DE")
    NORDIC_ACCENT = ColorPair(fill="#F5F5F5")

    # 13. Lemonade
    LEMON_MAIN = ColorPair(fill="#FFFACD")
    LEMON_SEC = ColorPair(fill="#FF69B4")
    LEMON_ACCENT = ColorPair(fill="#E0FFFF")

    # 14. Vintage Paper
    VINTAGE_MAIN = ColorPair(fill="#FAEBD7")
    VINTAGE_SEC = ColorPair(fill="#D2B48C")
    VINTAGE_ACCENT = ColorPair(fill="#F5DEB3")

    # 15. Soft Berry
    BERRY_MAIN = ColorPair(fill="#D8BFD8")
    BERRY_SEC = ColorPair(fill="#DDA0DD")
    BERRY_ACCENT = ColorPair(fill="#FFE4E1")

    # 16. Glacier
    GLACIER_MAIN = ColorPair(fill="#AFEEEE")
    GLACIER_SEC = ColorPair(fill="#E0FFFF")
    GLACIER_ACCENT = ColorPair(fill="#F0F8FF")

    # 17. Sandstone
    SAND_MAIN = ColorPair(fill="#F4A460")
    SAND_SEC = ColorPair(fill="#DEB887")
    SAND_ACCENT = ColorPair(fill="#FFF8DC")

    # 18. Periwinkle
    PERI_MAIN = ColorPair(fill="#CCCCFF")
    PERI_SEC = ColorPair(fill="#B0C4DE")
    PERI_ACCENT = ColorPair(fill="#E6E6FA")

    # 19. Eucalyptus
    EUC_MAIN = ColorPair(fill="#8FBC8F")
    EUC_SEC = ColorPair(fill="#66CDAA")
    EUC_ACCENT = ColorPair(fill="#F5FFFA")

    # 20. Mauve
    MAUVE_MAIN = ColorPair(fill="#E0B0FF")
    MAUVE_SEC = ColorPair(fill="#D8BFD8")
    MAUVE_ACCENT = ColorPair(fill="#FFF0F5")


@dataclass
class NodeStyle:
    """Visual configuration for a specific node type."""

    shape: str = "box"
    style: str = "filled,rounded"
    is_bold: bool = False
    margin: str = "0.1"
    penwidth: str = "1.5"
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
            "penwidth": self.penwidth,
        }


@dataclass
class GraphvizTheme:
    """Complete visual theme for Graphviz rendering."""

    # Global settings
    font_name: str = "Helvetica"
    base_font_size: int = 12
    edge_font_size: int = 10
    edge_color: str = EDGE_LIGHT  # Default to dark grey for light mode

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
                "penwidth": "1.5",
                "color": self.edge_color,
                "arrowsize": "0.7",
                "arrowhead": "vee",
            }


def create_custom_theme(
    func_color: ColorPair,
    pipe_color: ColorPair,
    input_color: ColorPair,
    edge_color: str = EDGE_LIGHT,
) -> GraphvizTheme:
    """Factory for custom colored themes."""
    return GraphvizTheme(
        edge_color=edge_color,
        node_styles={
            "function": NodeStyle(is_bold=True, color=func_color),
            "pipeline": NodeStyle(is_bold=True, color=pipe_color),
            "dual": NodeStyle(is_bold=True, color=func_color),
            "data": NodeStyle(is_bold=False, color=input_color),
            "bound_data": NodeStyle(style="filled,rounded", color=input_color),
            "group": NodeStyle(is_bold=False, color=input_color),
            "bound_group": NodeStyle(style="filled,rounded", color=input_color),
        },
    )


class SelectedColors:
    """
    NEW Default: User's chosen pastel palette with custom outlines.
    """

    # Function nodes: peach orange
    FUNCTION = ColorPair(fill="#ffd8a8", outline="#f08c00")

    # Pipeline nodes: lavender purple
    PIPELINE = ColorPair(fill="#d0bfff", outline="#6741d9")

    # Input / data nodes: soft green
    INPUT = ColorPair(fill="#b2f2bb", outline="#2f9e44")


class LegacyColors:
    """
    Previous default colors (preserved for backwards compatibility).
    Use style="legacy" to switch back.
    """

    FUNCTION = ColorPair(fill="#87D3F9")  # Sky Blue
    PIPELINE = ColorPair(fill="#D5BCFE")  # Plum
    INPUT = ColorPair(fill="#FFC661")  # Amber


class NeonColors:
    """
    Super vibrant neon colors for true Cyberpunk aesthetics.
    """

    # 1. Solid Neon - Bright fills, high contrast
    SOLID_FUNC = ColorPair(
        fill="#08F7FE", outline="#FFFFFF", text="#000000"
    )  # Electric Cyan
    SOLID_PIPE = ColorPair(
        fill="#FE53BB", outline="#FFFFFF", text="#000000"
    )  # Neon Pink/Magenta
    SOLID_INPUT = ColorPair(
        fill="#39FF14", outline="#FFFFFF", text="#000000"
    )  # Neon Green

    # 2. Wireframe Neon - Dark fills, glowing outlines
    WIRE_FUNC = ColorPair(
        fill="#0a1a1c", outline="#08F7FE", text="#08F7FE"
    )  # Dark Cyan bg
    WIRE_PIPE = ColorPair(
        fill="#1a0a16", outline="#FE53BB", text="#FE53BB"
    )  # Dark Pink bg
    WIRE_INPUT = ColorPair(
        fill="#0a1a0a", outline="#39FF14", text="#39FF14"
    )  # Dark Green bg


class CyberpunkColors:
    """
    High-saturation/Neon colors for Dark Mode.
    """

    # 1. Neon - High Contrast
    NEON_FUNC = ColorPair(
        fill="#00e5ff", outline="#ffffff", text="#000000"
    )  # Cyan Neon
    NEON_PIPE = ColorPair(
        fill="#d500f9", outline="#ffffff", text="#ffffff"
    )  # Magenta Neon
    NEON_INPUT = ColorPair(
        fill="#00e676", outline="#ffffff", text="#000000"
    )  # Green Neon

    # 2. Soft Cyber - Lighter, glowing
    SOFT_FUNC = ColorPair(fill="#4dd0e1", outline="#b2ebf2", text="#000000")
    SOFT_PIPE = ColorPair(fill="#e040fb", outline="#f3e5f5", text="#ffffff")
    SOFT_INPUT = ColorPair(fill="#69f0ae", outline="#b9f6ca", text="#000000")

    # 3. Deep Cyber - Darker fills, bright outlines
    DEEP_FUNC = ColorPair(
        fill="#01579b", outline="#4fc3f7", text="#e1f5fe"
    )  # Dark Blue
    DEEP_PIPE = ColorPair(
        fill="#4a148c", outline="#ea80fc", text="#f3e5f5"
    )  # Dark Purple
    DEEP_INPUT = ColorPair(
        fill="#1b5e20", outline="#69f0ae", text="#e8f5e9"
    )  # Dark Green


class UserColorPalette:
    """
    User-provided colors for creating theme variations.
    These are the colors you specified for mixing/matching.
    """

    # Greens
    MINT_GREEN = ColorPair(fill="#B0EEB9")  # Your input color
    BRIGHT_MINT = ColorPair(fill="#b7f5ce")  # Lighter mint
    TAILWIND_GREEN = ColorPair(fill="#86efac")  # Tailwind green-300

    # Blues
    SKY_BLUE = ColorPair(fill="#A3D5FC")  # Your function color
    BRIGHT_CYAN = ColorPair(fill="#9ee0ff")  # Bright cyan
    TAILWIND_BLUE = ColorPair(fill="#7dd3fc")  # Tailwind sky-300

    # Purples/Pinks
    LAVENDER = ColorPair(fill="#EBBCF6")  # Your pipeline color
    SOFT_PURPLE = ColorPair(fill="#e7b3ff")  # Soft purple

    # Warm colors
    SOFT_YELLOW = ColorPair(fill="#FBE897")  # Soft yellow
    SOFT_CORAL = ColorPair(fill="#FBC7C7")  # Soft coral/orange
    TAILWIND_ORANGE = ColorPair(fill="#fdba74")  # Tailwind orange-300


class DagPalettes:
    """
    Curated DAG-friendly palettes from modern UI/data-viz themes.

    Each triple is (FUNCTION, PIPELINE, INPUT).
    """

    # 1. Nord Soft — muted professional
    NORD_FUNC = ColorPair(fill="#88C0D0")
    NORD_PIPE = ColorPair(fill="#A3BE8C")
    NORD_INPUT = ColorPair(fill="#E5E9F0")

    # 2. Soft Candy — blue/pink/lilac pastels
    CANDY_FUNC = ColorPair(fill="#BDE0FE")
    CANDY_PIPE = ColorPair(fill="#FFC8DD")
    CANDY_INPUT = ColorPair(fill="#CDB4DB")

    # 3. Warm Flow — sunset oranges & corals
    WARM_FUNC = ColorPair(fill="#F8AD9D")
    WARM_PIPE = ColorPair(fill="#FFDAB9")
    WARM_INPUT = ColorPair(fill="#FBC4AB")

    # 4. Tropical Ocean — teal + coral
    TROPIC_FUNC = ColorPair(fill="#A5FFD6")
    TROPIC_PIPE = ColorPair(fill="#84DCC6")
    TROPIC_INPUT = ColorPair(fill="#FFA69E")

    # 5. Sunrise — blue + apricot + lemon
    SUNRISE_FUNC = ColorPair(fill="#79ADDC")
    SUNRISE_PIPE = ColorPair(fill="#FFC09F")
    SUNRISE_INPUT = ColorPair(fill="#FFEE93")

    # 6. Legacy - matches the old default
    LEGACY_FUNC = LegacyColors.FUNCTION
    LEGACY_PIPE = LegacyColors.PIPELINE
    LEGACY_INPUT = LegacyColors.INPUT


def create_default_theme() -> GraphvizTheme:
    """Factory for the default HyperNodes theme (new pastel palette)."""
    return create_custom_theme(
        SelectedColors.FUNCTION,
        SelectedColors.PIPELINE,
        SelectedColors.INPUT,
        edge_color=EDGE_LIGHT,
    )


# Global registry
DEFAULT_THEME = create_default_theme()

# Theme Registry using curated DAG palettes
DESIGN_STYLES: Dict[str, GraphvizTheme] = {
    # ========== DEFAULT (Light Mode Optimized) ==========
    "default": DEFAULT_THEME,
    # ========== LEGACY (your previous default) ==========
    "legacy": create_custom_theme(
        LegacyColors.FUNCTION,
        LegacyColors.PIPELINE,
        LegacyColors.INPUT,
        edge_color=EDGE_LIGHT,
    ),
    # ========== DARK MODE / CYBERPUNK ==========
    "dark_neon_solid": create_custom_theme(
        NeonColors.SOLID_FUNC,
        NeonColors.SOLID_PIPE,
        NeonColors.SOLID_INPUT,
        edge_color="#FFFFFF",  # White edges for maximum contrast against dark
    ),
    "dark_neon_wire": create_custom_theme(
        NeonColors.WIRE_FUNC,
        NeonColors.WIRE_PIPE,
        NeonColors.WIRE_INPUT,
        edge_color=EDGE_DARK,
    ),
    "dark_neon": create_custom_theme(
        CyberpunkColors.NEON_FUNC,
        CyberpunkColors.NEON_PIPE,
        CyberpunkColors.NEON_INPUT,
        edge_color=EDGE_DARK,
    ),
    "dark_soft": create_custom_theme(
        CyberpunkColors.SOFT_FUNC,
        CyberpunkColors.SOFT_PIPE,
        CyberpunkColors.SOFT_INPUT,
        edge_color=EDGE_DARK,
    ),
    "dark_deep": create_custom_theme(
        CyberpunkColors.DEEP_FUNC,
        CyberpunkColors.DEEP_PIPE,
        CyberpunkColors.DEEP_INPUT,
        edge_color=EDGE_DARK,
    ),
    # ========== VARIATIONS WITH YOUR COLORS (Using Light Edge) ==========
    # Variation 1: Bright Mint + Cyan + Soft Purple
    "bright_mint": create_custom_theme(
        UserColorPalette.BRIGHT_CYAN,  # func: #9ee0ff
        UserColorPalette.SOFT_PURPLE,  # pipe: #e7b3ff
        UserColorPalette.BRIGHT_MINT,  # input: #b7f5ce
    ),
    # Variation 2: Yellow + Green + Purple (warm accents)
    "sunny": create_custom_theme(
        UserColorPalette.MINT_GREEN,  # func: #B0EEB9
        UserColorPalette.LAVENDER,  # pipe: #EBBCF6
        UserColorPalette.SOFT_YELLOW,  # input: #FBE897
    ),
    # Variation 3: Coral + Orange + Blue
    "coral_sunset": create_custom_theme(
        UserColorPalette.SKY_BLUE,  # func: #A3D5FC
        UserColorPalette.SOFT_CORAL,  # pipe: #FBC7C7
        UserColorPalette.TAILWIND_ORANGE,  # input: #fdba74
    ),
    # Variation 4: Tailwind pastels (green, blue, orange)
    "tailwind_pastel": create_custom_theme(
        UserColorPalette.TAILWIND_BLUE,  # func: #7dd3fc
        UserColorPalette.TAILWIND_ORANGE,  # pipe: #fdba74
        UserColorPalette.TAILWIND_GREEN,  # input: #86efac
    ),
    # Variation 5: All cool tones (cyan + blue + mint)
    "cool_breeze": create_custom_theme(
        UserColorPalette.BRIGHT_CYAN,  # func: #9ee0ff
        UserColorPalette.TAILWIND_BLUE,  # pipe: #7dd3fc
        UserColorPalette.BRIGHT_MINT,  # input: #b7f5ce
    ),
    # Variation 6: Warm tones (yellow + coral + orange)
    "warm_glow": create_custom_theme(
        UserColorPalette.SOFT_YELLOW,  # func: #FBE897
        UserColorPalette.SOFT_CORAL,  # pipe: #FBC7C7
        UserColorPalette.TAILWIND_ORANGE,  # input: #fdba74
    ),
    # Variation 7: Purple dominant (soft purple + lavender + tailwind green)
    "lavender_fields": create_custom_theme(
        UserColorPalette.TAILWIND_GREEN,  # func: #86efac
        UserColorPalette.SOFT_PURPLE,  # pipe: #e7b3ff
        UserColorPalette.LAVENDER,  # input: #EBBCF6
    ),
    # Variation 8: High contrast pastel
    "vivid_pastel": create_custom_theme(
        UserColorPalette.TAILWIND_BLUE,  # func: #7dd3fc
        UserColorPalette.LAVENDER,  # pipe: #EBBCF6
        UserColorPalette.SOFT_YELLOW,  # input: #FBE897
    ),
    # ========== ORIGINAL CURATED PALETTES ==========
    # Nord Soft – muted, serious, great on light or slightly tinted backgrounds
    "nord_soft": create_custom_theme(
        DagPalettes.NORD_FUNC,
        DagPalettes.NORD_PIPE,
        DagPalettes.NORD_INPUT,
    ),
    # Soft Candy – playful, good for exploratory / dev DAGs
    "soft_candy": create_custom_theme(
        DagPalettes.CANDY_FUNC,
        DagPalettes.CANDY_PIPE,
        DagPalettes.CANDY_INPUT,
    ),
    # Warm Flow – warm, high-energy pipelines (alerts, experiments, etc.)
    "warm_flow": create_custom_theme(
        DagPalettes.WARM_FUNC,
        DagPalettes.WARM_PIPE,
        DagPalettes.WARM_INPUT,
    ),
    # Tropical Ocean – teal/coral contrast for visually busy graphs
    "tropical_ocean": create_custom_theme(
        DagPalettes.TROPIC_FUNC,
        DagPalettes.TROPIC_PIPE,
        DagPalettes.TROPIC_INPUT,
    ),
    # Sunrise – optimistic, good for presentations
    "sunrise": create_custom_theme(
        DagPalettes.SUNRISE_FUNC,
        DagPalettes.SUNRISE_PIPE,
        DagPalettes.SUNRISE_INPUT,
    ),
}
