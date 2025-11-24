from dataclasses import dataclass, field
from typing import Dict

@dataclass
class NodeStyle:
    """Style configuration for a specific node type."""
    bg_color: str
    border_color: str
    text_color: str
    accent_color: str  # Used for header background or icon
    accent_text_color: str = "#FFFFFF"
    
@dataclass
class GraphvizStyle:
    """Styling configuration for pipeline visualizations."""
    # Base settings
    font_name: str = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
    font_size: int = 14
    edge_font_size: int = 11
    background_color: str = "#F8FAFC"  # Very light slate/gray
    edge_color: str = "#64748b"  # Slate-500
    edge_penwidth: str = "1.5"
    arrow_size: str = "0.8"
    
    # Cluster settings
    cluster_border_color: str = "#cbd5e1"
    cluster_fill_color: str = "#FFFFFF"
    cluster_label_color: str = "#475569"
    
    # Node styles
    function_node: NodeStyle = field(default_factory=lambda: NodeStyle(
        bg_color="#FFFFFF", border_color="#6366f1", text_color="#1e293b", accent_color="#6366f1"  # Indigo
    ))
    pipeline_node: NodeStyle = field(default_factory=lambda: NodeStyle(
        bg_color="#FFFFFF", border_color="#f59e0b", text_color="#1e293b", accent_color="#f59e0b"  # Amber
    ))
    dual_node: NodeStyle = field(default_factory=lambda: NodeStyle(
        bg_color="#FFFFFF", border_color="#d946ef", text_color="#1e293b", accent_color="#d946ef"  # Fuchsia
    ))
    data_node: NodeStyle = field(default_factory=lambda: NodeStyle(
        bg_color="#FFFFFF", border_color="#64748b", text_color="#1e293b", accent_color="#64748b"  # Slate
    ))
    input_node: NodeStyle = field(default_factory=lambda: NodeStyle(
        bg_color="#FFFFFF", border_color="#64748b", text_color="#1e293b", accent_color="#64748b"  # Default to Slate
    ))
    group_node: NodeStyle = field(default_factory=lambda: NodeStyle(
        bg_color="#F8FAFC", border_color="#94a3b8", text_color="#0f172a", accent_color="#94a3b8"  # Grouped inputs
    ))
    
    # Legacy compatibility properties
    @property
    def func_node_color(self) -> str: return self.function_node.accent_color
    
    @property
    def dual_node_color(self) -> str: return self.dual_node.accent_color
    
    @property
    def pipeline_node_color(self) -> str: return self.pipeline_node.accent_color
    
    @property
    def data_node_color(self) -> str: return self.data_node.bg_color


# --- Themes ---

# Simple/Elegant Theme (Airflow-inspired)
SIMPLE_THEME = GraphvizStyle(
    background_color="var(--hn-surface-bg)",
    edge_color="var(--hn-edge)",
    cluster_fill_color="var(--hn-cluster-fill)",
    cluster_border_color="var(--hn-cluster-border)",
    cluster_label_color="var(--hn-cluster-text)",
    function_node=NodeStyle(
        bg_color="var(--hn-func-bg)",
        border_color="var(--hn-func-border)",
        text_color="var(--hn-func-text)",
        accent_color="var(--hn-func-border)", # Not used in simple mode
        accent_text_color="var(--hn-func-text)",
    ),
    pipeline_node=NodeStyle(
        bg_color="var(--hn-pipe-bg)",
        border_color="var(--hn-pipe-border)",
        text_color="var(--hn-pipe-text)",
        accent_color="var(--hn-pipe-border)",
        accent_text_color="var(--hn-pipe-text)",
    ),
    dual_node=NodeStyle(
        bg_color="var(--hn-dual-bg)",
        border_color="var(--hn-dual-border)",
        text_color="var(--hn-dual-text)",
        accent_color="var(--hn-dual-border)",
        accent_text_color="var(--hn-dual-text)",
    ),
    data_node=NodeStyle(
        bg_color="var(--hn-data-bg)",
        border_color="var(--hn-data-border)",
        text_color="var(--hn-data-text)",
        accent_color="var(--hn-data-border)",
        accent_text_color="var(--hn-data-text)",
    ),
    input_node=NodeStyle(
        bg_color="var(--hn-input-bg)",
        border_color="var(--hn-input-border)",
        text_color="var(--hn-input-text)",
        accent_color="var(--hn-input-border)",
        accent_text_color="var(--hn-input-text)",
    ),
    group_node=NodeStyle(
        bg_color="var(--hn-group-bg)",
        border_color="var(--hn-group-border)",
        text_color="var(--hn-group-text)",
        accent_color="var(--hn-group-border)",
        accent_text_color="var(--hn-group-text)",
    ),
)

AUTO_THEME = SIMPLE_THEME  # Make simple the default


LIGHT_THEME = GraphvizStyle(
    background_color="#ffffff",
    edge_color="#94a3b8",
    cluster_fill_color="#f8fafc",
    cluster_border_color="#e2e8f0",
    function_node=NodeStyle(
        bg_color="#dbeafe", border_color="#93c5fd", text_color="#1e3a8a", accent_color="#2563eb", accent_text_color="#ffffff"
    ),
    pipeline_node=NodeStyle(
        bg_color="#fef3c7", border_color="#fde68a", text_color="#78350f", accent_color="#d97706", accent_text_color="#ffffff"
    ),
    dual_node=NodeStyle(
        bg_color="#f3e8ff", border_color="#e9d5ff", text_color="#6b21a8", accent_color="#9333ea", accent_text_color="#ffffff"
    ),
    data_node=NodeStyle(
        bg_color="#f3f4f6", border_color="#d1d5db", text_color="#374151", accent_color="#4b5563", accent_text_color="#ffffff"
    ),
    input_node=NodeStyle(
        bg_color="#e0f2fe", border_color="#7dd3fc", text_color="#0c4a6e", accent_color="#0284c7", accent_text_color="#ffffff"
    ),
    group_node=NodeStyle(
        bg_color="#eef2ff", border_color="#c7d2fe", text_color="#312e81", accent_color="#818cf8", accent_text_color="#312e81"
    )
)

# Dark theme adapted for the professional look
DARK_THEME = GraphvizStyle(
    background_color="#18181b", # Zinc-950
    edge_color="#71717a", # Zinc-500
    cluster_fill_color="#27272a", # Zinc-800
    cluster_border_color="#3f3f46", # Zinc-700
    cluster_label_color="#a1a1aa", # Zinc-400
    function_node=NodeStyle(
        bg_color="#27272a", border_color="#3f3f46", text_color="#f4f4f5", accent_color="#a1a1aa", accent_text_color="#18181b"
    ),
    pipeline_node=NodeStyle(
        bg_color="#27272a", border_color="#d97706", text_color="#fef3c7", accent_color="#fbbf24", accent_text_color="#18181b"
    ),
    dual_node=NodeStyle(
        bg_color="#27272a", border_color="#c026d3", text_color="#fdf4ff", accent_color="#e879f9", accent_text_color="#18181b"
    ),
    data_node=NodeStyle(
        bg_color="#27272a", border_color="#52525b", text_color="#e4e4e7", accent_color="#71717a", accent_text_color="#18181b"
    ),
    input_node=NodeStyle(
        bg_color="#0b314c", border_color="#48c8ff", text_color="#e0f2fe", accent_color="#38bdf8", accent_text_color="#0b1021"
    ),
    group_node=NodeStyle(
        bg_color="#1f2937", border_color="#475569", text_color="#e5e7eb", accent_color="#475569", accent_text_color="#e5e7eb"
    )
)

# Kedro-inspired theme
KEDRO_THEME = GraphvizStyle(
    font_name="Titillium Web, sans-serif",
    background_color="#111111",
    edge_color="#6f6f6f",
    cluster_fill_color="#212121",
    cluster_border_color="#444444",
    cluster_label_color="#cccccc",
    function_node=NodeStyle(
        bg_color="#212121", border_color="#212121", text_color="#e0e0e0", accent_color="#e0e0e0", accent_text_color="#111111"
    ),
    pipeline_node=NodeStyle(
        bg_color="#212121", border_color="#f59e0b", text_color="#e0e0e0", accent_color="#f59e0b", accent_text_color="#111111"
    ),
    dual_node=NodeStyle(
        bg_color="#212121", border_color="#c026d3", text_color="#e0e0e0", accent_color="#e879f9", accent_text_color="#111111"
    ),
    data_node=NodeStyle(
        bg_color="#212121", border_color="#e0e0e0", text_color="#e0e0e0", accent_color="#e0e0e0", accent_text_color="#111111"
    )
)

# Legacy mapping
LEGACY_DEFAULT = GraphvizStyle(
    function_node=NodeStyle(bg_color="#FFFFFF", border_color="#87CEEB", text_color="#000000", accent_color="#87CEEB"),
    pipeline_node=NodeStyle(bg_color="#FFFFFF", border_color="#DDA0DD", text_color="#000000", accent_color="#DDA0DD"),
    dual_node=NodeStyle(bg_color="#FFFFFF", border_color="#FFA07A", text_color="#000000", accent_color="#FFA07A"),
    data_node=NodeStyle(bg_color="#90EE90", border_color="#90EE90", text_color="#000000", accent_color="#90EE90"),
)

DESIGN_STYLES: Dict[str, GraphvizStyle] = {
    "default": SIMPLE_THEME,
    "simple": SIMPLE_THEME,
    "kedro": KEDRO_THEME,
    "auto": SIMPLE_THEME,
    "light": LIGHT_THEME,
    "dark": DARK_THEME,
    "legacy": LEGACY_DEFAULT,
    "minimal": GraphvizStyle(
        function_node=NodeStyle(bg_color="#FFFFFF", border_color="#E8E8E8", text_color="#000000", accent_color="#FFFFFF", accent_text_color="#000000"),
        edge_color="#999999",
        cluster_fill_color="#FAFAFA",
    ),
}
