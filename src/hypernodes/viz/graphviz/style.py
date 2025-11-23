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

AUTO_THEME = GraphvizStyle(
    background_color="var(--hn-surface-bg)",
    edge_color="var(--hn-edge)",
    cluster_fill_color="var(--hn-cluster-fill)",
    cluster_border_color="var(--hn-cluster-border)",
    cluster_label_color="var(--hn-cluster-text)",
    function_node=NodeStyle(
        bg_color="var(--hn-node-bg)",
        border_color="var(--hn-node-border)",
        text_color="var(--hn-node-text)",
        accent_color="var(--hn-func-accent)",
        accent_text_color="var(--hn-func-text)",
    ),
    pipeline_node=NodeStyle(
        bg_color="var(--hn-pipe-bg)",
        border_color="var(--hn-pipe-border)",
        text_color="var(--hn-pipe-text)",
        accent_color="var(--hn-pipe-accent)",
        accent_text_color="var(--hn-pipe-accent-text)",
    ),
    dual_node=NodeStyle(
        bg_color="var(--hn-dual-bg)",
        border_color="var(--hn-dual-border)",
        text_color="var(--hn-dual-text)",
        accent_color="var(--hn-dual-accent)",
        accent_text_color="var(--hn-dual-accent-text)",
    ),
    data_node=NodeStyle(
        bg_color="var(--hn-data-bg)",
        border_color="var(--hn-data-border)",
        text_color="var(--hn-data-text)",
        accent_color="var(--hn-data-accent)",
        accent_text_color="var(--hn-data-text)",
    ),
)

LIGHT_THEME = GraphvizStyle(
    background_color="#ffffff",
    edge_color="#94a3b8",
    cluster_fill_color="#f8fafc",
    cluster_border_color="#e2e8f0",
    function_node=NodeStyle(
        bg_color="#eef2ff", border_color="#6366f1", text_color="#0f172a", accent_color="#6366f1", accent_text_color="#ffffff"
    ),
    pipeline_node=NodeStyle(
        bg_color="#fffbeb", border_color="#f59e0b", text_color="#0f172a", accent_color="#f59e0b", accent_text_color="#ffffff"
    ),
    dual_node=NodeStyle(
        bg_color="#fdf4ff", border_color="#d946ef", text_color="#0f172a", accent_color="#d946ef", accent_text_color="#ffffff"
    ),
    data_node=NodeStyle(
        bg_color="#f8fafc", border_color="#64748b", text_color="#334155", accent_color="#64748b", accent_text_color="#ffffff"
    )
)

# Dark theme adapted for the professional look
DARK_THEME = GraphvizStyle(
    background_color="#0B1120", # Deep dark blue/slate
    edge_color="#475569",
    cluster_fill_color="#0f172a",
    cluster_border_color="#1e293b",
    cluster_label_color="#94a3b8",
    function_node=NodeStyle(
        bg_color="#1e1b4b", border_color="#818cf8", text_color="#f8fafc", accent_color="#818cf8", accent_text_color="#ffffff"
    ),
    pipeline_node=NodeStyle(
        bg_color="#451a03", border_color="#fbbf24", text_color="#f8fafc", accent_color="#fbbf24", accent_text_color="#ffffff"
    ),
    dual_node=NodeStyle(
        bg_color="#4a044e", border_color="#e879f9", text_color="#f8fafc", accent_color="#e879f9", accent_text_color="#ffffff"
    ),
    data_node=NodeStyle(
        bg_color="#0f172a", border_color="#94a3b8", text_color="#cbd5e1", accent_color="#94a3b8", accent_text_color="#e2e8f0"
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
    "default": AUTO_THEME,  # Modern default (auto-detect)
    "auto": AUTO_THEME,
    "light": LIGHT_THEME,
    "dark": DARK_THEME,
    "legacy": LEGACY_DEFAULT,
    "minimal": GraphvizStyle(
        function_node=NodeStyle(bg_color="#FFFFFF", border_color="#E8E8E8", text_color="#000000", accent_color="#FFFFFF", accent_text_color="#000000"),
        edge_color="#999999",
        cluster_fill_color="#FAFAFA",
    ),
}
