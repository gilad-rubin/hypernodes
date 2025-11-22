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
    font_name: str = "Inter, Helvetica, Arial, sans-serif"
    font_size: int = 12
    edge_font_size: int = 10
    background_color: str = "#F8FAFC" # Very light slate/gray
    edge_color: str = "#64748b"  # Slate-500
    edge_penwidth: str = "1.2"
    arrow_size: str = "0.7"
    
    # Cluster settings
    cluster_border_color: str = "#cbd5e1"
    cluster_fill_color: str = "#FFFFFF"
    cluster_label_color: str = "#475569"
    
    # Node styles
    function_node: NodeStyle = field(default_factory=lambda: NodeStyle(
        bg_color="#FFFFFF", border_color="#cbd5e1", text_color="#0f172a", accent_color="#4f46e5" # Indigo-600
    ))
    pipeline_node: NodeStyle = field(default_factory=lambda: NodeStyle(
        bg_color="#fffbeb", border_color="#fcd34d", text_color="#451a03", accent_color="#d97706" # Amber-600
    ))
    dual_node: NodeStyle = field(default_factory=lambda: NodeStyle(
        bg_color="#fdf4ff", border_color="#f0abfc", text_color="#4a044e", accent_color="#c026d3" # Fuchsia-600
    ))
    data_node: NodeStyle = field(default_factory=lambda: NodeStyle(
        bg_color="#e2e8f0", border_color="#94a3b8", text_color="#334155", accent_color="#475569" # Slate-600
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

LIGHT_THEME = GraphvizStyle(
    background_color="#F8FAFC",
    edge_color="#64748b",
    cluster_fill_color="#FFFFFF",
    cluster_border_color="#cbd5e1",
    function_node=NodeStyle(
        bg_color="#FFFFFF", border_color="#cbd5e1", text_color="#0f172a", accent_color="#4f46e5", accent_text_color="#ffffff"
    ),
    pipeline_node=NodeStyle(
        bg_color="#fffbeb", border_color="#fcd34d", text_color="#451a03", accent_color="#d97706", accent_text_color="#ffffff"
    ),
    dual_node=NodeStyle(
        bg_color="#fdf4ff", border_color="#f0abfc", text_color="#4a044e", accent_color="#c026d3", accent_text_color="#ffffff"
    ),
    data_node=NodeStyle(
        bg_color="#f1f5f9", border_color="#94a3b8", text_color="#334155", accent_color="#475569", accent_text_color="#ffffff"
    )
)

# Dark theme adapted for the professional look
DARK_THEME = GraphvizStyle(
    background_color="#020617", # Slate-950
    edge_color="#475569", # Slate-600
    cluster_fill_color="#0f172a", # Slate-900
    cluster_border_color="#334155", # Slate-700
    cluster_label_color="#cbd5e1",
    function_node=NodeStyle(
        bg_color="#1e293b", border_color="#334155", text_color="#f1f5f9", accent_color="#6366f1", accent_text_color="#ffffff"
    ),
    pipeline_node=NodeStyle(
        bg_color="#2a1b08", border_color="#78350f", text_color="#f1f5f9", accent_color="#d97706", accent_text_color="#ffffff"
    ),
    dual_node=NodeStyle(
        bg_color="#2e1025", border_color="#86198f", text_color="#f1f5f9", accent_color="#c026d3", accent_text_color="#ffffff"
    ),
    data_node=NodeStyle(
        bg_color="#0f172a", border_color="#334155", text_color="#94a3b8", accent_color="#475569", accent_text_color="#e2e8f0"
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
    "default": LIGHT_THEME, # Modern default
    "light": LIGHT_THEME,
    "dark": DARK_THEME,
    "legacy": LEGACY_DEFAULT,
    "minimal": GraphvizStyle(
        function_node=NodeStyle(bg_color="#FFFFFF", border_color="#E8E8E8", text_color="#000000", accent_color="#FFFFFF", accent_text_color="#000000"),
        edge_color="#999999",
        cluster_fill_color="#FAFAFA",
    ),
}
