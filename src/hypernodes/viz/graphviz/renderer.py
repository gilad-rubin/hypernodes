import re
from typing import Any, Dict, List, Optional

try:
    import graphviz

    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

from ..layout_estimator import LayoutEstimator
from ..structures import (
    DataNode,
    DualNode,
    FunctionNode,
    GroupDataNode,
    PipelineNode,
    VisualizationGraph,
    VizNode,
)
from ..utils import read_viz_asset
from .style import DESIGN_STYLES

COLOR_TOKEN_TO_PLACEHOLDER: Dict[str, str] = {
    "var(--hn-node-bg)": "#100001",
    "var(--hn-node-border)": "#100002",
    "var(--hn-node-text)": "#100003",
    "var(--hn-cluster-border)": "#100004",
    "var(--hn-cluster-text)": "#100005",
    "var(--hn-edge)": "#100006",
    "var(--hn-func-accent)": "#100010",
    "var(--hn-func-text)": "#100011",
    "var(--hn-func-bg)": "#100012",
    "var(--hn-func-border)": "#100013",
    "var(--hn-pipe-accent)": "#100020",
    "var(--hn-pipe-accent-text)": "#100021",
    "var(--hn-pipe-bg)": "#100022",
    "var(--hn-pipe-border)": "#100023",
    "var(--hn-pipe-text)": "#100024",
    "var(--hn-dual-accent)": "#100030",
    "var(--hn-dual-accent-text)": "#100031",
    "var(--hn-dual-bg)": "#100032",
    "var(--hn-dual-border)": "#100033",
    "var(--hn-dual-text)": "#100034",
    "var(--hn-data-bg)": "#100040",
    "var(--hn-data-border)": "#100041",
    "var(--hn-data-text)": "#100042",
    "var(--hn-data-accent)": "#100043",
    "var(--hn-input-bg)": "#100044",
    "var(--hn-input-border)": "#100045",
    "var(--hn-input-text)": "#100046",
    "var(--hn-input-accent)": "#100047",
    "var(--hn-surface-bg)": "#100050",
    "var(--hn-cluster-fill)": "#100051",
}

CSS_VAR_FALLBACKS: Dict[str, str] = {
    "--hn-surface-bg": "#f8fafc",
    "--hn-edge": "#94a3b8",
    "--hn-node-bg": "#ffffff",
    "--hn-node-border": "#e2e8f0",
    "--hn-node-text": "#0f172a",
    "--hn-cluster-border": "#cbd5e1",
    "--hn-cluster-text": "#475569",
    "--hn-cluster-fill": "#f8fafc",
    "--hn-func-accent": "#2563eb",
    "--hn-func-text": "#ffffff",
    "--hn-func-bg": "#dbeafe",
    "--hn-func-border": "#93c5fd",
    "--hn-pipe-accent": "#d97706",
    "--hn-pipe-accent-text": "#ffffff",
    "--hn-pipe-bg": "#fef3c7",
    "--hn-pipe-border": "#fde68a",
    "--hn-pipe-text": "#78350f",
    "--hn-dual-accent": "#9333ea",
    "--hn-dual-accent-text": "#ffffff",
    "--hn-dual-bg": "#f3e8ff",
    "--hn-dual-border": "#e9d5ff",
    "--hn-dual-text": "#6b21a8",
    "--hn-data-bg": "#f3f4f6",
    "--hn-data-border": "#d1d5db",
    "--hn-data-text": "#374151",
    "--hn-data-accent": "#4b5563",
    "--hn-input-bg": "#e0f2fe",
    "--hn-input-border": "#7dd3fc",
    "--hn-input-text": "#0c4a6e",
    "--hn-input-accent": "#0284c7",
}


def _extract_var_name(token: str) -> Optional[str]:
    if not token or not token.startswith("var("):
        return None
    inner = token[4:-1] if token.endswith(")") else token[4:]
    var_name = inner.split(",", 1)[0].strip()
    return var_name if var_name else None


def _build_hex_to_var_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for token, placeholder in COLOR_TOKEN_TO_PLACEHOLDER.items():
        var_name = _extract_var_name(token)
        if not var_name:
            continue
        fallback = CSS_VAR_FALLBACKS.get(var_name, "#000000")
        mapping[placeholder.lstrip("#").upper()] = f"var({var_name}, {fallback})"
    return mapping


HEX_TO_VAR = _build_hex_to_var_map()


def _color_token_to_hex(value: Optional[str], fallback: str = "#000000") -> str:
    if not value:
        return fallback
    normalized = value.strip()
    if not normalized:
        return fallback
    lower = normalized.lower()
    if lower in {"transparent", "none"}:
        return lower
    placeholder = COLOR_TOKEN_TO_PLACEHOLDER.get(normalized)
    if placeholder:
        return placeholder
    if normalized.startswith("var("):
        return fallback
    return normalized


def _strip_svg_prolog(svg_data: str) -> str:
    """Remove XML/DOCTYPE declarations that break HTML embedding."""
    lines = []
    skipping_doctype = False
    for line in svg_data.splitlines():
        stripped = line.strip()
        if stripped.startswith("<?xml"):
            continue
        if stripped.startswith("<!DOCTYPE"):
            skipping_doctype = True
            continue
        if skipping_doctype:
            if stripped.endswith(">"):
                skipping_doctype = False
            continue
        lines.append(line)
    return "\n".join(lines)


def _wrap_svg_html(svg_data: str) -> str:
    """Wrap responsive SVG in a scrollable container with theme support."""
    theme_utils = read_viz_asset("theme_utils.js")

    # Unique ID for this visualization to avoid conflicts
    import uuid

    viz_id = f"viz-{str(uuid.uuid4())[:8]}"

    script = f"""
    <script>
    (function() {{
        {theme_utils}
        
        const container = document.getElementById("{viz_id}");
        if (container && window.HyperNodesTheme) {{
            const {{ detectHostTheme }} = window.HyperNodesTheme;
            
            const THEME_VARS = {{
                light: {{
                    '--hn-surface-bg': '#ffffff',
                    '--hn-edge': '#94a3b8',
                    '--hn-node-bg': '#ffffff',
                    '--hn-node-border': '#e2e8f0',
                    '--hn-node-text': '#0f172a',
                    '--hn-cluster-border': '#cbd5e1',
                    '--hn-cluster-text': '#475569',
                    '--hn-func-accent': '#10b981',
                    '--hn-func-text': '#065f46',
                    '--hn-func-bg': '#d1fae5',
                    '--hn-func-border': '#10b981',
                    '--hn-pipe-accent': '#a855f7',
                    '--hn-pipe-accent-text': '#1e3a8a',
                    '--hn-pipe-bg': '#f3e8ff',
                    '--hn-pipe-border': '#a855f7',
                    '--hn-pipe-text': '#581c87',
                    '--hn-dual-accent': '#9333ea',
                    '--hn-dual-accent-text': '#ffffff',
                    '--hn-dual-bg': '#f3e8ff',
                    '--hn-dual-border': '#c026d3',
                    '--hn-dual-text': '#6b21a8',
                    '--hn-data-bg': '#f1f5f9',
                    '--hn-data-border': '#64748b',
                    '--hn-data-text': '#334155',
                    '--hn-data-accent': '#475569',
                    '--hn-input-bg': '#bae6fd',
                    '--hn-input-border': '#38bdf8',
                    '--hn-input-text': '#0c4a6e',
                    '--hn-input-accent': '#0284c7'
                }},
                dark: {{
                    '--hn-surface-bg': '#18181b',
                    '--hn-edge': '#71717a',
                    '--hn-node-bg': '#27272a',
                    '--hn-node-border': '#3f3f46',
                    '--hn-node-text': '#f4f4f5',
                    '--hn-cluster-border': '#3f3f46',
                    '--hn-cluster-text': '#a1a1aa',
                    '--hn-cluster-fill': #27272a',
                    '--hn-func-accent': '#10b981',
                    '--hn-func-text': '#d1fae5',
                    '--hn-func-bg': '#064e3b',
                    '--hn-func-border': '#10b981',
                    '--hn-pipe-accent': '#a855f7',
                    '--hn-pipe-accent-text': '#dbeafe',
                    '--hn-pipe-bg': '#581c87',
                    '--hn-pipe-border': '#a855f7',
                    '--hn-pipe-text': '#f3e8ff',
                    '--hn-dual-accent': '#e879f9',
                    '--hn-dual-accent-text': '#fdf4ff',
                    '--hn-dual-bg': '#4a044e',
                    '--hn-dual-border': '#c026d3',
                    '--hn-dual-text': '#fdf4ff',
                    '--hn-data-bg': '#27272a',
                    '--hn-data-border': '#94a3b8',
                    '--hn-data-text': '#e4e4e7',
                    '--hn-data-accent': '#71717a',
                    '--hn-input-bg': '#082f49',
                    '--hn-input-border': '#0ea5e9',
                    '--hn-input-text': '#e0f2fe',
                    '--hn-input-accent': '#38bdf8'
                }}
            }};

            function applyTheme() {{
                const detected = detectHostTheme();
                const theme = detected.theme;
                const vars = THEME_VARS[theme] || THEME_VARS.light;
                const surface = detected.background || vars['--hn-surface-bg'] || (theme === 'dark' ? '#020617' : '#ffffff');

                container.style.backgroundColor = surface;
                container.style.setProperty('--hn-surface-bg', surface);
                container.style.setProperty('--hn-cluster-fill', surface);

                Object.entries(vars).forEach(([key, val]) => {{
                    if (key === '--hn-surface-bg') return;
                    container.style.setProperty(key, val);
                }});
            }}
            
            applyTheme();
            
            // Observe theme changes
            try {{
                const parentDoc = window.parent?.document;
                if (parentDoc) {{
                    const observer = new MutationObserver(applyTheme);
                    observer.observe(parentDoc.body, {{ attributes: true, attributeFilter: ['class', 'data-vscode-theme-kind', 'style'] }});
                }}
            }} catch(e) {{}}
            
            // Media query listener
            if (window.matchMedia) {{
                window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);
            }}
        }}
    }})();
    </script>
    """

    return (
        f'<div id="{viz_id}" style="width:100%; overflow-x:auto; padding-bottom:8px; transition: background-color 0.3s; background-color: var(--hn-surface-bg, transparent);">'
        f"{svg_data}"
        "</div>"
        f"{script}"
    )


class GraphvizRenderer:
    """Renders VisualizationGraph using Graphviz."""

    def __init__(self, style: str = "default", orient: str = "TB"):
        self.style = DESIGN_STYLES.get(style, DESIGN_STYLES["default"])
        self.orient = orient if orient in ["TB", "LR", "BT", "RL"] else "TB"

    def _post_process_svg(self, svg_data: str) -> str:
        """Replace hex placeholders with CSS variables and inject styles."""

        # 1. Replace Hex Placeholders with var(...) calls
        for hex_core, var_str in HEX_TO_VAR.items():
            pattern = re.compile(f"#{hex_core}", re.IGNORECASE)
            svg_data = pattern.sub(var_str, svg_data)

        # 2. Inject CSS Variables Definition
        # NOTE: These defaults are just fallbacks. The JS script overwrites them.
        style_block = """
        <style>
            :root {
                --hn-surface-bg: #ffffff;
                --hn-edge: #94a3b8;
                --hn-node-bg: #ffffff;
                --hn-node-border: #e2e8f0;
                --hn-node-text: #0f172a;
                --hn-cluster-border: #cbd5e1;
                --hn-cluster-text: #475569;
                --hn-cluster-fill: #f8fafc;
                --hn-func-accent: #10b981;
                --hn-func-text: #065f46;
                --hn-func-bg: #d1fae5;
                --hn-func-border: #10b981;
                --hn-pipe-accent: #a855f7;
                --hn-pipe-accent-text: #1e3a8a;
                --hn-pipe-bg: #f3e8ff;
                --hn-pipe-border: #a855f7;
                --hn-pipe-text: #581c87;
                --hn-dual-accent: #9333ea;
                --hn-dual-accent-text: #ffffff;
                --hn-dual-bg: #f3e8ff;
                --hn-dual-border: #c026d3;
                --hn-dual-text: #6b21a8;
                --hn-data-bg: #f1f5f9;
                --hn-data-border: #64748b;
                --hn-data-text: #334155;
                --hn-data-accent: #475569;
                --hn-input-bg: #bae6fd;
                --hn-input-border: #38bdf8;
                --hn-input-text: #0c4a6e;
                --hn-input-accent: #0284c7;
            }
            
            @media (prefers-color-scheme: dark) {
                :root {
                    --hn-surface-bg: #18181b;
                    --hn-edge: #71717a;
                    --hn-node-bg: #27272a;
                    --hn-node-border: #3f3f46;
                    --hn-node-text: #f4f4f5;
                    --hn-cluster-border: #3f3f46;
                    --hn-cluster-text: #a1a1aa;
                    --hn-cluster-fill: #27272a;
                    --hn-func-accent: #10b981;
                    --hn-func-text: #d1fae5;
                    --hn-func-bg: #064e3b;
                    --hn-func-border: #10b981;
                    --hn-pipe-accent: #a855f7;
                    --hn-pipe-accent-text: #dbeafe;
                    --hn-pipe-bg: #581c87;
                    --hn-pipe-border: #a855f7;
                    --hn-pipe-text: #f3e8ff;
                    --hn-dual-accent: #e879f9;
                    --hn-dual-accent-text: #fdf4ff;
                    --hn-dual-bg: #4a044e;
                    --hn-dual-border: #c026d3;
                    --hn-dual-text: #fdf4ff;
                    --hn-data-bg: #27272a;
                    --hn-data-border: #94a3b8;
                    --hn-data-text: #e4e4e7;
                    --hn-data-accent: #71717a;
                    --hn-input-bg: #082f49;
                    --hn-input-border: #0ea5e9;
                    --hn-input-text: #e0f2fe;
                    --hn-input-accent: #38bdf8;
                }
            }
            
            .hn-cluster polygon {
                stroke: var(--hn-cluster-border) !important;
                fill: var(--hn-cluster-fill) !important;
            }
            .hn-cluster text {
                fill: var(--hn-cluster-text) !important;
            }
            
            .hn-edge path,
            .hn-edge polygon {
                stroke: var(--hn-edge) !important;
                fill: var(--hn-edge) !important;
            }
            .hn-edge text {
                fill: var(--hn-cluster-text) !important;
            }
        </style>
        """

        # Insert style block
        if "</svg>" in svg_data:
            return svg_data.replace("</svg>", style_block + "</svg>")
        return svg_data + style_block

    def _build_id_mapping(self, graph_data: VisualizationGraph) -> Dict[str, str]:
        """Build a mapping from raw node IDs to human-readable Graphviz identifiers.

        This ensures that tooltips and internal SVG IDs use descriptive names
        instead of raw Python object IDs. These IDs become the SVG <title> elements
        which appear as tooltips when hovering over nodes.
        """
        mapping = {}
        used_ids = set()

        for node in graph_data.nodes:
            # Generate base readable ID - use actual names without prefixes
            # since these become tooltips
            if isinstance(node, FunctionNode):
                readable_id = node.function_name
            elif isinstance(node, PipelineNode):
                readable_id = node.label
            elif isinstance(node, DataNode):
                readable_id = node.name
            elif isinstance(node, GroupDataNode):
                # This should rarely appear now that group_inputs defaults to False
                readable_id = "Grouped Inputs"
            else:
                # Fallback for unknown types
                readable_id = f"node_{abs(hash(node.id)) % 100000}"

            # Ensure uniqueness by appending counter if needed
            base_id = readable_id
            counter = 1
            while readable_id in used_ids:
                readable_id = f"{base_id}_{counter}"
                counter += 1

            used_ids.add(readable_id)
            mapping[node.id] = readable_id

        return mapping

    def _build_dot(self, graph_data: VisualizationGraph) -> "graphviz.Digraph":
        """Build the Graphviz Digraph object."""
        if not GRAPHVIZ_AVAILABLE:
            raise ImportError("Graphviz is not installed")

        # Build ID mapping for human-readable identifiers
        self._id_mapping = self._build_id_mapping(graph_data)

        # Estimate layout dimensions to set initial container size
        # This mimics the behavior in the interactive JS widget
        estimator = LayoutEstimator(graph_data)
        est_width, est_height = estimator.estimate()

        # Enforce minimums
        est_height = max(400, est_height)
        est_width = max(600, est_width)

        dot = graphviz.Digraph()

        graph_bg = _color_token_to_hex(self.style.background_color, "transparent")
        edge_color = _color_token_to_hex(self.style.edge_color, "#64748b")
        cluster_border_color = _color_token_to_hex(
            self.style.cluster_border_color, "#cbd5e1"
        )
        cluster_label_color = _color_token_to_hex(
            self.style.cluster_label_color, "#475569"
        )
        cluster_fill_color = _color_token_to_hex(
            self.style.cluster_fill_color, "transparent"
        )

        # Enhanced Graph Attributes
        dot.attr(
            "graph",
            bgcolor=graph_bg,
            rankdir=self.orient,
            splines="spline",  # Smoother lines
            nodesep="0.6",  # Compact horizontal space
            ranksep="0.8",  # Compact vertical space
            pad="0.55",
            fontname=self.style.font_name,
            overlap="false",
            outputorder="nodesfirst",
        )

        # Pass estimated dimensions to post-processor
        self._est_width = est_width
        self._est_height = est_height

        # Enhanced Node Attributes
        dot.attr(
            "node",
            shape="box",  # Default shape, will be overridden
            fontname=self.style.font_name,
            fontsize=str(self.style.font_size),
            margin="0.2,0.1",  # Padding inside node
            # Ensure no default colors interfere
            color="transparent",
            fillcolor="transparent",
            style="filled,rounded",
            height="0.55",  # Increased to 0.55 as requested
        )

        # Enhanced Edge Attributes
        # Use specific placeholder HEX for Edge
        dot.attr(
            "edge",
            arrowhead="vee",
            arrowsize=self.style.arrow_size,
            penwidth=self.style.edge_penwidth,
            color=edge_color,
            fontname=self.style.font_name,
            fontsize=str(self.style.edge_font_size),
            fontcolor=cluster_label_color,
            class_="hn-edge",
        )

        # Group nodes by parent for clustering
        nodes_by_parent: Dict[Optional[str], List[Any]] = {}
        for node in graph_data.nodes:
            parent = node.parent_id
            if parent not in nodes_by_parent:
                nodes_by_parent[parent] = []
            nodes_by_parent[parent].append(node)

        # Identify input nodes (DataNodes with no incoming edges)
        # Need edge targets
        target_ids = {e.target for e in graph_data.edges}
        self._input_ids = {
            n.id
            for n in graph_data.nodes
            if isinstance(n, (DataNode, GroupDataNode)) and n.id not in target_ids
        }

        # Recursive function to add nodes and clusters
        def add_nodes_recursive(parent_id: Optional[str], container):
            if parent_id not in nodes_by_parent:
                return

            for node in nodes_by_parent[parent_id]:
                if isinstance(node, PipelineNode) and node.is_expanded:
                    # Create cluster with mapped ID
                    mapped_id = self._id_mapping.get(node.id, node.id)
                    with container.subgraph(name=f"cluster_{mapped_id}") as c:
                        c.attr(label=node.label)
                        c.attr(style="rounded,dashed")
                        c.attr(color=cluster_border_color)
                        c.attr(fontcolor=cluster_label_color)
                        if cluster_fill_color and cluster_fill_color not in {
                            "transparent",
                            "none",
                        }:
                            c.attr(bgcolor=cluster_fill_color)
                            c.attr(fillcolor=cluster_fill_color)
                            c.attr(style="rounded,filled,dashed")
                        c.attr(fontname=self.style.font_name)
                        c.attr(margin="20")
                        c.attr(class_="hn-cluster")

                        # Recurse
                        add_nodes_recursive(node.id, c)
                else:
                    # Add node
                    self._add_node(node, container)

        # Start from root (None parent)
        add_nodes_recursive(None, dot)

        # Add edges using mapped IDs
        for edge in graph_data.edges:
            dot.edge(
                self._id_mapping.get(edge.source, edge.source),
                self._id_mapping.get(edge.target, edge.target),
                label=edge.label,
            )

        return dot

    def compute_layout(self, graph_data: VisualizationGraph) -> Dict[str, Any]:
        """Compute layout using Graphviz and return JSON data."""
        if not GRAPHVIZ_AVAILABLE:
            raise ImportError("Graphviz is not installed")

        dot = self._build_dot(graph_data)

        # Render to JSON
        # format='json' returns a bytes object containing JSON
        json_bytes = dot.pipe(format="json")

        import json

        return json.loads(json_bytes.decode("utf-8"))

    def render(self, graph_data: VisualizationGraph) -> str:
        """Render the graph to SVG."""
        if not GRAPHVIZ_AVAILABLE:
            return "<div>Graphviz not installed</div>"

        try:
            dot = self._build_dot(graph_data)
            svg_data = dot.pipe(format="svg").decode("utf-8")

            # Post-process to swap HEX for VARS
            svg_data = self._post_process_svg(svg_data)

            return _wrap_svg_html(_strip_svg_prolog(svg_data))
        except Exception as e:
            return f"<div>Error rendering graph: {e}</div>"

    def _add_node(self, node: VizNode, container):
        """Add a single node to the graph."""
        # Use mapped ID instead of raw node.id
        node_id = self._id_mapping.get(node.id, node.id)

        # Determine style
        if isinstance(node, FunctionNode):
            style_conf = (
                self.style.dual_node
                if isinstance(node, DualNode)
                else self.style.function_node
            )
            shape = "box"
            style_attr = "filled,rounded"  # Keep rounded
            label = node.label

        elif isinstance(node, PipelineNode):
            style_conf = self.style.pipeline_node
            shape = "box"
            style_attr = "filled,rounded"  # Keep rounded
            label = node.label

        elif isinstance(node, (DataNode, GroupDataNode)):
            # Check if it is an input node
            if node.id in getattr(self, "_input_ids", set()):
                style_conf = getattr(self.style, "input_node", self.style.data_node)
            else:
                style_conf = self.style.data_node
            
            # Data nodes as rounded rectangles (more rounded)
            shape = "box"
            style_attr = "filled,rounded"
            label = (
                node.name
                if isinstance(node, DataNode)
                else f"Inputs ({len(node.nodes)})"
            )

        else:
            style_conf = self.style.data_node
            shape = "box"
            style_attr = "filled"
            label = str(node.id)

        # Convert colors to hex placeholders
        bg = _color_token_to_hex(style_conf.bg_color, "#FFFFFF")
        border = _color_token_to_hex(style_conf.border_color, "#000000")
        text = _color_token_to_hex(style_conf.text_color, "#000000")

        container.node(
            node_id,
            label=label,
            shape=shape,
            style=style_attr,
            color=border,
            fillcolor=bg,
            fontcolor=text,
            penwidth="2.0",  # slightly thicker border for elegance
            # width and height are dynamic by default or constrained by global node attributes
        )
