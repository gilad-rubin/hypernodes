import re
import html
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
    VizEdge,
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
    "var(--hn-group-bg)": "#100052",
    "var(--hn-group-border)": "#100053",
    "var(--hn-group-text)": "#100054",
    "var(--hn-group-accent)": "#100055",
}

CSS_VAR_FALLBACKS: Dict[str, str] = {
    "--hn-surface-bg": "#f8fafc",
    "--hn-edge": "#94a3b8",
    "--hn-node-bg": "#f8fbff",
    "--hn-node-border": "#d6e3f0",
    "--hn-node-text": "#0f172a",
    "--hn-cluster-border": "#d5deea",
    "--hn-cluster-text": "#475569",
    "--hn-cluster-fill": "#f3f6fb",
    "--hn-func-accent": "#5b6ee1",
    "--hn-func-text": "#0f172a",
    "--hn-func-bg": "#e8edff",
    "--hn-func-border": "#5b6ee1",
    "--hn-pipe-accent": "#f4a11f",
    "--hn-pipe-accent-text": "#0f172a",
    "--hn-pipe-bg": "#fff3d9",
    "--hn-pipe-border": "#f4a11f",
    "--hn-pipe-text": "#0f172a",
    "--hn-dual-accent": "#d66ae0",
    "--hn-dual-accent-text": "#0f172a",
    "--hn-dual-bg": "#ffe8fb",
    "--hn-dual-border": "#d66ae0",
    "--hn-dual-text": "#0f172a",
    "--hn-data-bg": "#edf2f7",
    "--hn-data-border": "#5f6b8a",
    "--hn-data-text": "#0f172a",
    "--hn-data-accent": "#5f6b8a",
    "--hn-input-bg": "#e3f5ff",
    "--hn-input-border": "#28a4e2",
    "--hn-input-text": "#0f172a",
    "--hn-input-accent": "#1c9ad6",
    "--hn-group-bg": "#eef2ff",
    "--hn-group-border": "#c7d2fe",
    "--hn-group-text": "#312e81",
    "--hn-group-accent": "#818cf8",
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


def _wrap_text_lines(text: str, max_chars: int = 24) -> List[str]:
    """
    Soft-wrap text for HTML-like Graphviz labels.
    Break on underscores, hyphens, and whitespace while avoiding leading separators.
    """
    if text is None:
        return [""]

    tokens = re.split(r"([_\-\s]+)", text)
    lines: List[str] = []
    current = ""

    for tok in tokens:
        if not tok:
            continue

        token = " " if tok.strip() == "" else tok
        # Flush the line if adding the token would exceed the limit
        if current and len(current) + len(token) > max_chars:
            lines.append(current.strip())
            current = ""
            token = token.lstrip("_- ").lstrip()

        # If the token itself is too long, hard-split it
        while len(token) > max_chars:
            lines.append(token[:max_chars].strip())
            token = token[max_chars:]

        current += token

    if current.strip():
        lines.append(current.strip())

    return lines or [""]


def _collapse_output_nodes(graph_data: VisualizationGraph) -> VisualizationGraph:
    """Remove output data nodes (with source_id) and rewire edges to producers."""
    data_nodes: Dict[str, DataNode] = {
        n.id: n for n in graph_data.nodes if isinstance(n, DataNode) and n.source_id
    }
    if not data_nodes:
        return graph_data

    output_ids = set(data_nodes.keys())
    new_nodes = [n for n in graph_data.nodes if n.id not in output_ids]
    new_edges: List[VizEdge] = []

    for edge in graph_data.edges:
        src = edge.source
        tgt = edge.target

        if src in output_ids:
            src = data_nodes[src].source_id or src
        if tgt in output_ids:
            tgt = data_nodes[tgt].source_id or tgt

        if src == tgt:
            continue

        new_edges.append(VizEdge(source=src, target=tgt, label=edge.label))

    return VisualizationGraph(nodes=new_nodes, edges=new_edges)


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
    theme_utils = read_viz_asset("theme_utils.js") or ""

    # Unique ID for this visualization to avoid conflicts
    import uuid

    viz_id = f"viz-{str(uuid.uuid4())[:8]}"

    script = f"""
    <script>
    (function() {{
        {theme_utils}
        
        const debugLog = (...args) => console.debug('[HN-Graphviz]', ...args);
        debugLog('theme utils loaded', Boolean(window.HyperNodesTheme));
        
        const container = document.getElementById("{viz_id}");
        if (!container) {{
            debugLog('container not found');
            return;
        }}

        const themeApi = window.HyperNodesTheme || {{}};
        const detectHostTheme = typeof themeApi.detectHostTheme === 'function'
            ? themeApi.detectHostTheme
            : null;
        
        const THEME_VARS = {{
            light: {{
                '--hn-surface-bg': '#f8fafc',
                '--hn-edge': '#94a3b8',
                '--hn-node-bg': '#f8fbff',
                '--hn-node-border': '#d6e3f0',
                '--hn-node-text': '#0f172a',
                '--hn-cluster-border': '#d5deea',
                '--hn-cluster-fill': '#f3f6fb',
                '--hn-cluster-text': '#475569',
                '--hn-func-bg': '#e8edff',
                '--hn-func-border': '#5b6ee1',
                '--hn-func-text': '#0f172a',
                '--hn-pipe-bg': '#fff3d9',
                '--hn-pipe-border': '#f4a11f',
                '--hn-pipe-text': '#0f172a',
                '--hn-dual-bg': '#ffe8fb',
                '--hn-dual-border': '#d66ae0',
                '--hn-dual-text': '#0f172a',
                '--hn-data-bg': '#edf2f7',
                '--hn-data-border': '#5f6b8a',
                '--hn-data-text': '#0f172a',
                '--hn-input-bg': '#e3f5ff',
                '--hn-input-border': '#28a4e2',
                '--hn-input-text': '#0f172a',
                '--hn-group-bg': '#eef2ff',
                '--hn-group-border': '#c7d2fe',
                '--hn-group-text': '#312e81'
            }},
            dark: {{
                '--hn-surface-bg': '#0b1021',
                '--hn-edge': '#7c8aa5',
                '--hn-node-bg': '#111827',
                '--hn-node-border': '#334155',
                '--hn-node-text': '#e5e7eb',
                '--hn-cluster-border': '#334155',
                '--hn-cluster-fill': '#111827',
                '--hn-cluster-text': '#cbd5e1',
                '--hn-func-bg': '#0f1b3d',
                '--hn-func-border': '#8c9bff',
                '--hn-func-text': '#e8edff',
                '--hn-pipe-bg': '#25160a',
                '--hn-pipe-border': '#ffb454',
                '--hn-pipe-text': '#fff3dc',
                '--hn-dual-bg': '#1c0f24',
                '--hn-dual-border': '#ff7bff',
                '--hn-dual-text': '#ffeefe',
                '--hn-data-bg': '#0b1d2c',
                '--hn-data-border': '#67e8f9',
                '--hn-data-text': '#e2e8f0',
                '--hn-input-bg': '#0b314c',
                '--hn-input-border': '#48c8ff',
                '--hn-input-text': '#e0f2ff',
                '--hn-group-bg': '#1f2937',
                '--hn-group-border': '#475569',
                '--hn-group-text': '#e5e7eb'
            }}
        }};

        const applyVars = (target, vars) => {{
            if (!target || !target.style) return;
            Object.entries(vars).forEach(([key, value]) => {{
                target.style.setProperty(key, value);
            }});
        }};

        function chooseTheme() {{
            const host = detectHostTheme ? detectHostTheme() : null;
            if (host) debugLog('host theme result:', host);
            const hostTheme = host?.theme;
            const hostBg = host?.background;
            const fallbackTheme = (() => {{
                if (window.matchMedia) {{
                    if (window.matchMedia('(prefers-color-scheme: light)').matches) return 'light';
                    if (window.matchMedia('(prefers-color-scheme: dark)').matches) return 'dark';
                }}
                return 'dark';
            }})();
            debugLog('fallback theme:', fallbackTheme);
            return {{
                theme: hostTheme || fallbackTheme,
                background: hostBg,
            }};
        }}

        function applyTheme() {{
            const {{ theme, background }} = chooseTheme();
            const vars = THEME_VARS[theme] || THEME_VARS.light;
            const surface = background || vars['--hn-surface-bg'];
            const clusterFill = vars['--hn-cluster-fill'];
            debugLog('applyTheme selection:', {{ theme, surface, clusterFill, keys: Object.keys(vars) }});

            container.style.backgroundColor = surface;
            container.style.setProperty('--hn-surface-bg', surface);
            container.style.setProperty('--hn-cluster-fill', clusterFill);
            container.dataset.hnTheme = theme;

            const svg = container.querySelector('svg');
            if (svg) {{
                applyVars(svg, vars);
                svg.style.backgroundColor = surface;
                
                const updateAttr = (selector, attr) => {{
                    svg.querySelectorAll(selector).forEach(el => {{
                        const current = el.getAttribute(attr);
                        if (!current || !current.startsWith('var(')) return;
                        const varName = current.match(/var\\(([^,)]+)/)?.[1];
                        const replacement = varName ? vars[varName] : null;
                        if (replacement) {{
                            el.setAttribute(attr, replacement);
                        }}
                    }});
                }};

                svg.querySelectorAll('polygon[fill]').forEach(el => {{
                    const currentFill = el.getAttribute('fill');
                    const varName = currentFill && currentFill.startsWith('var(')
                        ? currentFill.match(/var\\(([^,)]+)/)?.[1]
                        : null;

                    if (varName) {{
                        const replacement = vars[varName];
                        if (replacement) {{
                            el.setAttribute('fill', replacement);
                        }}
                        if (varName === '--hn-cluster-fill' && el.closest('g[class*=\"cluster\"]')) {{
                            el.setAttribute('fill', clusterFill);
                        }}
                    }}
                }});
                
                updateAttr('ellipse[fill]', 'fill');
                updateAttr('path[stroke]', 'stroke');
                updateAttr('text[fill]', 'fill');

                try {{
                    const containerBg = getComputedStyle(container).backgroundColor;
                    const svgBg = getComputedStyle(svg).backgroundColor;
                    debugLog('post-apply backgrounds:', {{ containerBg, svgBg }});
                }} catch (err) {{
                    debugLog('background read failed:', err);
                }}
            }}
        }}
            
            applyTheme();
            
            // Observe VSCode attribute changes (Method 3)
            const observer = new MutationObserver(applyTheme);
            observer.observe(document.body, {{ 
                attributes: true, 
                attributeFilter: ['class', 'data-vscode-theme-kind'] 
            }});
            observer.observe(document.documentElement, {{ 
                attributes: true, 
                attributeFilter: ['style'] 
            }});
            
            // Poll every 500ms as safety net
            setInterval(applyTheme, 500);
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
                --hn-surface-bg: #f8fafc;
                --hn-edge: #94a3b8;
                --hn-node-bg: #f8fbff;
                --hn-node-border: #d6e3f0;
                --hn-node-text: #0f172a;
                --hn-cluster-border: #d5deea;
                --hn-cluster-text: #475569;
                --hn-cluster-fill: #f3f6fb;
                --hn-func-accent: #5b6ee1;
                --hn-func-text: #0f172a;
                --hn-func-bg: #e8edff;
                --hn-func-border: #5b6ee1;
                --hn-pipe-accent: #f4a11f;
                --hn-pipe-accent-text: #0f172a;
                --hn-pipe-bg: #fff3d9;
                --hn-pipe-border: #f4a11f;
                --hn-pipe-text: #0f172a;
                --hn-dual-accent: #d66ae0;
                --hn-dual-accent-text: #0f172a;
                --hn-dual-bg: #ffe8fb;
                --hn-dual-border: #d66ae0;
                --hn-dual-text: #0f172a;
                --hn-data-bg: #edf2f7;
                --hn-data-border: #5f6b8a;
                --hn-data-text: #0f172a;
                --hn-data-accent: #5f6b8a;
                --hn-input-bg: #e3f5ff;
                --hn-input-border: #28a4e2;
                --hn-input-text: #0f172a;
                --hn-input-accent: #1c9ad6;
                --hn-group-bg: #eef2ff;
                --hn-group-border: #c7d2fe;
                --hn-group-text: #312e81;
                --hn-group-accent: #818cf8;
            }
            
            @media (prefers-color-scheme: dark) {
                :root {
                    --hn-surface-bg: #0b1021;
                    --hn-edge: #7c8aa5;
                    --hn-node-bg: #111827;
                    --hn-node-border: #334155;
                    --hn-node-text: #e5e7eb;
                    --hn-cluster-border: #334155;
                    --hn-cluster-text: #cbd5e1;
                    --hn-cluster-fill: #111827;
                    --hn-func-accent: #8c9bff;
                    --hn-func-text: #e8edff;
                    --hn-func-bg: #0f1b3d;
                    --hn-func-border: #8c9bff;
                    --hn-pipe-accent: #ffb454;
                    --hn-pipe-accent-text: #fff3dc;
                    --hn-pipe-bg: #25160a;
                    --hn-pipe-border: #ffb454;
                    --hn-pipe-text: #fff3dc;
                    --hn-dual-accent: #ff7bff;
                    --hn-dual-accent-text: #ffeefe;
                    --hn-dual-bg: #1c0f24;
                    --hn-dual-border: #ff7bff;
                    --hn-dual-text: #ffeefe;
                    --hn-data-bg: #0b1d2c;
                    --hn-data-border: #67e8f9;
                    --hn-data-text: #e2e8f0;
                    --hn-data-accent: #67e8f9;
                    --hn-input-bg: #0b314c;
                    --hn-input-border: #48c8ff;
                    --hn-input-text: #e0f2ff;
                    --hn-input-accent: #48c8ff;
                    --hn-group-bg: #1f2937;
                    --hn-group-border: #475569;
                    --hn-group-text: #e5e7eb;
                    --hn-group-accent: #475569;
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

        # Map node_id -> produced outputs for label decoration (use pre-collapse data)
        self._outputs_map: Dict[str, List[str]] = {}
        for n in graph_data.nodes:
            if isinstance(n, DataNode) and n.source_id:
                self._outputs_map.setdefault(n.source_id, []).append(n.name)

        # Collapse output data nodes (hide produced outputs)
        graph_data = _collapse_output_nodes(graph_data)

        # Enhanced Node Attributes
        dot.attr(
            "node",
            shape="box",  # Default shape, will be overridden
            fontname=self.style.font_name,
            fontsize=str(self.style.font_size),
            margin="0.18,0.09",  # Base padding (reduced)
            # Ensure no default colors interfere
            color="transparent",
            fillcolor="transparent",
            style="filled,rounded",
            fixedsize="false",
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

        elif isinstance(node, GroupDataNode):
            style_conf = getattr(self.style, "group_node", self.style.data_node)
            shape = "box"
            style_attr = "filled,rounded,dashed"
            label = f"Inputs ({len(node.nodes)})"

        elif isinstance(node, DataNode):
            # Check if it is an input node
            if node.id in getattr(self, "_input_ids", set()):
                style_conf = getattr(self.style, "input_node", self.style.data_node)
            else:
                style_conf = self.style.data_node

            shape = "box"
            style_attr = "filled,rounded"
            label = node.name

        else:
            style_conf = self.style.data_node
            shape = "box"
            style_attr = "filled"
            label = str(node.id)

        # Convert colors to hex placeholders
        bg = _color_token_to_hex(style_conf.bg_color, "#FFFFFF")
        border = _color_token_to_hex(style_conf.border_color, "#000000")
        text = _color_token_to_hex(style_conf.text_color, "#000000")
        font_sz = str(self.style.font_size)

        # Unified padding for all nodes
        # Use minimal margin to ensure accurate centering via HTML padding
        extra_attrs = {"margin": "0.03"}
        if isinstance(node, (DataNode, GroupDataNode)):
            extra_attrs["height"] = "0.5"

        # HTML-like labels for producers with outputs
        produced = self._outputs_map.get(node.id, [])
        if produced and not isinstance(node, (DataNode, GroupDataNode)):
            def _render_lines(lines: List[str], bold: bool = False, padding: str = "10") -> str:
                if not lines:
                    return ""
                start = "<B>" if bold else ""
                end = "</B>" if bold else ""
                return "".join(
                    f'<TR><TD ALIGN="CENTER" BALIGN="CENTER" VALIGN="MIDDLE" CELLPADDING="{padding}">'
                    f'<FONT POINT-SIZE="{font_sz}">{start}{html.escape(line)}{end}</FONT>'
                    "</TD></TR>"
                    for line in lines
                )

            title_lines = _wrap_text_lines(label)
            output_lines: List[str] = []
            for p in produced:
                output_lines.extend(_wrap_text_lines(p))

            divider_color = "#94a3b8"
            label = (
                '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0" ALIGN="CENTER">'
                f"{_render_lines(title_lines, bold=True, padding='10')}"
                '<TR><TD ALIGN="CENTER" BALIGN="CENTER" VALIGN="MIDDLE" CELLPADDING="8">'
                '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0" WIDTH="100%">'
                f'<TR><TD HEIGHT="1" WIDTH="100%" BGCOLOR="{divider_color}"></TD></TR>'
                "</TABLE>"
                "</TD></TR>"
                f"{_render_lines(output_lines, bold=False, padding='10')}"
                "</TABLE>>"
            )
        else:
            label_lines = _wrap_text_lines(label)

            def _render_simple(lines: List[str], bold: bool = False) -> str:
                start = "<B>" if bold else ""
                end = "</B>" if bold else ""
                return "".join(
                    f'<TR><TD ALIGN="CENTER" BALIGN="CENTER" VALIGN="MIDDLE" CELLPADDING="10">'
                    f'<FONT POINT-SIZE="{font_sz}">{start}{html.escape(line)}{end}</FONT>'
                    "</TD></TR>"
                    for line in lines
                )

            is_emphasized = isinstance(node, (FunctionNode, PipelineNode))
            label = (
                '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0" ALIGN="CENTER">'
                f"{_render_simple(label_lines, bold=is_emphasized)}"
                "</TABLE>>"
            )

        container.node(
            node_id,
            label=label,
            shape=shape,
            style=style_attr,
            color=border,
            fillcolor=bg,
            fontcolor=text,
            penwidth="2.0",  # slightly thicker border for elegance
            **extra_attrs
        )
