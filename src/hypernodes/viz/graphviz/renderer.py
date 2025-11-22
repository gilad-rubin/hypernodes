import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

from ..structures import (
    DataNode,
    DualNode,
    FunctionNode,
    GroupDataNode,
    PipelineNode,
    VisualizationGraph,
    VizNode,
)
from .style import DESIGN_STYLES, GraphvizStyle, NodeStyle


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


def _parse_svg_length(length: Optional[str]) -> Optional[float]:
    """Convert SVG length strings (pt, in, cm, etc.) to pixels."""
    if not length:
        return None
    try:
        length = length.strip()
        unit = "".join(ch for ch in length if ch.isalpha())
        value_str = "".join(ch for ch in length if (ch.isdigit() or ch in ".-"))
        value = float(value_str)
    except ValueError:
        return None

    unit = unit or "px"
    unit = unit.lower()
    if unit == "px":
        return value
    if unit == "pt":
        return value * (96.0 / 72.0)
    if unit == "in":
        return value * 96.0
    if unit == "cm":
        return value * (96.0 / 2.54)
    if unit == "mm":
        return value * (96.0 / 25.4)
    return value


def _make_svg_responsive(svg_data: str) -> str:
    """Adjust Graphviz SVG output for responsive display inside Jupyter cells."""
    cleaned_svg = _strip_svg_prolog(svg_data)
    try:
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        root = ET.fromstring(cleaned_svg)

        width_px = _parse_svg_length(root.attrib.get("width"))

        for attr in ("width", "height"):
            if attr in root.attrib:
                root.attrib.pop(attr)

        if "viewBox" not in root.attrib:
            return cleaned_svg

        existing_style = root.attrib.get("style", "")
        style_bits = [bit for bit in existing_style.split(";") if bit]
        if width_px:
            adjusted_width = max(width_px - 48.0, width_px * 0.85, 120.0)
            style_bits.append(f"width:min(100%, {adjusted_width:.2f}px)")
        else:
            style_bits.append("width:100%")
        style_bits.extend(["max-width:100%", "height:auto", "display:block"])
        root.attrib["style"] = ";".join(dict.fromkeys(style_bits)) + ";"

        return ET.tostring(root, encoding="unicode")
    except ET.ParseError:
        return cleaned_svg


def _wrap_svg_html(svg_data: str) -> str:
    """Wrap responsive SVG in a scrollable container for notebook rendering."""
    return (
        '<div style="width:100%; overflow-x:auto; padding-bottom:8px;">'
        f"{svg_data}"
        "</div>"
    )


class GraphvizRenderer:
    """Renders VisualizationGraph using Graphviz."""

    def __init__(self, style: str = "default"):
        self.style = DESIGN_STYLES.get(style, DESIGN_STYLES["default"])

    def render(self, graph_data: VisualizationGraph) -> str:
        """Render the graph to SVG."""
        if not GRAPHVIZ_AVAILABLE:
            return "<div>Graphviz not installed</div>"

        dot = graphviz.Digraph()
        
        # Enhanced Graph Attributes (from reference)
        dot.attr(
            "graph",
            bgcolor=self.style.background_color,
            rankdir="TB",
            splines="spline",  # Smoother lines
            nodesep="1.05",    # More horizontal space
            ranksep="1.25",    # More vertical space
            pad="0.55",
            fontname=self.style.font_name,
            overlap="false",
            outputorder="nodesfirst",
        )
        
        # Enhanced Node Attributes
        dot.attr(
            "node",
            shape="box",
            style="rounded,filled",
            fontname=self.style.font_name,
            fontsize=str(self.style.font_size),
            margin="0.16,0.10",
            penwidth="1.6",
        )
        
        # Enhanced Edge Attributes
        dot.attr(
            "edge",
            arrowhead="vee",
            arrowsize=self.style.arrow_size,
            penwidth=self.style.edge_penwidth,
            color=self.style.edge_color,
            fontname=self.style.font_name,
            fontsize=str(self.style.edge_font_size),
            fontcolor=self.style.cluster_label_color,
        )

        # Group nodes by parent for clustering
        nodes_by_parent: Dict[Optional[str], List[Any]] = {}
        for node in graph_data.nodes:
            parent = node.parent_id
            if parent not in nodes_by_parent:
                nodes_by_parent[parent] = []
            nodes_by_parent[parent].append(node)

        # Recursive function to add nodes and clusters
        def add_nodes_recursive(parent_id: Optional[str], container):
            if parent_id not in nodes_by_parent:
                return

            for node in nodes_by_parent[parent_id]:
                if isinstance(node, PipelineNode) and node.is_expanded:
                    # Create cluster
                    with container.subgraph(name=f"cluster_{node.id}") as c:
                        c.attr(label=node.label)
                        c.attr(style="filled,dashed,rounded")
                        c.attr(color=self.style.cluster_border_color)
                        c.attr(fillcolor=self.style.cluster_fill_color)
                        c.attr(fontcolor=self.style.cluster_label_color)
                        c.attr(fontname=self.style.font_name)
                        c.attr(margin="20")
                        
                        # Recurse
                        add_nodes_recursive(node.id, c)
                else:
                    # Add node
                    self._add_node(node, container)

        # Start from root (None parent)
        add_nodes_recursive(None, dot)

        # Add edges
        for edge in graph_data.edges:
            dot.edge(
                edge.source,
                edge.target,
                label=edge.label,
                # Edge attrs are already set globally, can override here if needed
            )

        try:
            svg_data = dot.pipe(format="svg").decode("utf-8")
            
            # --- Post-processing for "Professional" Look (from reference) ---
            
            # 1. Inject Drop Shadow Filter
            defs = """
              <defs>
                <filter id="cardShadow" x="-20%" y="-20%" width="140%" height="140%">
                  <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0F172A" flood-opacity="0.15"/>
                </filter>
                <style>
                  g.edge path, g.edge polygon {
                    stroke-linecap: round;
                    stroke-linejoin: round;
                  }
                  g.node { cursor: pointer; }
                </style>
              </defs>
            """
            svg_data = re.sub(r"(<svg[^>]*>)", r"\1\n" + defs, svg_data, count=1)
            
            # 2. Apply Shadow Filter to Nodes
            svg_data = svg_data.replace('class="node"', 'class="node" filter="url(#cardShadow)"')
            
            # 3. Improve Rendering Precision
            svg_data = svg_data.replace('stroke-opacity="1"', 'stroke-opacity="1" shape-rendering="geometricPrecision"')
            
            return _wrap_svg_html(_make_svg_responsive(svg_data))
        except Exception as e:
            return f"<div>Error rendering graph: {e}</div>"

    def _add_node(self, node: VizNode, container):
        """Add a single node to the graph."""
        if isinstance(node, FunctionNode):
            node_style = self.style.dual_node if isinstance(node, DualNode) else self.style.function_node
            subtitle = "DUAL NODE" if isinstance(node, DualNode) else "FUNCTION"
            
            label = self._create_clean_label(
                title=node.label,
                subtitle=subtitle,
                style=node_style
            )
            
            container.node(
                node.id,
                label=label,
                shape="plain",
                # Colors handled by HTML label, but set border for fallback
                color=node_style.border_color,
            )
            
        elif isinstance(node, PipelineNode): # Collapsed
            node_style = self.style.pipeline_node
            label = self._create_clean_label(
                title=node.label,
                subtitle="PIPELINE",
                style=node_style,
                is_expandable=True
            )
            
            # Add URL for interactivity
            container.node(
                node.id,
                label=label,
                shape="plain",
                URL=f"hypernodes:expand?id={node.id}",
                tooltip="Click to expand",
            )
            
        elif isinstance(node, DataNode):
            node_style = self.style.data_node
            label = self._create_clean_label(
                title=node.name,
                subtitle=node.type_hint or "Data",
                style=node_style,
                is_compact=True
            )
            
            container.node(
                node.id,
                label=label,
                shape="plain",
            )
            
        elif isinstance(node, GroupDataNode):
            node_style = self.style.data_node
            label = self._create_clean_label(
                title="Inputs",
                subtitle=f"{len(node.nodes)} items",
                style=node_style,
                is_compact=True
            )
            
            container.node(
                node.id,
                label=label,
                shape="plain",
            )

    def _create_clean_label(self, title: str, subtitle: str, style: NodeStyle, is_expandable: bool = False, is_compact: bool = False) -> str:
        """Create a clean, professional HTML label matching the reference style."""
        
        def escape(s):
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") if s else ""

        title = escape(title)
        subtitle = escape(subtitle)
        
        # Font sizes
        title_size = "12" if is_compact else "14"
        sub_size = "9" if is_compact else "10"
        
        # Colors
        # We use the style's specific colors. 
        # Note: Reference uses hardcoded slate scales, we map our theme to that.
        type_color = style.accent_color
        text_color = style.text_color
        
        # Expansion hint
        expand_row = ""
        if is_expandable:
            expand_row = f'<TR><TD><FONT POINT-SIZE="9" COLOR="{self.style.cluster_border_color}">Click to expand ↓</FONT></TD></TR>'

        # Cell padding controls "breathing room" inside the card
        # Reduce padding to make compact
        padding = "4" if is_compact else "6"
        
        return f'''<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="{padding}" BGCOLOR="{style.bg_color}" STYLE="ROUNDED">
            <TR><TD ALIGN="CENTER"><B><FONT POINT-SIZE="{title_size}" COLOR="{text_color}">{title}</FONT></B></TD></TR>
            <TR><TD ALIGN="CENTER"><FONT POINT-SIZE="{sub_size}" COLOR="{type_color}">{subtitle}</FONT></TD></TR>
            {expand_row}
        </TABLE>>'''
