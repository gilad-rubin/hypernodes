import html
import re
import subprocess
from typing import Dict, List, Optional, Union

from ..structures import (
    DataNode,
    DualNode,
    FunctionNode,
    GroupDataNode,
    PipelineNode,
    VisualizationGraph,
    VizEdge,
    VizNode,
)
from .style import DESIGN_STYLES, GraphvizTheme


class GraphvizRenderer:
    """Renders VisualizationGraph to Graphviz SVG via DOT format."""

    def __init__(self, style: Union[str, GraphvizTheme] = "default", separate_outputs: bool = False):
        """Initialize the renderer.
        
        Args:
            style: Visual style for the graph (theme name or GraphvizTheme object)
            separate_outputs: If True, render outputs as separate nodes.
                            If False (default), combine function nodes with their outputs.
        """
        if isinstance(style, str):
            self.style = DESIGN_STYLES.get(style, DESIGN_STYLES["default"])
        else:
            self.style = style
        self.separate_outputs = separate_outputs
        self.lines: List[str] = []
        self._indent_level = 0
        self.graph_data: Optional[VisualizationGraph] = None

    def render(self, graph_data: VisualizationGraph) -> str:
        """Generate SVG source code."""
        self.graph_data = graph_data
        self.lines = ["digraph G {"]
        self._indent_level += 1

        # Graph attributes
        for k, v in self.style.graph_attr.items():
            self._add_line(f'{k}="{v}";')

        # Node attributes
        node_attrs = ", ".join(
            f'{k}="{v}"' for k, v in self.style.node_attr_defaults.items()
        )
        self._add_line(f"node [{node_attrs}];")

        # Edge attributes
        edge_attrs = ", ".join(f'{k}="{v}"' for k, v in self.style.edge_attr.items())
        self._add_line(f"edge [{edge_attrs}];")
        self._add_line("")

        # Group nodes and outputs
        self.nodes_by_parent = {}
        self.outputs_by_source = {}
        node_map = {n.id: n for n in self.graph_data.nodes}

        for node in self.graph_data.nodes:
            # Group by parent_id
            self.nodes_by_parent.setdefault(node.parent_id, []).append(node)

            # Group outputs by source_id
            if isinstance(node, (DataNode, GroupDataNode)) and node.source_id:
                self.outputs_by_source.setdefault(node.source_id, []).append(node)

        # Start rendering recursively from root (parent_id=None)
        self._render_scope(None)

        # Render edges
        for edge in self.graph_data.edges:
            self._render_edge(edge, node_map)

        self._indent_level -= 1
        self.lines.append("}")

        dot_source = "\n".join(self.lines)

        # Convert to SVG using dot
        try:
            process = subprocess.run(
                ["dot", "-Tsvg"],
                input=dot_source.encode("utf-8"),
                capture_output=True,
                check=True,
            )
            svg_content = process.stdout.decode("utf-8")
            return self._post_process_svg(svg_content)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Graphviz failed: {e.stderr.decode('utf-8')}")
        except FileNotFoundError:
            raise RuntimeError(
                "Graphviz 'dot' executable not found. Please install Graphviz."
            )

    def _post_process_svg(self, svg_content: str) -> str:
        """Apply post-processing to make SVG adaptive to VS Code themes."""
        
        # 1. Remove background polygon (make it truly transparent)
        # Matches <polygon fill="..." stroke="..." points="..."/> that covers the graph
        # Graphviz typically puts this as the first polygon in the graph
        svg_content = re.sub(
            r'<polygon fill="[^"]*" stroke="[^"]*" points="[^"]*"/>', 
            '', 
            svg_content,
            count=1 
        )

        # 2. Replace structural colors with CSS variables for adaptivity
        
        # Identify the colors used in the current style that should be adaptive
        # Edges
        edge_color = self.style.edge_color
        # Universal outline (used for node borders)
        outline_color = self.style.node_styles['function'].color.outline 
        # Replacement target: 'currentColor' makes it inherit the text color of the container
        # (Dark in light mode, Light in dark mode)
        adaptive_color = "currentColor"

        # Replace hex colors with adaptive color
        # We use simple string replacement for robustness, assuming the hex codes are unique enough
        # and exactly match what we put in.
        
        if edge_color:
            # Replace edge stroke
            svg_content = svg_content.replace(f'stroke="{edge_color}"', f'stroke="{adaptive_color}"')
            # Replace edge fill (arrowheads)
            svg_content = svg_content.replace(f'fill="{edge_color}"', f'fill="{adaptive_color}"')
        
        return svg_content

    def _render_scope(self, parent_id: Optional[str]):
        """Recursively render nodes within a scope (cluster or root)."""
        nodes = self.nodes_by_parent.get(parent_id, [])

        for node in nodes:
            if self.separate_outputs:
                # Separate mode: outputs are rendered as individual nodes
                # 1. If PipelineNode AND expanded -> CLUSTER
                if isinstance(node, PipelineNode) and node.is_expanded:
                    self._render_cluster(node)

                # 2. If Function/Pipeline(collapsed)/Dual -> FUNCTION NODE only
                elif isinstance(node, (FunctionNode, PipelineNode, DualNode)):
                    self._render_function_node(node)

                # 3. If Data/Group -> DATA NODE (inputs OR outputs)
                elif isinstance(node, (DataNode, GroupDataNode)):
                    self._render_input_node(node)
            else:
                # Combined mode (default): outputs are part of function nodes
                # 1. If DataNode/GroupDataNode with source_id -> SKIP (handled by producer)
                if isinstance(node, (DataNode, GroupDataNode)) and node.source_id:
                    continue

                # 2. If PipelineNode AND expanded -> CLUSTER
                if isinstance(node, PipelineNode) and node.is_expanded:
                    self._render_cluster(node)

                # 3. If Function/Pipeline(collapsed)/Dual -> COMBINED NODE with outputs
                elif isinstance(node, (FunctionNode, PipelineNode, DualNode)):
                    outputs = self.outputs_by_source.get(node.id, [])
                    self._render_combined_node(node, outputs)

                # 4. If Input Node (Data/Group without source) -> INPUT NODE
                elif isinstance(node, (DataNode, GroupDataNode)):
                    self._render_input_node(node)

    def _render_cluster(self, node: PipelineNode):
        """Render a nested pipeline as a Graphviz cluster."""
        cluster_name = f"cluster_{abs(hash(node.id))}"
        label = node.label

        # Use the pipeline color's outline for the expanded cluster border
        # This matches the collapsed node outline color
        pipeline_style = self.style.node_styles.get("pipeline")
        border_color = pipeline_style.color.outline if pipeline_style else self.style.edge_color
        label_color = self.style.edge_color

        self._add_line(f'subgraph "{cluster_name}" {{')
        self._indent_level += 1

        self._add_line(f'label="{label}";')
        self._add_line('style="rounded,bold";')  # Solid outline, transparent background
        self._add_line(f'color="{border_color}";')
        self._add_line(f'fontcolor="{label_color}";')
        self._add_line(f'fontname="{self.style.font_name}";')
        self._add_line('margin="20";')
        # Match arrow width (usually 1.2)
        self._add_line('penwidth="1.5";')

        # Recurse
        self._render_scope(node.id)

        self._indent_level -= 1
        self._add_line("}")

    def _add_line(self, line: str):
        indent = "    " * self._indent_level
        self.lines.append(f"{indent}{line}")

    def _render_input_node(self, node: Union[DataNode, GroupDataNode]):
        # Determine config based on type
        config_key = "data"

        if isinstance(node, DataNode) and node.is_bound:
            config_key = "bound_data"
        elif isinstance(node, GroupDataNode):
            config_key = "bound_group" if node.is_bound else "group"

        config = self.style.node_styles.get(config_key, self.style.node_styles["data"])

        rows = []
        if isinstance(node, GroupDataNode):
            # Render each node in the group as a row
            for sub_node in node.nodes:
                label = sub_node.name
                type_hint = getattr(sub_node, "type_hint", None)
                if type_hint:
                    label += f" : {type_hint}"

                label_html = self._format_label_html(
                    label, color=config.color.text, is_bold=config.is_bold
                )
                rows.append(
                    f'<TR><TD ALIGN="CENTER" BALIGN="CENTER">{label_html}</TD></TR>'
                )
        else:
            label = self._get_label(node)
            # Add type hint if available
            type_hint = getattr(node, "type_hint", None)
            if type_hint:
                label += f" : {type_hint}"

            label_html = self._format_label_html(
                label, color=config.color.text, is_bold=config.is_bold
            )
            rows.append(
                f'<TR><TD ALIGN="CENTER" BALIGN="CENTER">{label_html}</TD></TR>'
            )

        table = f"""<
<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
{"".join(rows)}
</TABLE>
>"""
        # Use simple name for Graphviz identifier (becomes SVG title)
        gv_id = self._get_graphviz_id(node)
        self._add_line(
            f'"{gv_id}" [label={table}, shape="{config.shape}", style="{config.style}", fillcolor="{config.color.fill}", color="{config.color.outline}", margin="{config.margin}", penwidth="{config.penwidth}"];'
        )

    def _render_function_node(self, node: VizNode):
        """Render a FunctionNode, DualNode, or collapsed PipelineNode (separate mode)."""
        # Determine color and config
        config_key = "function"

        if isinstance(node, PipelineNode):
            config_key = "pipeline"
        elif isinstance(node, DualNode):
            config_key = "dual"

        config = self.style.node_styles.get(
            config_key, self.style.node_styles["function"]
        )

        node_label = self._get_label(node)
        node_label_html = self._format_label_html(
            node_label, color=config.color.text, is_bold=config.is_bold
        )

        rows = []
        # Header only (no outputs combined)
        rows.append(
            f'<TR><TD ALIGN="CENTER" BALIGN="CENTER">{node_label_html}</TD></TR>'
        )

        table = f"""<
<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
{"".join(rows)}
</TABLE>
>"""
        # Use simple name for Graphviz identifier (becomes SVG title)
        gv_id = self._get_graphviz_id(node)
        self._add_line(
            f'"{gv_id}" [label={table}, shape="{config.shape}", style="{config.style}", fillcolor="{config.color.fill}", color="{config.color.outline}", margin="{config.margin}", penwidth="{config.penwidth}"];'
        )

    def _render_combined_node(
        self, node: VizNode, outputs: List[Union[DataNode, GroupDataNode]]
    ):
        """Render a FunctionNode with its outputs combined (default mode)."""
        # Determine color and config
        config_key = "function"

        if isinstance(node, PipelineNode):
            config_key = "pipeline"
        elif isinstance(node, DualNode):
            config_key = "dual"

        config = self.style.node_styles.get(
            config_key, self.style.node_styles["function"]
        )

        node_label = self._get_label(node)
        node_label_html = self._format_label_html(
            node_label, color=config.color.text, is_bold=config.is_bold
        )

        rows = []
        # Header
        rows.append(
            f'<TR><TD ALIGN="CENTER" BALIGN="CENTER">{node_label_html}</TD></TR>'
        )

        if outputs:
            rows.append("<HR/>")
            # Outputs
            for out in outputs:
                out_label = self._get_label(out)
                out_label_esc = html.escape(out_label)

                # Add type hint if available
                type_hint = getattr(out, "type_hint", None)
                if type_hint:
                    out_label_esc += f" : <I>{html.escape(type_hint)}</I>"

                # Use same text color - outputs are typically not bold unless specified otherwise
                out_label_html = (
                    f'<FONT COLOR="{config.color.text}">{out_label_esc}</FONT>'
                )

                port_id = self._get_port_id(out)
                rows.append(
                    f'<TR><TD PORT="{port_id}" ALIGN="CENTER" BALIGN="CENTER">{out_label_html}</TD></TR>'
                )

        table = f"""<
<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
{"".join(rows)}
</TABLE>
>"""
        # Use simple name for Graphviz identifier (becomes SVG title)
        gv_id = self._get_graphviz_id(node)
        self._add_line(
            f'"{gv_id}" [label={table}, shape="{config.shape}", style="{config.style}", fillcolor="{config.color.fill}", color="{config.color.outline}", margin="{config.margin}", penwidth="{config.penwidth}"];'
        )

    def _render_edge(self, edge: VizEdge, node_map: Dict[str, VizNode]):
        source_id = edge.source
        target_id = edge.target

        source_node = node_map.get(source_id)
        target_node = node_map.get(target_id)

        if not source_node or not target_node:
            return

        if self.separate_outputs:
            # Separate mode: direct edges between all nodes
            # Use human-readable Graphviz IDs for both source and target
            gv_source = f'"{self._get_graphviz_id(source_node)}"'
            gv_target = f'"{self._get_graphviz_id(target_node)}"'
        else:
            # Combined mode: edges from outputs come from parent function node
            # For DataNode outputs that are part of a combined node,
            # connect from the parent combined node instead
            actual_source_id = source_id
            actual_source_node = source_node
            if isinstance(source_node, (DataNode, GroupDataNode)) and source_node.source_id:
                actual_source_id = source_node.source_id
                actual_source_node = node_map.get(actual_source_id)
                if not actual_source_node:
                    return

            # Skip internal edges (Function -> its own Output) in combined mode
            if (
                isinstance(target_node, (DataNode, GroupDataNode))
                and target_node.source_id == actual_source_id
            ):
                return

            gv_source = f'"{self._get_graphviz_id(actual_source_node)}"'
            gv_target = f'"{self._get_graphviz_id(target_node)}"'

        self._add_line(f"{gv_source} -> {gv_target};")

    def _get_graphviz_id(self, node: VizNode) -> str:
        """Get a human-readable Graphviz identifier for a node.
        
        This becomes the <title> element in the SVG output.
        Uses label/name instead of full ID to avoid prefixes like 'processor__'.
        """
        if isinstance(node, GroupDataNode):
            # Use first node's name for group
            if node.nodes:
                return node.nodes[0].name
            return "group"
        if hasattr(node, "label") and node.label:
            return node.label
        if hasattr(node, "name") and node.name:
            return node.name
        # Fallback to full ID
        return node.id

    def _get_label(self, node: VizNode) -> str:
        if isinstance(node, GroupDataNode):
            # Should not be called for GroupDataNode in new logic, but keep fallback
            return ", ".join(n.name for n in node.nodes)
        if hasattr(node, "label") and node.label:
            return node.label
        if hasattr(node, "name") and node.name:
            return node.name
        return "Node"

    def _format_label_html(
        self, label: str, color: Optional[str] = None, is_bold: bool = True
    ) -> str:
        """Format label for HTML context."""
        esc_label = html.escape(label)
        content = f"<B>{esc_label}</B>" if is_bold else esc_label
        if color:
            return f'<FONT COLOR="{color}">{content}</FONT>'
        return content

    def _get_port_id(self, node: VizNode) -> str:
        return "p" + str(abs(hash(node.id)))
