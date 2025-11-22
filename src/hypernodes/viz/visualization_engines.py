"""Pluggable visualization engines for rendering serialized pipeline graphs.

This module provides a protocol for visualization engines and implementations
for different frontends (Graphviz, IPyWidget, etc.).
"""

import base64
import json
from typing import Any, Dict, List, Literal, Optional, Protocol, Union

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

try:
    import ipywidgets as widgets
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False

from .visualization import (
    GraphvizStyle,
    DESIGN_STYLES,
    _escape_html,
    _make_svg_responsive,
    _wrap_svg_html,
)


class VisualizationEngine(Protocol):
    """Protocol for pluggable visualization engines.
    
    Engines take serialized graph data and render it in their specific format.
    """
    
    def render(self, serialized_graph: Dict[str, Any], **options) -> Any:
        """Render a serialized graph with engine-specific options.
        
        Args:
            serialized_graph: Dictionary with levels, nodes, edges
            **options: Engine-specific rendering options
        
        Returns:
            Engine-specific output (e.g., graphviz.Digraph, HTML, etc.)
        """
        ...


class GraphvizEngine:
    """Graphviz-based visualization engine.
    
    Renders serialized graphs using Graphviz with customizable styling.
    """
    
    def render(
        self,
        serialized_graph: Dict[str, Any],
        filename: Optional[str] = None,
        orient: Literal["TB", "LR", "BT", "RL"] = "TB",
        style: Union[str, GraphvizStyle] = "default",
        show_legend: bool = False,
        show_types: bool = True,
        return_type: Literal["auto", "graphviz", "html"] = "auto",
        flatten: bool = False,
        group_inputs: Optional[bool] = None,
        min_arg_group_size: Optional[int] = None,
        **kwargs
    ) -> Any:
        """Render serialized graph using Graphviz.
        
        Args:
            serialized_graph: Dictionary with levels, nodes, edges
            filename: Output filename (e.g., "pipeline.svg")
            orient: Graph orientation ("TB", "LR", "BT", "RL")
            style: Style name from DESIGN_STYLES or GraphvizStyle object
            show_legend: Whether to show a legend explaining node types
            show_types: Whether to show type hints and default values
            return_type: "auto", "graphviz", or "html"
            flatten: If True, render nested pipelines inline without containers
            group_inputs: If True, group multiple inputs targeting the same node
            min_arg_group_size: Minimum inputs required to form a group (None = never group)
            **kwargs: Additional graphviz attributes
        
        Returns:
            graphviz.Digraph object (or HTML in Jupyter if return_type="html")
        """
        if not GRAPHVIZ_AVAILABLE:
            raise ImportError(
                "Graphviz is not installed. Install it with: uv add hypernodes[viz]"
            )
        
        # Resolve style
        if isinstance(style, str):
            if style not in DESIGN_STYLES:
                raise ValueError(
                    f"Unknown style '{style}'. Choose from: {list(DESIGN_STYLES.keys())}"
                )
            style_obj = DESIGN_STYLES[style]
        else:
            style_obj = style
        
        # Create graphviz digraph
        dot = graphviz.Digraph(comment="Pipeline")
        dot.attr(rankdir=orient)
        dot.attr(bgcolor=style_obj.background_color)
        dot.attr(fontname=style_obj.font_name)
        dot.attr(fontsize=str(style_obj.font_size))
        dot.graph_attr.update({
            "ranksep": "0.8",
            "nodesep": "0.5",
            "pad": "0.06",
        })
        
        # Extract data from serialized graph
        levels = serialized_graph.get("levels", [])
        nodes = serialized_graph.get("nodes", [])
        edges = serialized_graph.get("edges", [])
        applied_options = serialized_graph.get("applied_options") or {}
        node_map = {node["id"]: node for node in nodes}
        bound_inputs_set = set().union(
            *[set(level.get("bound_inputs_at_this_level", [])) for level in levels]
        )
        effective_group_inputs = (
            group_inputs
            if group_inputs is not None
            else applied_options.get("group_inputs", True)
        )
        effective_min_group_size = (
            min_arg_group_size
            if min_arg_group_size is not None
            else applied_options.get("min_arg_group_size", 2)
        )
        grouped_inputs = self._select_grouped_inputs(
            serialized_graph.get("grouped_inputs", {}),
            fallback_edges=edges,
            bound_inputs_set=bound_inputs_set,
            group_inputs=effective_group_inputs,
            min_group_size=effective_min_group_size,
        )
        
        # Build level hierarchy and place nodes/inputs/groups in their levels
        level_map = {level["level_id"]: level for level in levels}
        children_by_level: Dict[str, list[str]] = {}
        for level in levels:
            pid = level.get("parent_level_id")
            if pid:
                children_by_level.setdefault(pid, []).append(level["level_id"])

        nodes_by_level: Dict[str, list[Dict[str, Any]]] = {}
        for node in nodes:
            level_id = node.get("level_id", "root")
            nodes_by_level.setdefault(level_id, []).append(node)

        input_levels = serialized_graph.get("input_levels", {}) or {}
        inputs_by_level: Dict[str, list[str]] = {}
        for name, lvl in input_levels.items():
            inputs_by_level.setdefault(lvl, []).append(name)
        # Ensure we don't drop inputs if mapping missing
        for edge in edges:
            src = edge.get("source")
            if isinstance(src, str) and src.startswith("input_"):
                param = src.replace("input_", "")
                if param not in input_levels:
                    inputs_by_level.setdefault("root", []).append(param)

        grouped_param_names = {
            param
            for buckets in grouped_inputs.values()
            for params in buckets.values()
            for param in params
        }
        input_params: Dict[str, set[str]] = {}
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if isinstance(source, str) and source.startswith("input_") and isinstance(target, str):
                param_name = source.replace("input_", "")
                input_params.setdefault(param_name, set()).add(target)

        def build_input_label(param_name: str) -> str:
            label = param_name
            for node in nodes:
                for input_info in node.get("inputs", []):
                    if input_info["name"] == param_name and input_info.get("type_hint"):
                        type_hint = input_info["type_hint"]
                        default_val = input_info.get("default_value", "")
                        if default_val:
                            label = f"{param_name} : {type_hint} = {default_val}"
                        else:
                            label = f"{param_name} : {type_hint}"
                        return _escape_html(label)
            return _escape_html(label)

        def render_level(container, level_id: str):
            # add nodes
            for node in nodes_by_level.get(level_id, []):
                node_id = node["id"]
                node_type = node.get("node_type", "STANDARD")
                if node_type == "PIPELINE":
                    if node.get("is_expanded", False):
                        # When expanded, only render its nested cluster (no standalone node)
                        continue
                    label = self._create_pipeline_node_label(node, style_obj)
                    container.node(
                        node_id,
                        label=label,
                        shape="box",
                        style="rounded,filled",
                        fillcolor=style_obj.pipeline_node_color,
                        fontname=style_obj.font_name,
                        fontsize=str(style_obj.font_size),
                        penwidth=str(style_obj.node_border_width),
                        margin=style_obj.node_padding,
                    )
                else:
                    label = self._create_node_label(node, style_obj, show_types)
                    node_color = (
                        style_obj.dual_node_color
                        if node_type == "DUAL"
                        else style_obj.func_node_color
                    )
                    container.node(
                        node_id,
                        label=label,
                        shape="box",
                        style="rounded,filled",
                        fillcolor=node_color,
                        fontname=style_obj.font_name,
                        fontsize=str(style_obj.font_size),
                        penwidth=str(style_obj.node_border_width),
                        margin=style_obj.node_padding,
                    )

            # add inputs for this level (except grouped)
            for param_name in inputs_by_level.get(level_id, []):
                if param_name in grouped_param_names:
                    continue
                is_bound = param_name in bound_inputs_set
                node_style = "dashed,filled" if is_bound else "filled"
                container.node(
                    f"input_{param_name}",
                    label=build_input_label(param_name),
                    shape="box",
                    style=node_style,
                    fillcolor=style_obj.arg_node_color,
                    fontname=style_obj.font_name,
                    fontsize=str(style_obj.font_size),
                    penwidth=str(style_obj.node_border_width),
                    margin=style_obj.node_padding,
                )

            # add grouped input nodes for this level
            if group_inputs and min_arg_group_size is not None:
                for consumer_id, buckets in grouped_inputs.items():
                    consumer_level = node_map.get(consumer_id, {}).get("level_id", "root")
                    if consumer_level != level_id:
                        continue
                    for group_type, params in buckets.items():
                        if not params:
                            continue
                        group_id = f"group_{consumer_id}_{group_type}"
                        label = self._create_group_label(
                            params, node_map.get(consumer_id), show_types=show_types
                        )
                        node_style = "dashed,filled" if group_type == "bound" else "filled"
                        container.node(
                            group_id,
                            label=label,
                            shape="box",
                            style=node_style,
                            fillcolor=style_obj.grouped_args_node_color,
                            fontname=style_obj.font_name,
                            fontsize=str(style_obj.font_size),
                            penwidth=str(style_obj.node_border_width),
                            margin=style_obj.node_padding,
                        )
                        container.edge(
                            group_id,
                            consumer_id,
                            color=style_obj.grouped_args_edge_color,
                            penwidth=str(style_obj.edge_width),
                        )

            # recurse into children
            for child_level in children_by_level.get(level_id, []):
                if flatten and level_id == "root":
                    render_level(container, child_level)
                    continue
                label = child_level
                parent_pnode_id = level_map.get(child_level, {}).get("parent_pipeline_node_id")
                if parent_pnode_id and parent_pnode_id in node_map:
                    label = node_map[parent_pnode_id].get("label", label)
                with container.subgraph(name=f"cluster_{child_level}") as sub:
                    sub.attr(
                        label=label,
                        fontname=style_obj.font_name,
                        fontsize=str(style_obj.font_size),
                        color=style_obj.cluster_border_color,
                        penwidth=str(style_obj.cluster_border_width),
                        style="rounded,filled",
                        fillcolor=style_obj.cluster_fill_color,
                        labelloc="t",
                        labeljust="c",
                        margin="16",
                    )
                    render_level(sub, child_level)

        render_level(dot, "root")

        # Add edges
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            edge_type = edge.get("edge_type", "data_flow")
            mapping_label = edge.get("mapping_label")

            # Skip edges that are represented by grouped inputs
            if (
                group_inputs
                and min_arg_group_size is not None
                and isinstance(source, str)
                and source.startswith("input_")
            ):
                param_name = source.replace("input_", "")
                if param_name in grouped_param_names:
                    consumers = input_params.get(param_name, set())
                    if len(consumers) == 1 and target in consumers:
                        continue
            
            # Determine edge color based on type
            if edge_type == "parameter_flow":
                edge_color = style_obj.arg_edge_color
            else:
                edge_color = style_obj.output_edge_color
            
            label_str = mapping_label if mapping_label else ""
            
            dot.edge(
                source,
                target,
                label=label_str,
                color=edge_color,
                penwidth=str(style_obj.edge_width),
                fontsize=str(style_obj.edge_font_size),
                fontname=style_obj.font_name,
            )

        # Add legend if requested
        if show_legend:
            self._add_legend(dot, style_obj)
        
        # Render to file if filename provided
        if filename:
            if "." in filename:
                base_name, format_ext = filename.rsplit(".", 1)
                dot.render(base_name, format=format_ext, cleanup=True)
            else:
                dot.render(filename, format="svg", cleanup=True)
        
        # Handle return type
        if return_type == "html":
            try:
                from IPython.display import HTML
                svg_data = dot.pipe(format="svg").decode("utf-8")
                responsive_svg = _make_svg_responsive(svg_data)
                return HTML(_wrap_svg_html(responsive_svg))
            except ImportError:
                return dot
        elif return_type == "auto":
            try:
                from IPython import get_ipython
                if get_ipython() is not None:
                    from IPython.display import HTML
                    svg_data = dot.pipe(format="svg").decode("utf-8")
                    responsive_svg = _make_svg_responsive(svg_data)
                    return HTML(_wrap_svg_html(responsive_svg))
            except ImportError:
                pass
            return dot
        else:
            return dot
    
    def _create_node_label(
        self,
        node: Dict[str, Any],
        style: GraphvizStyle,
        show_types: bool = True
    ) -> str:
        """Create HTML label for a function node from serialized data."""
        label = node.get("label", "unknown")
        output_names = node.get("output_names", [])
        node_type = node.get("node_type", "STANDARD")
        
        # Add visual indicator for DualNode
        if node_type == "DUAL":
            label = f"{label} ◆"
        
        # Choose color based on node type
        node_color = style.dual_node_color if node_type == "DUAL" else style.func_node_color
        
        # Format outputs
        output_str = ", ".join(output_names) if output_names else "result"
        
        # Get return type if available and show_types is True
        # For now, we'll skip return type since it's not in serialized data
        # Could be added later if needed
        
        # Escape HTML
        label_esc = _escape_html(label)
        output_str_esc = _escape_html(output_str)
        
        html_label = f'''<
<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">
  <TR><TD><B>{label_esc}</B></TD></TR>
  <TR><TD BGCOLOR="{node_color}">{output_str_esc}</TD></TR>
</TABLE>>'''
        
        return html_label
    
    def _create_pipeline_node_label(
        self,
        node: Dict[str, Any],
        style: GraphvizStyle
    ) -> str:
        """Create label for a collapsed pipeline node."""
        label = node.get("label", "pipeline")
        output_names = node.get("output_names", [])
        
        outputs = ", ".join(output_names) if output_names else "..."
        
        label_esc = _escape_html(label)
        outputs_esc = _escape_html(outputs)
        
        html_label = f'''<
<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">
  <TR><TD><B>{label_esc}</B> ⚙</TD></TR>
  <TR><TD BGCOLOR="{style.pipeline_node_color}">{outputs_esc}</TD></TR>
</TABLE>>'''
        
        return html_label
    
    def _add_legend(self, dot: "graphviz.Digraph", style: GraphvizStyle):
        """Add legend to the graph."""
        with dot.subgraph(name="cluster_legend") as legend:
            legend.attr(label="Legend", fontsize=str(style.legend_font_size))
            legend.attr(style="filled", fillcolor=style.legend_background_color)
            legend.attr(fontname=style.font_name)
            
            legend.node(
                "legend_input_unbound",
                label="Input (Unbound)",
                shape="box",
                style="filled",
                fillcolor=style.arg_node_color,
                fontname=style.font_name,
                fontsize=str(style.legend_font_size),
            )
            
            legend.node(
                "legend_input_bound",
                label="Input (Bound)",
                shape="box",
                style="dashed,filled",
                fillcolor=style.arg_node_color,
                fontname=style.font_name,
                fontsize=str(style.legend_font_size),
            )
            
            legend.node(
                "legend_function",
                label="Function",
                shape="box",
                style="rounded,filled",
                fillcolor=style.func_node_color,
                fontname=style.font_name,
                fontsize=str(style.legend_font_size),
            )
            
            legend.node(
                "legend_dual",
                label="Dual Node ◆",
                shape="box",
                style="rounded,filled",
                fillcolor=style.dual_node_color,
                fontname=style.font_name,
                fontsize=str(style.legend_font_size),
            )
            
            legend.node(
                "legend_pipeline",
                label="Pipeline ⚙",
                shape="box",
                style="rounded,filled",
                fillcolor=style.pipeline_node_color,
                fontname=style.font_name,
                fontsize=str(style.legend_font_size),
            )
            
            legend.node(
                "legend_mapping",
                label="a → b: Parameter Mapping",
                shape="plaintext",
                fontname=style.font_name,
                fontsize=str(style.legend_font_size),
            )

    def _compute_grouped_inputs(
        self,
        edges: list[Dict[str, Any]],
        bound_inputs: set[str],
        group_inputs: bool,
        min_group_size: Optional[int],
    ) -> Dict[str, Dict[str, list[str]]]:
        """Front-end-only grouping: inputs → grouped nodes by consumer."""
        if not group_inputs or min_group_size is None:
            return {}

        input_consumers: Dict[str, set[str]] = {}
        for edge in edges:
            if edge.get("edge_type") != "parameter_flow":
                continue
            source = edge.get("source")
            target = edge.get("target")
            if isinstance(source, str) and source.startswith("input_") and isinstance(
                target, str
            ):
                param_name = source.replace("input_", "")
                input_consumers.setdefault(param_name, set()).add(target)

        grouped: Dict[str, Dict[str, list[str]]] = {}
        for param_name, consumers in input_consumers.items():
            # Only group if there is exactly one consumer
            if len(consumers) != 1:
                continue
            consumer = next(iter(consumers))
            group_type = "bound" if param_name in bound_inputs else "unbound"
            grouped.setdefault(consumer, {"bound": [], "unbound": []})[group_type].append(
                param_name
            )

        # Enforce minimum size and drop empty groups
        filtered: Dict[str, Dict[str, list[str]]] = {}
        for consumer, buckets in grouped.items():
            kept = {
                kind: params
                for kind, params in buckets.items()
                if len(params) >= min_group_size
            }
            if kept:
                filtered[consumer] = kept
        return filtered

    def _select_grouped_inputs(
        self,
        grouped_inputs: Dict[str, Dict[str, list[str]]],
        fallback_edges: List[Dict[str, Any]],
        bound_inputs_set: set[str],
        group_inputs: bool,
        min_group_size: Optional[int],
    ) -> Dict[str, Dict[str, list[str]]]:
        """Prefer pre-computed grouped inputs, falling back to local computation."""
        if not group_inputs or min_group_size is None:
            return {}

        if grouped_inputs:
            filtered: Dict[str, Dict[str, list[str]]] = {}
            for consumer, groups in grouped_inputs.items():
                bound = [
                    p for p in groups.get("bound", []) if len(groups.get("bound", [])) >= min_group_size
                ]
                unbound = [
                    p for p in groups.get("unbound", []) if len(groups.get("unbound", [])) >= min_group_size
                ]
                if bound or unbound:
                    filtered[consumer] = {"bound": sorted(bound), "unbound": sorted(unbound)}
            return filtered

        # Backward compatibility: compute if serializer did not provide data
        return self._compute_grouped_inputs(
            fallback_edges,
            bound_inputs_set,
            group_inputs=True,
            min_group_size=min_group_size,
        )

    def _create_group_label(
        self,
        param_names: list[str],
        consumer_node: Optional[Dict[str, Any]],
        show_types: bool = True,
    ) -> str:
        """Create HTML label for grouped inputs based on serialized node metadata."""
        rows = []
        inputs_meta = consumer_node.get("inputs", []) if consumer_node else []
        meta_lookup = {i.get("name"): i for i in inputs_meta}
        for param in param_names:
            meta = meta_lookup.get(param, {})
            type_hint = meta.get("type_hint")
            default_val = meta.get("default_value")
            label = _escape_html(param)
            if show_types and type_hint:
                label += f" : {_escape_html(type_hint)}"
                if default_val:
                    label += f" = {_escape_html(str(default_val))}"
            rows.append(f"<TR><TD>{label}</TD></TR>")

        rows_html = "\n  ".join(rows)
        return f"""<
<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">
  {rows_html}
</TABLE>>"""


class IPyWidgetEngine:
    """IPyWidget-based interactive visualization engine."""

    def render(
        self,
        serialized_graph: Dict[str, Any],
        theme: str = "CYBERPUNK",
        **kwargs,
    ) -> Any:
        """Render serialized graph as interactive widget."""
        if not IPYWIDGETS_AVAILABLE:
            raise ImportError(
                "ipywidgets is not installed. Install it with: uv add ipywidgets"
            )

        from .visualization_widget import (
            generate_widget_html,
            transform_to_react_flow,
        )

        react_flow_data = transform_to_react_flow(serialized_graph, theme=theme)
        html_content = generate_widget_html(react_flow_data)

        html_b64 = base64.b64encode(html_content.encode("utf-8")).decode("utf-8")
        iframe_html = (
            f'<iframe src="data:text/html;base64,{html_b64}" '
            f'width="100%" height="600px" frameborder="0"></iframe>'
        )
        return widgets.HTML(value=iframe_html)


def get_engine(engine_name: str) -> VisualizationEngine:
    """Get a visualization engine by name.
    
    Args:
        engine_name: Name of the engine ("graphviz" or "ipywidget")
    
    Returns:
        Engine instance
    
    Raises:
        ValueError: If engine name is unknown
    """
    engines = {
        "graphviz": GraphvizEngine(),
        "ipywidget": IPyWidgetEngine(),
    }
    
    if engine_name not in engines:
        raise ValueError(
            f"Unknown engine '{engine_name}'. Choose from: {list(engines.keys())}"
        )
    
    return engines[engine_name]
