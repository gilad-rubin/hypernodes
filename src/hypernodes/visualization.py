"""Pipeline visualization using Graphviz with multiple design variations."""
import inspect
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Literal, Union, get_type_hints
import networkx as nx
import graphviz

from .node import Node
from .pipeline import Pipeline, PipelineNode


# Maximum label length before truncation
MAX_LABEL_LENGTH = 30


def _escape_html(text: str) -> str:
    """Escape HTML special characters for use in graphviz labels."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


@dataclass
class GraphvizStyle:
    """Styling configuration for pipeline visualizations.
    
    Attributes:
        func_node_color: Background color for function nodes
        arg_node_color: Background color for input parameter nodes
        grouped_args_node_color: Background color for grouped input nodes
        arg_edge_color: Color for edges from input parameters
        output_edge_color: Color for edges between functions
        grouped_args_edge_color: Color for edges from grouped inputs
        font_name: Font family for all text
        font_size: Font size for node labels
        edge_font_size: Font size for edge labels
        legend_font_size: Font size for legend text
        background_color: Background color for the graph
        legend_background_color: Background color for legend box
        node_border_width: Width of node borders in pixels
        edge_width: Width of edges in pixels
        node_padding: Padding inside nodes
        cluster_border_color: Color for nested pipeline container borders
        cluster_border_width: Width of cluster borders
        cluster_fill_color: Fill color for nested pipeline containers
    """
    
    # Node colors
    func_node_color: str = "#87CEEB"  # Sky blue
    arg_node_color: str = "#90EE90"  # Light green
    grouped_args_node_color: str = "#90EE90"  # Light green (same as args)
    
    # Edge colors
    arg_edge_color: str = "#666666"  # Dark gray
    output_edge_color: str = "#333333"  # Darker gray
    grouped_args_edge_color: str = "#666666"  # Dark gray
    
    # Typography
    font_name: str = "Helvetica"
    font_size: int = 13
    edge_font_size: int = 11
    legend_font_size: int = 11
    
    # Background
    background_color: str = "#FFFFFF"  # White
    legend_background_color: str = "#F5F5F5"  # Light gray
    
    # Dimensions
    node_border_width: int = 2
    edge_width: int = 2
    node_padding: str = "0.16,0.10"  # width, height padding (tighter)
    
    # Cluster styling (for nested pipelines)
    cluster_border_color: str = "#999999"  # Medium gray
    cluster_border_width: int = 2
    cluster_fill_color: str = "#F9F9F9"  # Very light gray


# Predefined design variations
DESIGN_STYLES = {
    "default": GraphvizStyle(),
    
    "minimal": GraphvizStyle(
        func_node_color="#FFFFFF",
        arg_node_color="#F5F5F5",
        grouped_args_node_color="#F5F5F5",
        arg_edge_color="#CCCCCC",
        output_edge_color="#999999",
        grouped_args_edge_color="#CCCCCC",
        font_size=11,
        edge_font_size=9,
        node_border_width=1,
        edge_width=1,
        cluster_border_color="#DDDDDD",
        cluster_border_width=1,
        cluster_fill_color="#FAFAFA",
    ),
    
    "vibrant": GraphvizStyle(
        func_node_color="#FFB6C1",  # Light pink
        arg_node_color="#98FB98",  # Pale green
        grouped_args_node_color="#FFE4B5",  # Moccasin
        arg_edge_color="#FF6B6B",  # Light red
        output_edge_color="#4ECDC4",  # Turquoise
        grouped_args_edge_color="#FFA500",  # Orange
        font_size=13,
        edge_font_size=11,
        node_border_width=3,
        edge_width=3,
        cluster_border_color="#9B59B6",  # Purple
        cluster_border_width=3,
        cluster_fill_color="#F8E8FF",  # Very light purple
    ),
    
    "monochrome": GraphvizStyle(
        func_node_color="#E0E0E0",  # Light gray
        arg_node_color="#F5F5F5",  # Very light gray
        grouped_args_node_color="#ECECEC",  # Light gray
        arg_edge_color="#999999",
        output_edge_color="#666666",
        grouped_args_edge_color="#999999",
        font_name="Courier",
        font_size=11,
        edge_font_size=9,
        background_color="#FFFFFF",
        cluster_border_color="#888888",
        cluster_border_width=2,
        cluster_fill_color="#F8F8F8",
    ),
    
    "dark": GraphvizStyle(
        func_node_color="#3A506B",  # Dark blue-gray
        arg_node_color="#5C7C99",  # Medium blue-gray
        grouped_args_node_color="#6C8CA0",  # Light blue-gray
        arg_edge_color="#A8DADC",  # Light cyan
        output_edge_color="#F1FAEE",  # Off-white
        grouped_args_edge_color="#A8DADC",
        font_name="Helvetica",
        font_size=12,
        edge_font_size=10,
        background_color="#1D3557",  # Dark blue
        legend_background_color="#2A4A6A",
        node_border_width=2,
        edge_width=2,
        cluster_border_color="#A8DADC",
        cluster_border_width=2,
        cluster_fill_color="#2A4A6A",
    ),
    
    "professional": GraphvizStyle(
        func_node_color="#E8F4F8",  # Very light blue
        arg_node_color="#FFF9E6",  # Very light yellow
        grouped_args_node_color="#F0F8E8",  # Very light green
        arg_edge_color="#7D8A95",  # Blue-gray
        output_edge_color="#4A5568",  # Dark gray
        grouped_args_edge_color="#7D8A95",
        font_name="Arial",
        font_size=12,
        edge_font_size=10,
        background_color="#FFFFFF",
        node_border_width=2,
        edge_width=2,
        cluster_border_color="#CBD5E0",
        cluster_border_width=2,
        cluster_fill_color="#F7FAFC",
    ),
    
    "pastel": GraphvizStyle(
        func_node_color="#FFD6E8",  # Pastel pink
        arg_node_color="#C7E9F1",  # Pastel blue
        grouped_args_node_color="#E8F3D6",  # Pastel green
        arg_edge_color="#B19CD9",  # Pastel purple
        output_edge_color="#FF9AAD",  # Pastel coral
        grouped_args_edge_color="#FFD93D",  # Pastel yellow
        font_size=12,
        edge_font_size=10,
        node_border_width=2,
        edge_width=2,
        cluster_border_color="#FFB3D9",
        cluster_border_width=2,
        cluster_fill_color="#FFF5F8",
    ),
}


def _truncate_type(type_str: str, max_length: int = MAX_LABEL_LENGTH) -> str:
    """Truncate type string to max length."""
    if len(type_str) <= max_length:
        return type_str
    return type_str[:max_length - 3] + "..."


def _format_type_hint(param_name: str, func: Any) -> str:
    """Extract and format type hint for a parameter."""
    try:
        hints = get_type_hints(func)
        if param_name in hints:
            type_obj = hints[param_name]
            # Get string representation
            type_str = str(type_obj)
            # Clean up typing module prefix
            type_str = type_str.replace("typing.", "")
            # Clean up class representation
            type_str = type_str.replace("<class '", "").replace("'>", "")
            # Remove module prefixes like __main__., mymodule., etc.
            import re
            type_str = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\.', '', type_str)
            return _truncate_type(type_str)
    except Exception:
        pass
    return ""


def _format_default_value(param_name: str, func: Any) -> str:
    """Extract and format default value for a parameter."""
    try:
        sig = inspect.signature(func)
        param = sig.parameters[param_name]
        if param.default != inspect.Parameter.empty:
            default_str = str(param.default)
            if len(default_str) > 20:
                default_str = default_str[:17] + "..."
            return f" = {default_str}"
    except Exception:
        pass
    return ""


def _format_return_type(func: Any) -> str:
    """Extract and format return type hint."""
    try:
        hints = get_type_hints(func)
        if "return" in hints:
            type_obj = hints["return"]
            type_str = str(type_obj)
            type_str = type_str.replace("typing.", "")
            # Clean up class representation
            type_str = type_str.replace("<class '", "").replace("'>", "")
            # Remove module prefixes like __main__., mymodule., etc.
            import re
            type_str = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\.', '', type_str)
            return _truncate_type(type_str)
    except Exception:
        pass
    return ""


def build_graph(
    pipeline: Pipeline,
    depth: Optional[int] = 1,
    _current_depth: int = 1,
    _parent_output_map: Optional[Dict[str, Any]] = None,
) -> nx.DiGraph:
    """Build NetworkX graph from pipeline.
    
    Args:
        pipeline: Pipeline to visualize
        depth: How many levels of nesting to expand (None = all levels)
        _current_depth: Internal tracker for recursion depth
        _parent_output_map: Mapping from output names to producing nodes (internal)
        
    Returns:
        NetworkX DiGraph with nodes and edges
    """
    g = nx.DiGraph()
    
    # Build a map of which nodes produce which outputs in THIS pipeline
    # If a nested pipeline is expanded at this depth, map its outputs to the
    # actual inner producing Node (not the Pipeline object). Otherwise map to
    # the Pipeline object (collapsed).
    local_output_map: Dict[str, Any] = {}
    for n in pipeline.nodes:
        # Check if this is a Pipeline or PipelineNode wrapping a Pipeline
        inner_pipeline = None
        if isinstance(n, Pipeline):
            inner_pipeline = n
        elif isinstance(n, PipelineNode):
            inner_pipeline = n.pipeline
        
        if inner_pipeline is not None:
            expanded = (depth is None or _current_depth < depth)
            if expanded:
                for inner in inner_pipeline.nodes:
                    local_output_map[inner.output_name] = inner
            else:
                for inner in inner_pipeline.nodes:
                    local_output_map[inner.output_name] = n
        else:
            outputs = n.output_name
            if isinstance(outputs, tuple):
                for out in outputs:
                    local_output_map[out] = n
            else:
                local_output_map[outputs] = n
    
    for node in pipeline.nodes:
        # Check if this is a Pipeline or PipelineNode wrapping a Pipeline
        inner_pipeline = None
        if isinstance(node, Pipeline):
            inner_pipeline = node
        elif isinstance(node, PipelineNode):
            inner_pipeline = node.pipeline
        
        # Handle nested pipelines (both direct Pipeline and PipelineNode)
        if inner_pipeline is not None:
            # Check if we should expand this nested pipeline
            should_expand = (depth is None or _current_depth < depth)
            
            if should_expand:
                # Recursively build graph for nested pipeline
                nested_graph = build_graph(inner_pipeline, depth, _current_depth + 1, local_output_map)
                # Add nested graph as a subgraph
                g = nx.compose(g, nested_graph)
                # Note: Edges into the nested pipeline are handled inside the
                # nested_graph via the _parent_output_map. We don't duplicate
                # connections here to avoid stray inputs or pipeline-ID nodes.
            else:
                # Treat as a single collapsed node
                g.add_node(node, node_type="pipeline")
                
                # Add edges for pipeline's parameters
                # Use node.parameters for PipelineNode, root_args for Pipeline
                params = node.parameters if isinstance(node, PipelineNode) else inner_pipeline.root_args
                for param in params:
                    if param in local_output_map:
                        producer = local_output_map[param]
                        g.add_edge(producer, node, param_name=param)
                    else:
                        # External input
                        g.add_edge(param, node, param_name=param)
        else:
            # Regular node
            g.add_node(node, node_type="function")
            
            # Add edges based on dependencies
            for param in node.parameters:
                if param in local_output_map:
                    # Dependency: another node's output
                    producer = local_output_map[param]
                    g.add_edge(producer, node, param_name=param)
                elif _parent_output_map is not None and param in _parent_output_map:
                    # Produced by parent (e.g., sibling nested pipeline)
                    producer = _parent_output_map[param]
                    g.add_edge(producer, node, param_name=param)
                else:
                    # Root argument: external input
                    g.add_edge(param, node, param_name=param)
    
    return g


def _identify_grouped_inputs(
    g: nx.DiGraph,
    min_group_size: Optional[int] = 2,
) -> Dict[Node, List[str]]:
    """Identify input parameters that should be grouped together.
    
    Groups inputs that are used exclusively by a single function.
    
    Args:
        g: NetworkX graph
        min_group_size: Minimum number of inputs to create a group (None = no grouping)
        
    Returns:
        Dictionary mapping nodes to lists of input parameters to group
    """
    if min_group_size is None:
        return {}
    
    # Find string nodes (input parameters)
    input_nodes = [n for n in g.nodes() if isinstance(n, str)]
    
    # Map each input to its consumers
    input_consumers: Dict[str, List[Node]] = {}
    for input_node in input_nodes:
        consumers = list(g.successors(input_node))
        # Filter to only Node/Pipeline instances
        consumers = [c for c in consumers if isinstance(c, (Node, Pipeline))]
        input_consumers[input_node] = consumers
    
    # Group inputs by their consumer
    consumer_inputs: Dict[Node, List[str]] = {}
    for input_node, consumers in input_consumers.items():
        # Only group if used by exactly one consumer
        if len(consumers) == 1:
            consumer = consumers[0]
            if consumer not in consumer_inputs:
                consumer_inputs[consumer] = []
            consumer_inputs[consumer].append(input_node)
    
    # Filter to only groups that meet min_group_size
    grouped = {
        node: inputs
        for node, inputs in consumer_inputs.items()
        if len(inputs) >= min_group_size
    }
    
    return grouped


def _create_node_label(
    node: Node,
    style: GraphvizStyle,
    show_types: bool = True,
) -> str:
    """Create HTML label for a function node."""
    # Handle PipelineNode which wraps a Pipeline
    if hasattr(node.func, '__name__'):
        func_name = node.func.__name__
    elif isinstance(node.func, Pipeline):
        # PipelineNode case - check if it has a custom name
        if hasattr(node, 'name') and node.name:
            func_name = node.name
        else:
            func_name = "nested_pipeline"
    else:
        func_name = str(node.func)
    output_name = node.output_name
    
    # Handle tuple output names
    if isinstance(output_name, tuple):
        output_name = ", ".join(output_name)
    
    # Get return type
    return_type = ""
    if show_types:
        return_type = _format_return_type(node.func)
    
    # Build HTML table - simpler layout without nested cells
    # Escape HTML special chars in user content
    func_name_esc = _escape_html(func_name)
    output_name_esc = _escape_html(output_name)
    return_type_esc = _escape_html(return_type) if return_type else ""
    
    if return_type:
        label = f'''<
<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">
  <TR><TD><B>{func_name_esc}</B></TD></TR>
  <TR><TD BGCOLOR="{style.func_node_color}">{output_name_esc} : {return_type_esc}</TD></TR>
</TABLE>>'''
    else:
        label = f'''<
<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">
  <TR><TD><B>{func_name_esc}</B></TD></TR>
  <TR><TD BGCOLOR="{style.func_node_color}">{output_name_esc}</TD></TR>
</TABLE>>'''
    
    return label


def _create_input_label(
    param_name: str,
    node: Optional[Node] = None,
    style: GraphvizStyle = None,
    show_types: bool = True,
) -> str:
    """Create label for an input parameter node."""
    param_name_esc = _escape_html(param_name)
    
    if node and show_types:
        type_hint = _format_type_hint(param_name, node.func)
        default_val = _format_default_value(param_name, node.func)
        
        if type_hint:
            type_hint_esc = _escape_html(type_hint)
            default_val_esc = _escape_html(default_val)
            return f"{param_name_esc} : {type_hint_esc}{default_val_esc}"
    
    return param_name_esc


def _create_grouped_inputs_label(
    param_names: List[str],
    consumer_node: Node,
    style: GraphvizStyle,
    show_types: bool = True,
) -> str:
    """Create HTML label for grouped input parameters."""
    rows = []
    for param in param_names:
        param_esc = _escape_html(param)
        if show_types:
            type_hint = _format_type_hint(param, consumer_node.func)
            default_val = _format_default_value(param, consumer_node.func)
            if type_hint:
                type_hint_esc = _escape_html(type_hint)
                default_val_esc = _escape_html(default_val)
                rows.append(f'<TR><TD ALIGN="LEFT">{param_esc} : {type_hint_esc}{default_val_esc}</TD></TR>')
            else:
                rows.append(f'<TR><TD ALIGN="LEFT">{param_esc}</TD></TR>')
        else:
            rows.append(f'<TR><TD ALIGN="LEFT">{param_esc}</TD></TR>')
    
    rows_html = "\n  ".join(rows)
    label = f'''<
<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">
  {rows_html}
</TABLE>>'''
    
    return label


def _create_pipeline_label(
    pipeline: Pipeline,
    style: GraphvizStyle,
) -> str:
    """Create label for a collapsed pipeline node."""
    # Get pipeline name - prioritize user-provided name
    if hasattr(pipeline, "name") and pipeline.name:
        pipeline_name = pipeline.name
    else:
        # Use output names as identifier
        outputs = pipeline.output_name
        if outputs:
            # Take first output or join if multiple
            if len(outputs) == 1:
                pipeline_name = f"{outputs[0]}_pipeline"
            else:
                pipeline_name = "nested_pipeline"
        else:
            pipeline_name = "pipeline"
    
    # Show output mapping
    outputs = ", ".join(pipeline.output_name) if pipeline.output_name else "..."
    
    # Escape HTML
    pipeline_name_esc = _escape_html(pipeline_name)
    outputs_esc = _escape_html(outputs)
    
    label = f'''<
<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">
  <TR><TD><B>{pipeline_name_esc}</B></TD></TR>
  <TR><TD><I>(Pipeline)</I></TD></TR>
  <TR><TD BGCOLOR="{style.func_node_color}">{outputs_esc}</TD></TR>
</TABLE>>'''
    
    return label


def visualize(
    pipeline: Pipeline,
    filename: Optional[str] = None,
    orient: Literal["TB", "LR", "BT", "RL"] = "TB",
    depth: Optional[int] = 1,
    flatten: bool = False,
    min_arg_group_size: Optional[int] = None,  # Changed default from 2 to None to disable grouping
    show_legend: bool = False,
    show_types: bool = True,
    style: Union[str, GraphvizStyle] = "default",
    return_type: Literal["auto", "graphviz", "html"] = "auto",
) -> Union[graphviz.Digraph, Any]:
    """Visualize a pipeline using Graphviz.
    
    Args:
        pipeline: Pipeline to visualize
        filename: Output filename (e.g., "pipeline.svg"). If None, returns object
        orient: Graph orientation ("TB", "LR", "BT", "RL")
        depth: Expansion depth for nested pipelines (1=collapsed, None=fully expand)
        flatten: If True, render nested pipelines inline without containers
        min_arg_group_size: Minimum inputs to group together (None=no grouping)
        show_legend: Whether to show a legend explaining node types
        show_types: Whether to show type hints and default values
        style: Style name from DESIGN_STYLES or GraphvizStyle object
        return_type: "auto", "graphviz", or "html"
        
    Returns:
        graphviz.Digraph object (or HTML in Jupyter if return_type="html")
    """
    # Resolve style
    if isinstance(style, str):
        if style not in DESIGN_STYLES:
            raise ValueError(f"Unknown style '{style}'. Choose from: {list(DESIGN_STYLES.keys())}")
        style_obj = DESIGN_STYLES[style]
    else:
        style_obj = style
    
    # Build graph
    g = build_graph(pipeline, depth=depth)
    
    # Identify grouped inputs
    grouped_inputs = _identify_grouped_inputs(g, min_arg_group_size)
    
    # Create graphviz digraph
    dot = graphviz.Digraph(comment="Pipeline")
    dot.attr(rankdir=orient)
    dot.attr(bgcolor=style_obj.background_color)
    dot.attr(fontname=style_obj.font_name)
    dot.attr(fontsize=str(style_obj.font_size))
    # Compact layout with better spacing
    dot.graph_attr.update({
        "ranksep": "0.8",  # Increased from 0.4 for better arrow length
        "nodesep": "0.5",  # Increased from 0.3 for better spacing
        "pad": "0.06",
    })
    
    # Add nodes with optional pipeline clusters
    added_grouped_inputs = set()  # Track grouped inputs already added
    
    def _pipeline_display_name(pl: Pipeline) -> str:
        # Prefer explicit id/name; fall back to outputs
        if hasattr(pl, 'id') and pl.id and not pl.id.startswith('pipeline_'):
            return str(pl.id)
        if hasattr(pl, 'name') and pl.name:
            return str(pl.name)
        outs = pl.output_name
        if outs:
            return outs[0] if len(outs) == 1 else "pipeline"
        return "pipeline"
    
    def add_nodes_in_container(container: graphviz.Digraph, pl: Pipeline, current_depth: int = 1):
        for item in pl.nodes:
            # Check if this is a Pipeline or PipelineNode wrapping a Pipeline
            inner_pipeline = None
            if isinstance(item, Pipeline):
                inner_pipeline = item
            elif isinstance(item, PipelineNode):
                inner_pipeline = item.pipeline
            
            if inner_pipeline is not None:
                should_expand = (depth is None or current_depth < depth)
                if should_expand:
                    # Expanded nested pipeline
                    if not flatten:
                        with container.subgraph(name=f"cluster_{id(inner_pipeline)}") as sub:
                            sub.attr(
                                label=_pipeline_display_name(inner_pipeline),
                                fontname=style_obj.font_name,
                                fontsize=str(style_obj.font_size),
                                color=style_obj.cluster_border_color,
                                penwidth=str(style_obj.cluster_border_width),
                                style="rounded,filled",
                                fillcolor=style_obj.cluster_fill_color,
                                labelloc="b",  # Revert: label inside cluster at bottom (previous behavior)
                                labeljust="c",
                                margin="16",
                                # External title drawn ON TOP of edges with semi-transparent background
                                xlabel=(f'''<
<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4" BGCOLOR="#FFFFFFCC">
  <TR><TD><B>{_escape_html(_pipeline_display_name(inner_pipeline))}</B></TD></TR>
 </TABLE>>'''),
                            )
                            add_nodes_in_container(sub, inner_pipeline, current_depth + 1)
                    else:
                        # Flattened view: add its contents directly without a cluster
                        add_nodes_in_container(container, inner_pipeline, current_depth + 1)
                else:
                    # Collapsed pipeline as a single node
                    # For PipelineNode, check if it has a custom name
                    if isinstance(item, PipelineNode) and item.name:
                        # Create custom label for named PipelineNode
                        label = _create_pipeline_label(inner_pipeline, style_obj)
                        # Override with custom name
                        pipeline_name_esc = _escape_html(item.name)
                        outputs = ", ".join(inner_pipeline.output_name) if inner_pipeline.output_name else "..."
                        outputs_esc = _escape_html(outputs)
                        label = f'''<
<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">
  <TR><TD><B>{pipeline_name_esc}</B></TD></TR>
  <TR><TD><I>(Pipeline)</I></TD></TR>
  <TR><TD BGCOLOR="{style_obj.func_node_color}">{outputs_esc}</TD></TR>
</TABLE>>'''
                    else:
                        label = _create_pipeline_label(inner_pipeline, style_obj)
                    
                    container.node(
                        str(id(item)),
                        label=label,
                        shape="box",
                        style="rounded,filled",
                        fillcolor=style_obj.func_node_color,
                        fontname=style_obj.font_name,
                        fontsize=str(style_obj.font_size),
                        penwidth=str(style_obj.node_border_width),
                        margin=style_obj.node_padding,
                    )
                    if item in grouped_inputs:
                        grp_label = _create_grouped_inputs_label(
                            grouped_inputs[item],
                            None,
                            style_obj,
                            show_types=False,
                        )
                        container.node(
                            f"group_{id(item)}",
                            label=grp_label,
                            shape="box",
                            style="filled",
                            fillcolor=style_obj.grouped_args_node_color,
                            fontname=style_obj.font_name,
                            fontsize=str(style_obj.font_size),
                            penwidth=str(style_obj.node_border_width),
                            margin=style_obj.node_padding,
                        )
                        added_grouped_inputs.update(grouped_inputs[item])
            else:
                # Function node
                label = _create_node_label(item, style_obj, show_types)
                container.node(
                    str(id(item)),
                    label=label,
                    shape="box",
                    style="rounded,filled",
                    fillcolor=style_obj.func_node_color,
                    fontname=style_obj.font_name,
                    fontsize=str(style_obj.font_size),
                    penwidth=str(style_obj.node_border_width),
                    margin=style_obj.node_padding,
                )
                # Add grouped inputs node inside same container if applicable
                if item in grouped_inputs:
                    grp_label = _create_grouped_inputs_label(grouped_inputs[item], item, style_obj, show_types)
                    container.node(
                        f"group_{id(item)}",
                        label=grp_label,
                        shape="box",
                        style="filled",
                        fillcolor=style_obj.grouped_args_node_color,
                        fontname=style_obj.font_name,
                        fontsize=str(style_obj.font_size),
                        penwidth=str(style_obj.node_border_width),
                        margin=style_obj.node_padding,
                    )
                    added_grouped_inputs.update(grouped_inputs[item])
    
    # Start with the root pipeline - no outer cluster for top level
    # Only nested pipelines get clusters
    add_nodes_in_container(dot, pipeline, current_depth=1)
    
    # Add remaining input nodes (not grouped) at top level
    for n in g.nodes():
        if isinstance(n, str) and n not in added_grouped_inputs:
            consumers = [c for c in g.successors(n) if isinstance(c, Node)]
            consumer_node = consumers[0] if consumers else None
            label = _create_input_label(n, consumer_node, style_obj, show_types)
            dot.node(
                str(id(n)),
                label=label,
                shape="box",
                style="dashed,filled",
                fillcolor=style_obj.arg_node_color,
                fontname=style_obj.font_name,
                fontsize=str(style_obj.font_size),
                penwidth=str(style_obj.node_border_width),
                margin=style_obj.node_padding,
            )
    
    # Add edges
    for source, target in g.edges():
        # Handle grouped inputs
        if isinstance(target, (Node, Pipeline)):
            if target in grouped_inputs:
                # Check if source is one of the grouped inputs
                if isinstance(source, str) and source in grouped_inputs[target]:
                    # Skip individual edges, we'll add one from the group
                    continue
        
        # Determine node IDs
        if isinstance(source, str):
            source_id = str(id(source))
            edge_color = style_obj.arg_edge_color
        else:
            source_id = str(id(source))
            edge_color = style_obj.output_edge_color
        
        if isinstance(target, str):
            target_id = str(id(target))
        else:
            target_id = str(id(target))
        
        dot.edge(
            source_id,
            target_id,
            label="",
            color=edge_color,
            penwidth=str(style_obj.edge_width),
        )
    
    # Add edges from grouped inputs to consumers
    for consumer_node, input_params in grouped_inputs.items():
        group_id = f"group_{id(consumer_node)}"
        dot.edge(
            group_id,
            str(id(consumer_node)),
            color=style_obj.grouped_args_edge_color,
            penwidth=str(style_obj.edge_width),
        )
    
    # Add legend if requested
    if show_legend:
        with dot.subgraph(name="cluster_legend") as legend:
            legend.attr(label="Legend", fontsize=str(style_obj.legend_font_size))
            legend.attr(style="filled", fillcolor=style_obj.legend_background_color)
            legend.attr(fontname=style_obj.font_name)
            
            legend.node(
                "legend_input",
                label="Input",
                shape="box",
                style="dashed,filled",
                fillcolor=style_obj.arg_node_color,
                fontsize=str(style_obj.legend_font_size),
            )
            legend.node(
                "legend_function",
                label="Function",
                shape="box",
                style="rounded,filled",
                fillcolor=style_obj.func_node_color,
                fontsize=str(style_obj.legend_font_size),
            )
            legend.node(
                "legend_grouped",
                label="Grouped Inputs",
                shape="box",
                style="filled",
                fillcolor=style_obj.grouped_args_node_color,
                fontsize=str(style_obj.legend_font_size),
            )
    
    # Render to file if filename provided
    if filename:
        # Extract format from filename
        if "." in filename:
            base_name, format_ext = filename.rsplit(".", 1)
            dot.render(base_name, format=format_ext, cleanup=True)
        else:
            dot.render(filename, format="svg", cleanup=True)
    
    # Handle return type
    if return_type == "html":
        # Return HTML for Jupyter
        try:
            from IPython.display import HTML
            svg_data = dot.pipe(format="svg").decode("utf-8")
            return HTML(svg_data)
        except ImportError:
            return dot
    elif return_type == "auto":
        # Auto-detect: HTML in Jupyter, graphviz otherwise
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                from IPython.display import HTML
                svg_data = dot.pipe(format="svg").decode("utf-8")
                return HTML(svg_data)
        except ImportError:
            pass
        return dot
    else:
        return dot
