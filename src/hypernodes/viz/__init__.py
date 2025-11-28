"""Visualization package for hypernodes."""

from typing import Any, Optional, Union

from ..pipeline import Pipeline
from .graphviz.renderer import GraphvizRenderer
from .graphviz.style import DESIGN_STYLES, GraphvizTheme
from .js.renderer import JSRenderer
from .ui_handler import UIHandler


def visualize(
    pipeline: Pipeline,
    filename: Optional[str] = None,
    engine: Union[str, Any] = "graphviz",
    depth: Optional[int] = 1,
    interactive: bool = False,
    separate_outputs: bool = False,
    show_types: bool = True,
    **kwargs
):
    """Visualize a pipeline.

    Args:
        pipeline: The pipeline to visualize.
        filename: Optional filename to save the visualization to.
        engine: "graphviz" or "ipywidget" (or custom).
        depth: Initial expansion depth.
        interactive: Whether to use interactive widget (for graphviz).
        separate_outputs: If True, render outputs as separate nodes.
                         If False (default), combine function nodes with their outputs.
        show_types: If True (default), show type hints on nodes.
        **kwargs: Additional options passed to the renderer.

    Returns:
        The visualization object (Digraph, Widget, etc.)
    """
    if engine == "graphviz":
        if interactive:
            from .graphviz.widget import GraphvizWidget
            return GraphvizWidget(pipeline, depth=depth, theme=kwargs.get("style", "default"), **kwargs)
            
        # Static rendering
        # Default group_inputs to True for Graphviz unless explicitly disabled
        group_inputs = kwargs.get("group_inputs", True)
        handler = UIHandler(pipeline, depth=depth, group_inputs=group_inputs)
        # For static Graphviz, don't traverse collapsed pipelines (they should remain truly collapsed)
        graph_data = handler.get_visualization_data(traverse_collapsed=False)
        
        renderer = GraphvizRenderer(style=kwargs.get("style", "default"), separate_outputs=separate_outputs)
        svg_content = renderer.render(graph_data)
        
        if filename:
            with open(filename, "w") as f:
                f.write(svg_content)
                
        from IPython.display import HTML
        return HTML(svg_content)

    elif engine == "ipywidget":
        from .visualization_widget import PipelineWidget
        return PipelineWidget(pipeline, depth=depth, separate_outputs=separate_outputs, show_types=show_types, **kwargs)

    else:
        raise ValueError(f"Unknown engine: {engine}")
