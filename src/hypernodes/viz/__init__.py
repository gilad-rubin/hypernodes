"""Visualization package for hypernodes."""

from typing import Any, Optional, Union

from ..pipeline import Pipeline
from .graphviz.renderer import GraphvizRenderer
from .graphviz.style import DESIGN_STYLES, GraphvizStyle
from .js.renderer import JSRenderer
from .ui_handler import UIHandler


def visualize(
    pipeline: Pipeline,
    filename: Optional[str] = None,
    engine: Union[str, Any] = "graphviz",
    depth: Optional[int] = 1,
    interactive: bool = False,
    **kwargs
):
    """Visualize a pipeline.

    Args:
        pipeline: The pipeline to visualize.
        filename: Optional filename to save the visualization to.
        engine: "graphviz" or "ipywidget" (or custom).
        depth: Initial expansion depth.
        interactive: Whether to use interactive widget (for graphviz).
        **kwargs: Additional options passed to the renderer.

    Returns:
        The visualization object (Digraph, Widget, etc.)
    """
    if engine == "graphviz":
        if interactive:
            from .graphviz.widget import GraphvizWidget
            return GraphvizWidget(pipeline, depth=depth, theme=kwargs.get("style", "default"), **kwargs)
            
        # Static rendering
        handler = UIHandler(pipeline, depth=depth)
        # For static Graphviz, don't traverse collapsed pipelines (they should remain truly collapsed)
        graph_data = handler.get_visualization_data(traverse_collapsed=False)
        
        renderer = GraphvizRenderer(style=kwargs.get("style", "default"))
        svg_content = renderer.render(graph_data)
        
        if filename:
            with open(filename, "w") as f:
                f.write(svg_content)
                
        from IPython.display import HTML
        return HTML(svg_content)

    elif engine == "ipywidget":
        from .visualization_widget import PipelineWidget
        return PipelineWidget(pipeline, depth=depth, **kwargs)

    else:
        raise ValueError(f"Unknown engine: {engine}")
