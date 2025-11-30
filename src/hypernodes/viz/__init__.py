"""Visualization package for hypernodes."""

import html as html_module
from typing import Any, Optional, Union

from .graphviz.renderer import GraphvizRenderer
from .graphviz.style import DESIGN_STYLES, GraphvizTheme
from .js.html_generator import generate_widget_html
from .js.renderer import JSRenderer
from .layout_estimator import LayoutEstimator
from .state_simulator import (
    diagnose_all_states,
    simulate_collapse_expand_cycle,
    simulate_state,
    verify_edge_alignment,
    verify_state,
)
from .ui_handler import UIHandler


# Lazy imports for ipywidgets-dependent modules
def diagnose_widget(*args, **kwargs):
    """Diagnose widget rendering issues. Requires ipywidgets."""
    from .debug import diagnose_widget as _diagnose_widget
    return _diagnose_widget(*args, **kwargs)


def quick_check(*args, **kwargs):
    """Quick check for widget rendering. Requires ipywidgets."""
    from .debug import quick_check as _quick_check
    return _quick_check(*args, **kwargs)


__all__ = [
    # Internals (for advanced use)
    "UIHandler",
    "GraphvizRenderer",
    "JSRenderer",
    # Widgets
    "PipelineWidget",
    "ScrollablePipelineWidget",
    # State simulation (for testing)
    "simulate_state",
    "verify_state",
    "verify_edge_alignment",
    "simulate_collapse_expand_cycle",
    "diagnose_all_states",
    # Debug (requires ipywidgets)
    "diagnose_widget",
    "quick_check",
    # Graphviz styling
    "DESIGN_STYLES",
    "GraphvizTheme",
]


# Lazy imports for widget classes (requires ipywidgets)
def __getattr__(name):
    if name == "PipelineWidget":
        from .visualization_widget import PipelineWidget
        return PipelineWidget
    elif name == "ScrollablePipelineWidget":
        from .visualization_widget import ScrollablePipelineWidget
        return ScrollablePipelineWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _detect_environment() -> str:
    """Detect the current execution environment."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return "terminal"
        shell_name = shell.__class__.__name__
        if shell_name in ("ZMQInteractiveShell", "Shell"):  # Jupyter, Colab
            return "jupyter"
        return "terminal"
    except ImportError:
        return "terminal"


def _render_interactive(
    pipeline: Any,
    filename: Optional[str],
    depth: Optional[int],
    separate_outputs: bool,
    show_types: bool,
    **kwargs
) -> Optional[str]:
    """Render interactive JS visualization."""
    # Build visualization data
    handler = UIHandler(pipeline, depth=depth, group_inputs=True)
    graph_data = handler.get_visualization_data(traverse_collapsed=True)
    
    # Estimate dimensions
    estimator = LayoutEstimator(graph_data)
    est_width, est_height = estimator.estimate()
    width = max(600, est_width)
    height = max(400, est_height)
    
    # Render to React Flow format
    renderer = JSRenderer()
    rf_data = renderer.render(
        graph_data,
        theme="auto",
        separate_outputs=separate_outputs,
        show_types=show_types,
    )
    
    # Generate HTML
    html_content = generate_widget_html(rf_data)
    
    # Save to file if requested
    if filename:
        # Ensure .html extension
        if not filename.endswith('.html'):
            filename = filename + '.html'
        with open(filename, "w") as f:
            f.write(html_content)
        return None
    
    # Display or return
    env = _detect_environment()
    if env == "jupyter":
        from IPython.display import HTML, display
        escaped = html_module.escape(html_content, quote=True)
        # CSS fix for VS Code white background
        css_fix = """
        <style>
        .cell-output-ipywidget-background { background-color: transparent !important; }
        .jp-OutputArea-output { background-color: transparent; }
        </style>
        """
        iframe_html = (
            f"{css_fix}"
            f'<iframe srcdoc="{escaped}" '
            f'width="{width}" height="{height}" frameborder="0" '
            f'style="border: none; width: {width}px; max-width: 100%; '
            f'height: {height}px; display: block; background: transparent; margin: 0 auto;" '
            f'sandbox="allow-scripts allow-same-origin allow-popups allow-forms">'
            f"</iframe>"
        )
        display(HTML(iframe_html))
        return None
    else:
        return html_content


def _render_graphviz(
    pipeline: Any,
    filename: Optional[str],
    depth: Optional[int],
    separate_outputs: bool,
    show_types: bool,
    orient: str,
    flatten: bool,
    show_legend: bool,
    style: Union[str, Any],
    **kwargs
) -> Any:
    """Render static Graphviz visualization."""
    # If flatten, set depth to None (fully expanded)
    if flatten:
        depth = None
    
    # Build visualization data
    group_inputs = kwargs.get("group_inputs", True)
    handler = UIHandler(pipeline, depth=depth, group_inputs=group_inputs)
    graph_data = handler.get_visualization_data(traverse_collapsed=False)
    
    # Render to SVG
    renderer = GraphvizRenderer(
        style=style,
        separate_outputs=separate_outputs,
        show_types=show_types,
    )
    svg_content = renderer.render(graph_data)
    
    # Save to file if requested
    if filename:
        with open(filename, "w") as f:
            f.write(svg_content)
        return None
    
    # Display or return
    env = _detect_environment()
    if env == "jupyter":
        from IPython.display import HTML
        return HTML(svg_content)
    else:
        return svg_content
