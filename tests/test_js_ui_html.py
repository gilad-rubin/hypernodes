import pytest

from hypernodes import Pipeline, node
from hypernodes.viz.js.html_generator import generate_widget_html
from hypernodes.viz.js.renderer import JSRenderer
from hypernodes.viz.ui_handler import UIHandler


def test_js_ui_html_has_no_python_comments_in_js():
    """Ensure generated HTML doesn't include Python-style comments that break JS."""

    @node(output_name="y")
    def add(x: int, offset: int = 1) -> int:
        return x + offset

    pipeline = Pipeline([add]).bind(offset=1)
    handler = UIHandler(pipeline)
    graph_data = handler.get_visualization_data()
    
    # Use JSRenderer to transform to React Flow format
    renderer = JSRenderer()
    react_flow_data = renderer.render(graph_data)
    
    html = generate_widget_html(react_flow_data)

    # Regression guard: Python comments must not leak into JS blocks
    assert "# Let ELK handle size" not in html
    assert len(html) > 0


def test_debug_overlay_present():
    """Ensure debug overlay components are present in the generated HTML."""
    @node(output_name="y")
    def add(x: int) -> int:
        return x + 1

    pipeline = Pipeline([add])
    handler = UIHandler(pipeline)
    graph_data = handler.get_visualization_data()
    
    renderer = JSRenderer()
    react_flow_data = renderer.render(graph_data)
    
    html = generate_widget_html(react_flow_data)
    
    # Check for debug overlay controls
    assert "Debug overlays" in html
    assert "NODE BOUNDS" in html
    assert "DEBUG: Green=source, Blue=target" in html


def test_theme_debug_panel_position():
    """Ensure Theme Debug panel is positioned at bottom-left to avoid overlap."""
    @node(output_name="y")
    def add(x: int) -> int:
        return x + 1

    pipeline = Pipeline([add])
    handler = UIHandler(pipeline)
    graph_data = handler.get_visualization_data()
    
    renderer = JSRenderer()
    react_flow_data = renderer.render(graph_data)
    
    html = generate_widget_html(react_flow_data)
    
    # Check for bottom-left positioning in the Panel component
    # The HTML generator uses <${Panel} position="bottom-left" ...
    # which renders to React code. We check for the string presence.
    assert 'position="bottom-left"' in html
