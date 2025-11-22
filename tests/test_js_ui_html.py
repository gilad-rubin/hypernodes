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
