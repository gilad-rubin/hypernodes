from hypernodes import Pipeline, node
from hypernodes.viz.js.renderer import JSRenderer
from hypernodes.viz.js.html_generator import generate_widget_html
from hypernodes.viz.ui_handler import UIHandler


@node(output_name="y")
def incr(x: int) -> int:
    return x + 1


def test_input_types_render_in_html():
    pipeline = Pipeline(nodes=[incr])
    handler = UIHandler(pipeline, depth=1, group_inputs=False, show_output_types=True)
    graph = handler.get_visualization_data(traverse_collapsed=True)

    rf = JSRenderer().render(graph, separate_outputs=True, show_types=True)
    html = generate_widget_html(rf)

    assert '"typeHint": "int"' in html or '"typeHint":"int"' in html
