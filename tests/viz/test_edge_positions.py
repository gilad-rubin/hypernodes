from hypernodes import Pipeline, node
from hypernodes.viz.ui_handler import UIHandler
from hypernodes.viz.js.renderer import JSRenderer


@node(output_name="mid")
def mid(x: int) -> int:
    return x + 1


@node(output_name="done")
def done(mid: int) -> int:
    return mid * 2


def test_edges_use_vertical_handles():
    pipeline = Pipeline(nodes=[mid, done])
    handler = UIHandler(pipeline, depth=2)
    graph = handler.get_visualization_data(traverse_collapsed=True)

    rf = JSRenderer().render(graph, separate_outputs=True, show_types=True)

    for edge in rf["edges"]:
        assert edge.get("sourcePosition") == "bottom"
        assert edge.get("targetPosition") == "top"
