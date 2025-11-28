from hypernodes import Pipeline, node
from hypernodes.viz.structures import DataNode
from hypernodes.viz.ui_handler import UIHandler


@node(output_name="inner_out")
def produce(x: int) -> int:
    return x + 1


@node(output_name="done")
def consume(inner_out: int) -> int:
    return inner_out


def test_collapsed_pipeline_outputs_anchor_to_pipeline_in_interactive_graph():
    inner = Pipeline(nodes=[produce], name="inner")
    outer = Pipeline(nodes=[inner.as_node(name="inner_step"), consume], name="outer")

    handler = UIHandler(outer, depth=1)
    graph_data = handler.get_visualization_data(traverse_collapsed=True)

    pipeline_id = "inner_step"
    boundary_outputs = [
        n for n in graph_data.nodes
        if isinstance(n, DataNode) and n.source_id == pipeline_id
    ]
    assert boundary_outputs, "collapsed pipeline should expose boundary outputs"

    output_node = boundary_outputs[0]
    assert output_node.name == "inner_out"

    inner_output_id = f"{pipeline_id}__inner_out"
    edge_pairs = {(e.source, e.target) for e in graph_data.edges}

    assert (inner_output_id, output_node.id) in edge_pairs
    assert (output_node.id, "consume") in edge_pairs
