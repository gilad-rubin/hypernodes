"""Ensure collapsed visualizations still include nested nodes for interactive expansion."""

from hypernodes import Pipeline, node
from hypernodes.viz.ui_handler import UIHandler
from hypernodes.viz.structures import FunctionNode, PipelineNode


def test_collapsed_view_contains_nested_nodes():
    """Nested pipelines should be present in the graph even when initially collapsed."""

    @node(output_name="inner_out")
    def inner(x: int) -> int:
        return x * 2

    @node(output_name="final")
    def consumer(inner_out: int) -> int:
        return inner_out + 1

    inner_pipeline = Pipeline(nodes=[inner], name="inner")
    outer = Pipeline(nodes=[inner_pipeline.as_node(name="inner_pipeline"), consumer], name="outer")

    handler = UIHandler(outer, depth=1)
    viz_graph = handler.get_visualization_data()

    pipeline_node = next(n for n in viz_graph.nodes if isinstance(n, PipelineNode))
    assert not pipeline_node.is_expanded

    inner_nodes = [n for n in viz_graph.nodes if isinstance(n, FunctionNode) and n.function_name == "inner"]
    assert inner_nodes, "Nested pipeline nodes should be included even when collapsed"

    # Nested nodes should remain attached to their pipeline parent, ready for expansion in the UI.
    assert all(n.parent_id == pipeline_node.id for n in inner_nodes)
    # Ensure nested nodes participate in the graph (edges exist) so expansion can reveal full details.
    inner_id = inner_nodes[0].id
    assert any(e.source == inner_id or e.target == inner_id for e in viz_graph.edges)
