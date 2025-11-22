from hypernodes import Pipeline, node
from hypernodes.viz.ui_handler import UIHandler
from hypernodes.viz.structures import DataNode, PipelineNode, FunctionNode


def test_orphaned_inputs_in_nested_pipeline():
    """Test that bound inputs don't become orphaned nodes when visualizing nested pipelines."""
    @node(output_name="clean")
    def clean_text(raw_text: str, suffix: str = "", language: str = "en") -> str:
        return f"{language}:{raw_text.strip()}{suffix}"

    inner = Pipeline(
        nodes=[clean_text],
        name="InnerPipeline",
    ).bind(suffix="*", language="en")

    inner_node = inner.as_node(
        name="InnerPipeline",
        input_mapping={"text_step": "raw_text"},
    )

    @node(output_name="text_step")
    def add_bang(text: str) -> str:
        return text + "!"

    outer = Pipeline(
        nodes=[add_bang, inner_node],
        name="OuterPipeline",
    )

    # Case 1: Collapsed (depth=1)
    # Bound inputs should be associated with the InnerPipeline node
    handler = UIHandler(outer, depth=1)
    viz_graph = handler.get_visualization_data()
    
    # Find input nodes (DataNode with no source_id and is_bound=True)
    bound_inputs = [n for n in viz_graph.nodes if isinstance(n, DataNode) and n.source_id is None and n.is_bound]
    
    # Should have bound inputs for suffix and language
    bound_names = [n.name for n in bound_inputs]
    # They may or may not be visible at depth=1, depending on visualization strategy
    # The key is that when expanded, they should be in the correct place

    # Case 2: Expanded (depth=2)
    # Bound inputs should be inside the nested pipeline, not at root level
    handler_expanded = UIHandler(outer, depth=2)
    viz_graph_expanded = handler_expanded.get_visualization_data()
    
    # Find the clean_text node (should be visible when expanded)
    clean_text_nodes = [n for n in viz_graph_expanded.nodes if isinstance(n, FunctionNode) and "clean_text" in n.function_name]
    assert len(clean_text_nodes) > 0, "clean_text node should be visible when expanded"
    
    # Bound inputs (suffix, language) should have same parent as clean_text
    # or should not be visible at root level
    clean_text_node = clean_text_nodes[0]
    clean_text_parent = clean_text_node.parent_id
    
    # Find all bound input nodes
    bound_inputs_expanded = [n for n in viz_graph_expanded.nodes if isinstance(n, DataNode) and n.is_bound and n.source_id is None]
    
    for bound_input in bound_inputs_expanded:
        if bound_input.name in ["suffix", "language"]:
            # These should have the same parent as clean_text (inside the nested pipeline)
            # or not be present at all (if collapsed)
            assert bound_input.parent_id == clean_text_parent, \
                f"{bound_input.name} should be in the same parent as clean_text, not at root level"
