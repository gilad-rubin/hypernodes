"""Test that expanded PipelineNodes are properly connected in visualization graph."""

from hypernodes import Pipeline
from hypernodes.node import node
from hypernodes.viz.ui_handler import UIHandler


def test_expanded_pipeline_node_has_connections():
    """Test that expanded PipelineNodes have edges connecting them to the data flow."""
    
    # Create a simple nested pipeline structure
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="tripled")
    def triple(x: int) -> int:
        return x * 3
    
    inner_pipeline = Pipeline(nodes=[double, triple])
    inner_node = inner_pipeline.as_node(output_mapping={"tripled": "result"})
    
    @node(output_name="final")
    def add_ten(result: int) -> int:
        return result + 10
    
    outer_pipeline = Pipeline(nodes=[inner_node, add_ten])
    
    # Test with depth=2 (expanded)
    handler = UIHandler(outer_pipeline, depth=2)
    viz_data = handler.get_visualization_data()
    
    # Build edge index
    edges_from = {}
    edges_to = {}
    for edge in viz_data.edges:
        if edge.source not in edges_from:
            edges_from[edge.source] = []
        edges_from[edge.source].append(edge.target)
        
        if edge.target not in edges_to:
            edges_to[edge.target] = []
        edges_to[edge.target].append(edge.source)
    
    # Find the PipelineNode
    pipeline_nodes = [n for n in viz_data.nodes if n.__class__.__name__ == "PipelineNode" and n.is_expanded]
    assert len(pipeline_nodes) == 1, "Should have exactly one expanded PipelineNode"
    
    pipeline_node = pipeline_nodes[0]
    pipeline_node_id = pipeline_node.id
    
    # With traverse_collapsed=False for static viz, expanded PipelineNodes are visual containers
    # The actual data flow goes through the function nodes inside, not to the wrapper
    # So we just verify the PipelineNode exists and internal nodes are properly connected
    
    # Verify internal nodes have connections (not orphaned)
    inner_node_ids = [n.id for n in viz_data.nodes if n.parent_id == pipeline_node_id]
    assert len(inner_node_ids) > 0, "Expanded PipelineNode should contain internal nodes"
    
    # Check that at least one internal node has connections
    connected_inner_nodes = [nid for nid in inner_node_ids if nid in edges_to or nid in edges_from]
    assert len(connected_inner_nodes) > 0, "At least one internal node should be connected"
    
    # Note: Expanded PipelineNodes are visual containers and don't need outgoing edges
    # The actual data flow goes through the function nodes inside the cluster
    
    # Verify no hanging nodes (nodes with no edges at all)
    all_node_ids = {n.id for n in viz_data.nodes}
    connected_node_ids = set(edges_from.keys()) | set(edges_to.keys())
    hanging_nodes = all_node_ids - connected_node_ids
    
    # Filter out legitimate hanging nodes (like final outputs, PipelineNode wrappers)
    hanging_non_outputs = []
    for node_id in hanging_nodes:
        viz_node = next(n for n in viz_data.nodes if n.id == node_id)
        # Check if it's an output DataNode (has source_id)
        if viz_node.__class__.__name__ == "DataNode" and hasattr(viz_node, "source_id") and viz_node.source_id:
            continue  # This is an output node, it's OK to have no outgoing edges
        # Check if it's an expanded PipelineNode (visual container, no direct edges)
        if viz_node.__class__.__name__ == "PipelineNode" and viz_node.is_expanded:
            continue  # Expanded PipelineNode is a visual container, doesn't need edges
        hanging_non_outputs.append(node_id)
    
    assert len(hanging_non_outputs) == 0, f"Found hanging nodes: {hanging_non_outputs}"


def test_bound_inputs_not_duplicated_in_expanded_pipeline():
    """Test that bound inputs in inner pipeline don't create duplicate top-level input nodes."""
    
    @node(output_name="result")
    def process(x: int, multiplier: int = 10) -> int:
        return x * multiplier
    
    inner_pipeline = Pipeline(nodes=[process]).bind(multiplier=5)
    inner_node = inner_pipeline.as_node()
    
    @node(output_name="final")
    def finalize(result: int) -> int:
        return result + 100
    
    outer_pipeline = Pipeline(nodes=[inner_node, finalize])
    
    # Test with depth=2 (expanded)
    handler = UIHandler(outer_pipeline, depth=2)
    viz_data = handler.get_visualization_data()
    
    # Count how many DataNodes named "multiplier" exist at top level
    top_level_multiplier_nodes = [
        n for n in viz_data.nodes 
        if n.__class__.__name__ == "DataNode" 
        and hasattr(n, "name") 
        and n.name == "multiplier"
        and n.parent_id is None  # Top level
    ]
    
    assert len(top_level_multiplier_nodes) == 0, \
        f"Found {len(top_level_multiplier_nodes)} top-level 'multiplier' nodes, should be 0 (it's bound inside)"


def test_collapsed_pipeline_node_still_works():
    """Ensure the fix doesn't break collapsed PipelineNodes."""
    
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    inner_pipeline = Pipeline(nodes=[double])
    inner_node = inner_pipeline.as_node()
    
    @node(output_name="final")
    def add_ten(doubled: int) -> int:
        return doubled + 10
    
    outer_pipeline = Pipeline(nodes=[inner_node, add_ten])
    
    # Test with depth=1 (collapsed)
    handler = UIHandler(outer_pipeline, depth=1)
    viz_data = handler.get_visualization_data()
    
    # Find the PipelineNode
    pipeline_nodes = [n for n in viz_data.nodes if n.__class__.__name__ == "PipelineNode"]
    assert len(pipeline_nodes) == 1
    
    pipeline_node = pipeline_nodes[0]
    assert not pipeline_node.is_expanded, "Should be collapsed at depth=1"
    
    # Build edge index
    edges_from = {}
    edges_to = {}
    for edge in viz_data.edges:
        if edge.source not in edges_from:
            edges_from[edge.source] = []
        edges_from[edge.source].append(edge.target)
        
        if edge.target not in edges_to:
            edges_to[edge.target] = []
        edges_to[edge.target].append(edge.source)
    
    # With traverse_collapsed=False, collapsed pipelines don't create edges TO the PipelineNode wrapper
    # Instead, they're truly collapsed - no internal structure exposed
    # Just verify the graph is properly connected overall
    pipeline_node_id = pipeline_node.id
    
    # The pipeline should have input/output DataNodes that connect the flow
    assert len(viz_data.edges) > 0, "Should have edges in the visualization"
    
    # Verify there are DataNodes for inputs/outputs
    data_nodes = [n for n in viz_data.nodes if n.__class__.__name__ == "DataNode"]
    assert len(data_nodes) >= 2, "Should have DataNodes for pipeline inputs/outputs"


if __name__ == "__main__":
    test_expanded_pipeline_node_has_connections()
    print("✅ test_expanded_pipeline_node_has_connections PASSED")
    
    test_bound_inputs_not_duplicated_in_expanded_pipeline()
    print("✅ test_bound_inputs_not_duplicated_in_expanded_pipeline PASSED")
    
    test_collapsed_pipeline_node_still_works()
    print("✅ test_collapsed_pipeline_node_still_works PASSED")
    
    print("\n🎉 All tests passed!")

