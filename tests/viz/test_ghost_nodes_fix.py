"""Test that collapsed pipelines don't create ghost nodes in visualization."""

from hypernodes import Pipeline, node
from hypernodes.viz.graph_walker import GraphWalker


def test_no_ghost_nodes_in_collapsed_pipeline():
    """Verify collapsed pipelines don't create ghost text-only nodes."""
    
    # Inner pipeline
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="result")
    def add_ten(doubled: int) -> int:
        return doubled + 10
    
    inner = Pipeline(nodes=[double, add_ten])
    
    # Outer pipeline with collapsed inner pipeline
    @node(output_name="final")
    def process_result(result: int) -> int:
        return result * 100
    
    outer = Pipeline(nodes=[inner.as_node(), process_result])
    
    # Get visualization with collapsed pipeline (depth=1, no expansion)
    walker = GraphWalker(outer, expanded_nodes=set(), group_inputs=False)
    graph_data = walker.get_visualization_data()
    
    # Get all node IDs from nodes list
    node_ids = {n.id for n in graph_data.nodes}
    
    # Get all node IDs referenced in edges
    edge_node_ids = set()
    for edge in graph_data.edges:
        edge_node_ids.add(edge.source)
        edge_node_ids.add(edge.target)
    
    # Check: all edge nodes should exist in nodes list
    ghost_nodes = edge_node_ids - node_ids
    
    print("\n📊 Visualization Statistics:")
    print(f"   Nodes in nodes list: {len(node_ids)}")
    print(f"   Nodes in edges: {len(edge_node_ids)}")
    print(f"   Ghost nodes (in edges but not in nodes): {len(ghost_nodes)}")
    
    if ghost_nodes:
        print("\n👻 GHOST NODES FOUND:")
        for ghost in ghost_nodes:
            print(f"   - {ghost}")
            # Find edges using this ghost node
            for edge in graph_data.edges:
                if edge.source == ghost or edge.target == ghost:
                    print(f"     Edge: {edge.source} -> {edge.target}")
    else:
        print("\n✅ SUCCESS: No ghost nodes!")
    
    # Print actual nodes for debugging
    print("\n📝 All nodes in visualization:")
    for n in graph_data.nodes:
        print(f"   - {n.id} ({type(n).__name__})")
    
    # Print all edges
    print("\n🔗 All edges:")
    for edge in graph_data.edges:
        print(f"   - {edge.source} -> {edge.target}")
    
    assert len(ghost_nodes) == 0, f"Found ghost nodes: {ghost_nodes}"


def test_no_ghost_nodes_with_output_mapping():
    """Test collapsed pipeline with output remapping doesn't create ghosts."""
    
    @node(output_name="cleaned")
    def clean_text(text: str) -> str:
        return text.strip()
    
    inner = Pipeline(nodes=[clean_text])
    
    @node(output_name="final")
    def uppercase(processed: str) -> str:
        return processed.upper()
    
    # Map inner "cleaned" to outer "processed"
    mapped = inner.as_node(output_mapping={"cleaned": "processed"})
    outer = Pipeline(nodes=[mapped, uppercase])
    
    # Get visualization with collapsed pipeline
    walker = GraphWalker(outer, expanded_nodes=set(), group_inputs=False)
    graph_data = walker.get_visualization_data()
    
    # Check for ghost nodes
    node_ids = {n.id for n in graph_data.nodes}
    edge_node_ids = set()
    for edge in graph_data.edges:
        edge_node_ids.add(edge.source)
        edge_node_ids.add(edge.target)
    
    ghost_nodes = edge_node_ids - node_ids
    
    assert len(ghost_nodes) == 0, f"Found ghost nodes with output_mapping: {ghost_nodes}"
