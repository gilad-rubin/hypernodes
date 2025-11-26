import pytest
from hypernodes.pipeline import Pipeline
from hypernodes.node import Node
from hypernodes.viz.graph_walker import GraphWalker
from hypernodes.viz.structures import (
    DataNode,
    DualNode,
    FunctionNode,
    GroupDataNode,
    PipelineNode,
    VizEdge,
    VisualizationGraph,
)

def simple_func(a: int) -> int:
    return a + 1

def dual_func(a: int) -> int:
    return a * 2

def test_graph_walker_simple_pipeline():
    node1 = Node(simple_func, output_name="b")
    pipeline = Pipeline([node1])
    
    walker = GraphWalker(pipeline, expanded_nodes=set())
    graph = walker.get_visualization_data()
    
    assert len(graph.nodes) > 0
    assert any(isinstance(n, FunctionNode) and n.function_name == "simple_func" for n in graph.nodes)
    
    # Check for DataNodes (input 'a' and output)
    data_nodes = [n for n in graph.nodes if isinstance(n, DataNode)]
    assert len(data_nodes) >= 2
    assert any(n.name == "a" for n in data_nodes)

def test_graph_walker_nested_pipeline_collapsed():
    node1 = Node(simple_func, output_name="b")
    inner_pipeline = Pipeline([node1], name="inner")
    
    outer_node = inner_pipeline.as_node(name="nested")
    outer_pipeline = Pipeline([outer_node])
    
    walker = GraphWalker(outer_pipeline, expanded_nodes=set())
    graph = walker.get_visualization_data()
    
    # Should see a PipelineNode, not the inner function
    pipeline_nodes = [n for n in graph.nodes if isinstance(n, PipelineNode)]
    assert len(pipeline_nodes) == 1
    assert pipeline_nodes[0].label == "nested"
    assert not pipeline_nodes[0].is_expanded
    
    # Should NOT see inner function
    assert not any(isinstance(n, FunctionNode) and n.function_name == "simple_func" for n in graph.nodes)

def test_graph_walker_nested_pipeline_expanded():
    node1 = Node(simple_func, output_name="b")
    inner_pipeline = Pipeline([node1], name="inner")
    
    outer_node = inner_pipeline.as_node(name="nested")
    outer_pipeline = Pipeline([outer_node])
    
    # Get ID to expand - uses human-readable label now
    node_id = "nested"
    
    walker = GraphWalker(outer_pipeline, expanded_nodes={node_id})
    graph = walker.get_visualization_data()
    
    # Should see PipelineNode expanded
    pipeline_nodes = [n for n in graph.nodes if isinstance(n, PipelineNode)]
    assert len(pipeline_nodes) == 1
    assert pipeline_nodes[0].is_expanded
    
    # Should see inner function now
    assert any(isinstance(n, FunctionNode) and n.function_name == "simple_func" for n in graph.nodes)

def test_graph_walker_dual_node():
    node1 = Node(dual_func, output_name="b")
    node1.is_dual_node = True # Simulate dual node property
    pipeline = Pipeline([node1])
    
    walker = GraphWalker(pipeline, expanded_nodes=set())
    graph = walker.get_visualization_data()
    
    # Should see DualNode
    assert any(isinstance(n, DualNode) for n in graph.nodes)

def test_graph_walker_group_inputs():
    # Create a scenario where multiple inputs come from same source or are bound
    node1 = Node(simple_func, output_name="b")
    pipeline = Pipeline([node1]).bind(a=1)
    
    # Force multiple bound inputs to test grouping logic if possible
    # Or just test that bound inputs are created
    
    walker = GraphWalker(pipeline, expanded_nodes=set(), group_inputs=True)
    graph = walker.get_visualization_data()
    
    # Check for bound DataNode
    bound_nodes = [n for n in graph.nodes if isinstance(n, DataNode) and n.is_bound]
    assert len(bound_nodes) > 0
    assert bound_nodes[0].name == "a"
