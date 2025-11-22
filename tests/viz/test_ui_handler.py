import pytest
from hypernodes.pipeline import Pipeline
from hypernodes.node import Node
from hypernodes.viz.ui_handler import UIHandler
from hypernodes.viz.structures import PipelineNode

def simple_func(a: int) -> int:
    return a + 1

def test_ui_handler_initialization():
    node1 = Node(simple_func, output_name="b")
    pipeline = Pipeline([node1])
    
    handler = UIHandler(pipeline, depth=1)
    graph = handler.get_visualization_data()
    
    assert len(graph.nodes) > 0
    assert handler.depth == 1

def test_ui_handler_expansion():
    node1 = Node(simple_func, output_name="b")
    inner_pipeline = Pipeline([node1], name="inner")
    outer_node = inner_pipeline.as_node(name="nested")
    outer_pipeline = Pipeline([outer_node])
    
    handler = UIHandler(outer_pipeline, depth=1) # Collapsed by default
    graph = handler.get_visualization_data()
    
    # Find pipeline node
    pipeline_nodes = [n for n in graph.nodes if isinstance(n, PipelineNode)]
    assert len(pipeline_nodes) == 1
    assert not pipeline_nodes[0].is_expanded
    
    node_id = pipeline_nodes[0].id
    
    # Expand
    handler.expand_node(node_id)
    graph = handler.get_visualization_data()
    
    pipeline_nodes = [n for n in graph.nodes if isinstance(n, PipelineNode)]
    assert len(pipeline_nodes) == 1
    assert pipeline_nodes[0].is_expanded
    assert node_id in handler.expanded_nodes

def test_ui_handler_collapse():
    node1 = Node(simple_func, output_name="b")
    inner_pipeline = Pipeline([node1], name="inner")
    outer_node = inner_pipeline.as_node(name="nested")
    outer_pipeline = Pipeline([outer_node])
    
    # Start fully expanded
    handler = UIHandler(outer_pipeline, depth=None) 
    graph = handler.get_visualization_data()
    
    pipeline_nodes = [n for n in graph.nodes if isinstance(n, PipelineNode)]
    assert len(pipeline_nodes) == 1
    assert pipeline_nodes[0].is_expanded
    
    node_id = pipeline_nodes[0].id
    
    # Collapse
    handler.collapse_node(node_id)
    graph = handler.get_visualization_data()
    
    pipeline_nodes = [n for n in graph.nodes if isinstance(n, PipelineNode)]
    assert len(pipeline_nodes) == 1
    assert not pipeline_nodes[0].is_expanded
    assert node_id not in handler.expanded_nodes

def test_ui_handler_toggle():
    node1 = Node(simple_func, output_name="b")
    inner_pipeline = Pipeline([node1], name="inner")
    outer_node = inner_pipeline.as_node(name="nested")
    outer_pipeline = Pipeline([outer_node])
    
    handler = UIHandler(outer_pipeline, depth=1)
    graph = handler.get_visualization_data()
    
    pipeline_nodes = [n for n in graph.nodes if isinstance(n, PipelineNode)]
    node_id = pipeline_nodes[0].id
    
    # Toggle ON
    handler.toggle_node(node_id)
    assert node_id in handler.expanded_nodes
    
    # Toggle OFF
    handler.toggle_node(node_id)
    assert node_id not in handler.expanded_nodes
