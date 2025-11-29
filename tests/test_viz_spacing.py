import pytest
from hypernodes.node import Node
from hypernodes.pipeline import Pipeline
from hypernodes.viz.ui_handler import UIHandler
from hypernodes.viz.js.renderer import JSRenderer
from hypernodes.viz.js.html_generator import generate_widget_html

def test_viz_spacing_height_calculation():
    """
    Test that the generated HTML contains the correct height calculation formula
    for function nodes with combined outputs.
    """
    def simple_func(a: int) -> int:
        return a
        
    node = Node(simple_func, output_name="out")
    pipeline = Pipeline([node])
    
    # Generate visualization
    handler = UIHandler(pipeline, depth=1, show_output_types=True)
    graph_data = handler.get_visualization_data(traverse_collapsed=True)
    
    renderer = JSRenderer()
    react_flow_data = renderer.render(
        graph_data,
        separate_outputs=False,
        show_types=True
    )
    
    html_content = generate_widget_html(react_flow_data)
    
    # Check for the updated height formula
    # We look for the string "height = 40 + (n.data.outputs.length * 24);"
    assert "height = 40 + (n.data.outputs.length * 24)" in html_content
    
    # Also check for the gap-2 class in OutputsSection
    assert 'className="flex flex-col items-start gap-2"' in html_content

def test_viz_spacing_width_calculation():
    """
    Test that the generated HTML contains the dynamic width calculation logic.
    """
    def simple_func(a: int) -> int:
        return a
        
    node = Node(simple_func, output_name="out")
    pipeline = Pipeline([node])
    
    handler = UIHandler(pipeline, depth=1, show_output_types=True)
    graph_data = handler.get_visualization_data(traverse_collapsed=True)
    
    renderer = JSRenderer()
    react_flow_data = renderer.render(
        graph_data,
        separate_outputs=False,
        show_types=True
    )
    
    html_content = generate_widget_html(react_flow_data)
    
    # Check for width calculation logic
    assert "let calculatedWidth = Math.max(180, labelLen * 8 + 80);" in html_content
    assert "const requiredOutputWidth = (maxOutputLen * 7) + 50;" in html_content

def test_viz_spacing_collapsed_pipeline_height():
    """
    Test that collapsed pipelines with combined outputs use the correct height formula.
    """
    def inner_func(a: int) -> int:
        return a
        
    node = Node(inner_func, output_name="inner_out")
    inner_pipeline = Pipeline([node])
    
    # Wrap as a node
    pipeline_node = inner_pipeline.as_node(name="InnerPipeline")
    
    outer_pipeline = Pipeline([pipeline_node])
    
    handler = UIHandler(outer_pipeline, depth=0, show_output_types=True) # depth=0 to collapse
    graph_data = handler.get_visualization_data(traverse_collapsed=False)
    
    renderer = JSRenderer()
    react_flow_data = renderer.render(
        graph_data,
        separate_outputs=False,
        show_types=True
    )
    
    html_content = generate_widget_html(react_flow_data)
    
    # Check for the height formula in the collapsed pipeline block
    # We look for the specific block structure or just the formula presence 
    # since we added it to a new location.
    # To be safe, we can check if the formula appears twice in the file now 
    # (once for standard nodes, once for pipelines)
    assert html_content.count("height = 40 + (n.data.outputs.length * 24);") == 2
