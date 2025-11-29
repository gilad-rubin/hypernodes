from hypernodes.node import Node
from hypernodes.pipeline import Pipeline
from typing import List, Dict, Any

def complex_function(a: int) -> (List[Dict[str, Any]], str):
    """A function with complex type hints."""
    return [{"a": 1}], "done"

# Create a node
node = Node(complex_function, output_name=("data", "status"))

# Create a pipeline
pipeline = Pipeline([node])

# Visualize using JS renderer manually
from hypernodes.viz.ui_handler import UIHandler
from hypernodes.viz.js.renderer import JSRenderer
from hypernodes.viz.js.html_generator import generate_widget_html

print("Generating visualization data...")
handler = UIHandler(pipeline, depth=1, show_output_types=True)
graph_data = handler.get_visualization_data(traverse_collapsed=True)

print("Rendering to React Flow format...")
renderer = JSRenderer()
react_flow_data = renderer.render(
    graph_data,
    separate_outputs=False,
    show_types=True
)

print("Generating HTML...")
html_content = generate_widget_html(react_flow_data)

output_path = "outputs/viz_spacing_test.html"
with open(output_path, "w") as f:
    f.write(html_content)

print(f"Visualization generated at {output_path}")
