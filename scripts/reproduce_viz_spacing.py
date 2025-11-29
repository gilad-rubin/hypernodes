from hypernodes.node import Node
from hypernodes.pipeline import Pipeline
from typing import List, Dict, Any

def func_1_out(a: int) -> int:
    return a

def func_2_out(a: int) -> (int, int):
    return a, a

def func_3_out(a: int) -> (int, int, int):
    return a, a, a

# Create nodes with varying outputs
n1 = Node(func_1_out, output_name="o1")
n2 = Node(func_2_out, output_name=("o2a", "o2b"))
n3 = Node(func_3_out, output_name=("o3a", "o3b", "o3c"))

# Create a pipeline
pipeline = Pipeline([n1, n2, n3])

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

# Inject debug flag
html_content = html_content.replace(
    'const initialDebugOverlays = debugParam === \'overlays\' || debugParam === \'true\' || debugParam === \'1\';',
    'const initialDebugOverlays = true;'
)

output_path = "outputs/viz_spacing_test.html"
with open(output_path, "w") as f:
    f.write(html_content)

print(f"Visualization generated at {output_path}")
