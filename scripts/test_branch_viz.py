"""Test script for branch node visualization."""

from hypernodes import Pipeline, branch, node
from hypernodes.viz.graph_walker import GraphWalker
from hypernodes.viz.js.renderer import JSRenderer
from hypernodes.viz.js.html_generator import generate_widget_html

# Example with longer function names to test text wrapping
@node(output_name="confidence_score")
def calculate_confidence(data: dict) -> float:
    return data.get("confidence", 0.5)

@node(output_name="analysis_result")
def process_high_confidence(confidence_score: float) -> str:
    return f"High confidence: {confidence_score}"

@node(output_name="analysis_result")
def process_low_confidence(confidence_score: float) -> str:
    return f"Low confidence: {confidence_score}"

@branch(when_true=process_high_confidence, when_false=process_low_confidence)
def should_proceed(confidence_score: float) -> bool:
    """Check if confidence is above threshold."""
    return confidence_score > 0.7

pipeline = Pipeline(
    nodes=[calculate_confidence, should_proceed, process_high_confidence, process_low_confidence]
)

# Generate visualization - light theme
walker = GraphWalker(pipeline, expanded_nodes=set())
graph = walker.get_visualization_data()

renderer = JSRenderer()
rf_data_light = renderer.render(graph, theme="light", separate_outputs=False)
html_light = generate_widget_html(rf_data_light)

output_path_light = "outputs/test_branch_viz_light.html"
with open(output_path_light, "w") as f:
    f.write(html_light)
print(f"Generated: {output_path_light}")

