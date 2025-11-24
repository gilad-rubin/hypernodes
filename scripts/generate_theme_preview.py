import os
import sys

# Add src to path
sys.path.append(os.path.abspath("src"))

from hypernodes import Pipeline, node
from hypernodes.viz.graph_walker import GraphWalker
from hypernodes.viz.graphviz.renderer import GraphvizRenderer
from hypernodes.viz.graphviz.style import DESIGN_STYLES


# Define a sample pipeline
@node(output_name="raw_data")
def load_data():
    return [1, 2, 3]


@node(output_name="cleaned")
def clean(data):
    return data


@node(output_name="processed")
def process(data):
    return [x * 2 for x in data]


sub_pipeline = Pipeline(nodes=[process])
# Note: We are using the pipeline object directly in the node list,
# but wrapped by .as_node() happens implicitly or explicitly.
# Here we do it explicitly.
sub_node = sub_pipeline.as_node()


@node(output_name="result")
def analyze(data):
    return sum(data)


# Create a pipeline with nested structure
main_pipeline = Pipeline(nodes=[load_data, clean, sub_node, analyze])

# --- CORRECT WAY TO GET IDs FOR EXPANSION ---
# 1. Run walker once to get the viz graph
walker_initial = GraphWalker(
    main_pipeline, expanded_nodes=set(), traverse_collapsed=True
)
viz_graph = walker_initial.get_visualization_data()

# 2. Find the PipelineNode in the viz graph
expanded_set = set()
for viz_node in viz_graph.nodes:
    # Visualization nodes have 'is_expanded' attribute if they are PipelineNodes
    # We check if it's a PipelineNode in the VIZ structure
    if hasattr(viz_node, "is_expanded"):
        expanded_set.add(viz_node.id)

# 3. Run walker again with expansion
walker = GraphWalker(
    main_pipeline, expanded_nodes=expanded_set, traverse_collapsed=True
)
graph_data = walker.get_visualization_data()

html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>HyperNodes Graphviz Themes</title>
    <style>
        body { font-family: sans-serif; padding: 20px; background: #f0f0f0; }
        .container { display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; width: 400px; }
        h2 { margin-top: 0; color: #333; text-transform: capitalize; font-size: 1.2em; margin-bottom: 10px; }
        svg { max-width: 100%; height: auto; }
        .color-chips { display: flex; justify-content: center; gap: 15px; margin-bottom: 15px; padding: 10px; background: #f8f9fa; border-radius: 4px; }
        .chip-container { display: flex; flex-direction: column; align-items: center; gap: 4px; }
        .chip { width: 30px; height: 30px; border-radius: 50%; border: 2px solid; }
        .chip-label { font-size: 0.75em; color: #666; font-weight: bold; }
        .hex-code { font-size: 0.7em; color: #999; font-family: monospace; }
    </style>
</head>
<body>
    <h1>Theme Options for HyperNodes</h1>
    <p>These themes are now available in <code>DESIGN_STYLES</code>. Default theme uses your selected colors.</p>
    <p><strong>Note:</strong> Nested pipeline borders should now be bolder (2.5 width).</p>
    <div class="container">
"""

# Iterate over available styles
for name, theme in DESIGN_STYLES.items():
    renderer = GraphvizRenderer(style=theme)
    svg = renderer.render(graph_data)

    # Extract colors for display
    func_style = theme.node_styles["function"]
    pipe_style = theme.node_styles["pipeline"]
    data_style = theme.node_styles["data"]

    html_content += f"""
        <div class="card">
            <h2>{name}</h2>
            <div class="color-chips">
                <div class="chip-container">
                    <span class="chip-label">Func</span>
                    <div class="chip" style="background: {func_style.color.fill}; border-color: {func_style.color.outline};"></div>
                    <span class="hex-code">{func_style.color.fill}</span>
                </div>
                <div class="chip-container">
                    <span class="chip-label">Pipe</span>
                    <div class="chip" style="background: {pipe_style.color.fill}; border-color: {pipe_style.color.outline};"></div>
                    <span class="hex-code">{pipe_style.color.fill}</span>
                </div>
                <div class="chip-container">
                    <span class="chip-label">Input</span>
                    <div class="chip" style="background: {data_style.color.fill}; border-color: {data_style.color.outline};"></div>
                    <span class="hex-code">{data_style.color.fill}</span>
                </div>
            </div>
            {svg}
        </div>
    """

html_content += """
    </div>
</body>
</html>
"""

with open("theme_preview.html", "w") as f:
    f.write(html_content)

print(f"Generated theme_preview.html with {len(DESIGN_STYLES)} options.")
