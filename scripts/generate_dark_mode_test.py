import os
import sys

# Add src to path
sys.path.append(os.path.abspath("src"))

from hypernodes import Pipeline, node
from hypernodes.viz.graph_walker import GraphWalker
from hypernodes.viz.graphviz.renderer import GraphvizRenderer


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
sub_node = sub_pipeline.as_node()


@node(output_name="result")
def analyze(data):
    return sum(data)


# Create a pipeline with nested structure
main_pipeline = Pipeline(nodes=[load_data, clean, sub_node, analyze])

# Get IDs for expansion
walker_initial = GraphWalker(
    main_pipeline, expanded_nodes=set(), traverse_collapsed=True
)
viz_graph = walker_initial.get_visualization_data()

expanded_set = set()
for viz_node in viz_graph.nodes:
    if hasattr(viz_node, "is_expanded"):
        expanded_set.add(viz_node.id)

walker = GraphWalker(
    main_pipeline, expanded_nodes=expanded_set, traverse_collapsed=True
)
graph_data = walker.get_visualization_data()

# Render SVG
renderer = GraphvizRenderer(style="default")
svg = renderer.render(graph_data)

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Dark Mode Visibility Test</title>
    <style>
        body {{ font-family: sans-serif; padding: 20px; margin: 0; }}
        .container {{ display: flex; gap: 30px; flex-wrap: wrap; }}
        .panel {{ flex: 1; min-width: 400px; padding: 30px; border-radius: 12px; }}
        .light {{ background: #ffffff; border: 1px solid #e0e0e0; }}
        .dark {{ background: #1e1e1e; }}
        .vscode-dark {{ background: #252526; }}
        h1 {{ color: #333; }}
        h2 {{ margin-top: 0; font-size: 1.2em; }}
        .light h2 {{ color: #333; }}
        .dark h2, .vscode-dark h2 {{ color: #cccccc; }}
        svg {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <h1>Dark Mode Visibility Test</h1>
    <p>Testing nested pipeline outline and label visibility across backgrounds:</p>
    <div class="container">
        <div class="panel light">
            <h2>Light Background (#ffffff)</h2>
            {svg}
        </div>
        <div class="panel dark">
            <h2>Dark Background (#1e1e1e)</h2>
            {svg}
        </div>
        <div class="panel vscode-dark">
            <h2>VS Code Dark (#252526)</h2>
            {svg}
        </div>
    </div>
</body>
</html>
"""

with open("dark_mode_test.html", "w") as f:
    f.write(html_content)

print("Generated dark_mode_test.html")

