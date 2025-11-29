"""Test visualization with initially collapsed pipeline."""
from hypernodes import Pipeline, node

@node(output_name="query")
def extract_query(eval_pair: dict) -> str:
    return eval_pair.get("query", "")

@node(output_name="documents")
def retrieve(query: str, num_results: int = 5) -> list[dict[str, str]]:
    return [{"text": f"Doc {i}"} for i in range(num_results)]

@node(output_name="answer")
def generate_answer(query: str, documents: list[dict[str, str]], model_name: str) -> str:
    return f"Answer from {model_name}"

@node(output_name=("score", "details"))
def evaluate(answer: str, expected: str) -> tuple[float, dict[str, any]]:
    return 0.9, {"match": True}

# Create nested pipeline
retrieval = Pipeline(nodes=[retrieve], name="retrieval")
rag = Pipeline(
    nodes=[extract_query, retrieval.as_node(name="retrieval_step"), generate_answer],
    name="rag"
)
eval_pipeline = Pipeline(
    nodes=[rag.as_node(name="rag_pipeline"), evaluate],
    name="evaluation"
)

# Use UIHandler and JSRenderer directly to generate HTML
from hypernodes.viz.ui_handler import UIHandler
from hypernodes.viz.js.renderer import JSRenderer
from hypernodes.viz.js.html_generator import generate_widget_html

def save_html(pipeline, output_path, depth, separate_outputs):
    handler = UIHandler(pipeline, depth=depth)
    graph_data = handler.get_visualization_data(traverse_collapsed=True)
    renderer = JSRenderer()
    rf_data = renderer.render(graph_data, theme="dark", separate_outputs=separate_outputs, show_types=True)
    html = generate_widget_html(rf_data)
    with open(output_path, 'w') as f:
        f.write(html)
    return html

# Generate with depth=1 (collapsed)
save_html(eval_pipeline, "outputs/test_collapsed.html", depth=1, separate_outputs=False)
print("Generated: outputs/test_collapsed.html (depth=1, combined)")

# Generate with separate outputs
save_html(eval_pipeline, "outputs/test_collapsed_separate.html", depth=1, separate_outputs=True)
print("Generated: outputs/test_collapsed_separate.html (depth=1, separate)")

# Generate with all expanded
save_html(eval_pipeline, "outputs/test_expanded.html", depth=99, separate_outputs=False)
print("Generated: outputs/test_expanded.html (depth=99, combined)")
