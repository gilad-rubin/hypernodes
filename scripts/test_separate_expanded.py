"""Test visualization with separate outputs and expanded pipeline."""

from hypernodes import Pipeline, node
from hypernodes.viz.js.html_generator import generate_widget_html
from hypernodes.viz.js.renderer import JSRenderer
from hypernodes.viz.ui_handler import UIHandler


@node(output_name="query")
def extract_query(eval_pair: dict) -> str:
    return eval_pair.get("query", "")


@node(output_name="documents")
def retrieve(query: str, num_results: int = 5) -> list[dict[str, str]]:
    return [{"text": f"Doc {i}"} for i in range(num_results)]


@node(output_name="answer")
def generate_answer(
    query: str, documents: list[dict[str, str]], model_name: str
) -> str:
    return f"Answer from {model_name}"


@node(output_name=("score", "details"))
def evaluate(answer: str, expected: str) -> tuple[float, dict[str, any]]:
    return 0.9, {"match": True}


retrieval = Pipeline(nodes=[retrieve], name="retrieval")
rag = Pipeline(
    nodes=[extract_query, retrieval.as_node(name="retrieval_step"), generate_answer],
    name="rag",
)
eval_pipeline = Pipeline(
    nodes=[rag.as_node(name="rag_pipeline"), evaluate], name="evaluation"
)

# Generate with expanded pipeline + separate outputs
handler = UIHandler(eval_pipeline, depth=99)
graph_data = handler.get_visualization_data(traverse_collapsed=True)
renderer = JSRenderer()
rf_data = renderer.render(
    graph_data,
    theme="dark",
    separate_outputs=False,
    show_types=True,
)

# Check edges
print("=== Edges involving 'answer' or 'generate_answer' ===")
for edge in rf_data["edges"]:
    if (
        "answer" in edge["source"]
        or "answer" in edge["target"]
        or "generate_answer" in edge["source"]
        or "evaluate" in edge["target"]
    ):
        print(f"  {edge['source']} -> {edge['target']}")

html = generate_widget_html(rf_data)
with open("outputs/test_separate_expanded.html", "w") as f:
    f.write(html)
print("\nGenerated: outputs/test_separate_expanded.html")
