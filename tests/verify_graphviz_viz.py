
import os
from hypernodes import node, Pipeline
from hypernodes.viz.ui_handler import UIHandler
from hypernodes.viz.graphviz.renderer import GraphvizRenderer

# Mock types
class VectorStore: pass
class LLM: pass
class EvaluationPair: pass
class JudgeEvaluator: pass
class Document: pass
class EvaluationResult: pass

@node(output_name="retrieved_docs")
def retrieve(query: str, vector_store: VectorStore, top_k: int = 2) -> list:
    return []

@node(output_name="answer")
def generate(query: str, retrieved_docs: list, llm: LLM) -> str:
    return "answer"

retrieval = Pipeline([retrieve, generate])

# Generate Graphviz HTML
print("Generating Graphviz visualization HTML...")
handler = UIHandler(retrieval)
graph_data = handler.get_visualization_data()

# Use 'default' style which should be our updated AUTO_THEME
renderer = GraphvizRenderer(style="default") 
html_content = renderer.render(graph_data)

output_path = "tests/viz_graphviz_output.html"
with open(output_path, "w") as f:
    f.write(html_content)

print(f"HTML written to {output_path}")

