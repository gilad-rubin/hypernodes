from hypernodes import Pipeline, node
from typing import get_type_hints

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

# Check type hints
print("=== Function type hints ===")
for fn in [extract_query, retrieve, generate_answer, evaluate]:
    try:
        hints = get_type_hints(fn.func)
        print(f"{fn.name}: {hints}")
    except Exception as e:
        print(f"{fn.name}: ERROR - {e}")

# Create pipeline and check what graph walker produces
retrieval = Pipeline(nodes=[retrieve], name="retrieval")
rag = Pipeline(
    nodes=[extract_query, retrieval.as_node(name="retrieval_step"), generate_answer],
    name="rag"
)
eval_pipeline = Pipeline(
    nodes=[rag.as_node(name="rag_pipeline"), evaluate],
    name="evaluation"
)

from hypernodes.viz.graph_walker import GraphWalker

walker = GraphWalker(eval_pipeline, expanded_nodes=set(), traverse_collapsed=True)
graph = walker.get_visualization_data()

print("\n=== DataNodes with type hints ===")
for node in graph.nodes:
    if hasattr(node, 'type_hint'):
        print(f"{node.id}: type_hint={node.type_hint}, is_bound={getattr(node, 'is_bound', None)}, source_id={getattr(node, 'source_id', None)}")
