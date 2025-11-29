from hypernodes import Pipeline, node
from hypernodes.pipeline_node import PipelineNode as HyperPipelineNode
from hypernodes.viz.graph_walker import GraphWalker
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

retrieval = Pipeline(nodes=[retrieve], name="retrieval")
rag = Pipeline(
    nodes=[extract_query, retrieval.as_node(name="retrieval_step"), generate_answer],
    name="rag"
)
eval_pipeline = Pipeline(
    nodes=[rag.as_node(name="rag_pipeline"), evaluate],
    name="evaluation"
)

# Test the _extract_input_type method directly
walker = GraphWalker(eval_pipeline, expanded_nodes=set(), traverse_collapsed=True)

rag_node = eval_pipeline.nodes[0]
print("Testing _extract_input_type on rag_node:")
for arg in ["eval_pair", "model_name", "num_results"]:
    result = walker._extract_input_type(rag_node, arg)
    print(f"  {arg}: {result}")

print("\nTesting _extract_input_type on extract_query:")
result = walker._extract_input_type(extract_query, "eval_pair")
print(f"  eval_pair: {result}")
