from hypernodes import Pipeline, node
from hypernodes.pipeline_node import PipelineNode as HyperPipelineNode
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

# Check the rag_pipeline node
rag_node = eval_pipeline.nodes[0]
print(f"rag_node type: {type(rag_node)}")
print(f"rag_node is HyperPipelineNode: {isinstance(rag_node, HyperPipelineNode)}")
print(f"rag_node.root_args: {rag_node.root_args}")
print(f"rag_node.input_mapping: {getattr(rag_node, 'input_mapping', None)}")

# Check inner pipeline
print(f"\nrag_node.pipeline: {rag_node.pipeline}")
print(f"rag_node.pipeline.nodes: {rag_node.pipeline.nodes}")

# Check if we can find the function that uses eval_pair
print("\n=== Looking for function that uses eval_pair ===")
if hasattr(rag_node.pipeline, "graph"):
    for inner_node in rag_node.pipeline.graph.execution_order:
        print(f"  {inner_node.name}: root_args={getattr(inner_node, 'root_args', None)}")
        if hasattr(inner_node, "root_args") and "eval_pair" in inner_node.root_args:
            print(f"    FOUND! This node uses eval_pair")
            if hasattr(inner_node, "func"):
                hints = get_type_hints(inner_node.func)
                print(f"    Type hints: {hints}")
