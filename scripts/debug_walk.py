from hypernodes import Pipeline, node
from hypernodes.pipeline_node import PipelineNode as HyperPipelineNode
from hypernodes.viz.graph_walker import GraphWalker
from hypernodes.viz.structures import DataNode
from typing import get_type_hints

# Patch to add debug output
original_handle_input = GraphWalker._handle_input_connections

def debug_handle_input(self, node, node_id, prefix, nodes_out, edges_out, scope):
    print(f"=== _handle_input_connections called ===")
    print(f"  node type: {type(node).__name__}")
    print(f"  node_id: {node_id}")
    if hasattr(node, "root_args"):
        for arg in node.root_args:
            if arg not in scope:
                type_hint = self._extract_input_type(node, arg)
                print(f"  Input {arg}: type_hint={type_hint}")
    return original_handle_input(self, node, node_id, prefix, nodes_out, edges_out, scope)

GraphWalker._handle_input_connections = debug_handle_input

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

walker = GraphWalker(eval_pipeline, expanded_nodes=set(), traverse_collapsed=True)
graph = walker.get_visualization_data()

print("\n=== Final DataNodes ===")
for n in graph.nodes:
    if isinstance(n, DataNode):
        print(f"{n.id}: type_hint={n.type_hint}")
