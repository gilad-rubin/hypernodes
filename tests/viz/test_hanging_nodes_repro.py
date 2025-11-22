
import pytest
from hypernodes import node, Pipeline
from hypernodes.viz.ui_handler import UIHandler

@node(output_name="docs")
def vector_store(query: str):
    return ["doc1", "doc2"]

@node(output_name="top_docs")
def top_k(docs: list, k: int = 2):
    return docs[:k]

@node(output_name="answer")
def llm(query: str, top_docs: list):
    return "answer"

@node(output_name="score")
def judge(query: str, answer: str, ground_truth: str):
    return 1.0

def test_nested_pipeline_parentage():
    # Retrieval pipeline
    retrieval = Pipeline(nodes=[vector_store, top_k, llm], name="RetrievalPipeline")

    # Evaluation pipeline
    retrieval_node = retrieval.as_node()
    evaluation = Pipeline(nodes=[retrieval_node, judge], name="EvaluationPipeline")

    # Metrics pipeline
    # Use map_over to match user scenario
    # User passes "eval_pairs" list.
    # Evaluation pipeline takes "eval_pair" (singular).
    # So metrics maps over "eval_pairs" -> "eval_pair".
    
    evaluation_node = evaluation.as_node(
        map_over="eval_pairs",
        input_mapping={"eval_pair": "eval_pairs"}, # Just dummy mapping
        name="EvalNode"
    )
    # Assume 'evaluation' pipeline inputs match 'eval_pair' somehow, 
    # or we just bind/ignore inputs for viz test.
    
    metrics = Pipeline(nodes=[evaluation_node], name="MetricsPipeline")
    
    # Helper to find node by label
    def find_node(nodes, label):
        for n in nodes:
            if hasattr(n, 'label') and n.label == label:
                return n
            if hasattr(n, 'function_name') and n.function_name == label:
                return n
            if hasattr(n, 'name') and n.name == label:
                return n
        return None

    # Helper to find node by ID
    def find_node_by_id(nodes, nid):
        for n in nodes:
            if n.id == nid:
                return n
        return None

    # --- Depth 2 ---
    # Evaluation expanded. Judge visible.
    handler = UIHandler(metrics, depth=2)
    data = handler.get_visualization_data()
    
    print("Depth 2 Nodes:", [f"{n.id} ({getattr(n, 'label', 'no-label')}) parent={n.parent_id}" for n in data.nodes])
    print("Depth 2 Edges:", [f"{e.source} -> {e.target}" for e in data.edges])

    # Find the top-level expanded node (Evaluation Node)
    # Note: metrics pipeline nodes are at top level (parent=None)
    eval_node = next(n for n in data.nodes if n.parent_id is None and hasattr(n, 'is_expanded') and n.is_expanded)
    
    judge_node = find_node(data.nodes, "judge")
    assert judge_node is not None
    
    # Check Judge parent
    assert judge_node.parent_id == eval_node.id, \
        f"Depth 2: Judge parent should be Evaluation ({eval_node.id}), but is {judge_node.parent_id}"

    # --- Depth 3 ---
    # Retrieval expanded.
    handler = UIHandler(metrics, depth=3)
    data = handler.get_visualization_data()
    
    print("Depth 3 Nodes:", [f"{n.id} ({getattr(n, 'label', 'no-label')}) parent={n.parent_id} expanded={getattr(n, 'is_expanded', False)}" for n in data.nodes])
    
    # 1. Evaluation Node (Top Level, Expanded)
    # Since 'metrics' pipeline wrapper is not shown as a node itself, 
    # the first node inside it (evaluation_node) is at parent=None.
    eval_viz_node = next(n for n in data.nodes if n.parent_id is None and hasattr(n, 'is_expanded') and n.is_expanded)
    print(f"Evaluation Viz Node: {eval_viz_node.id}")
    
    # 2. Retrieval Node (Child of Eval, Expanded)
    retrieval_viz_node = next((n for n in data.nodes if n.parent_id == eval_viz_node.id and hasattr(n, 'is_expanded') and n.is_expanded), None)
    
    assert retrieval_viz_node is not None, "Retrieval node should be present and expanded at depth 3"
    print(f"Retrieval Viz Node: {retrieval_viz_node.id}")
    
    # 3. Vector Store (Child of Retrieval)
    vs_node = find_node(data.nodes, "vector_store")
    assert vs_node is not None
    
    print(f"Vector Store Parent ID: {vs_node.parent_id}")
    
    assert vs_node.parent_id == retrieval_viz_node.id, \
        f"VectorStore parent should be Retrieval ({retrieval_viz_node.id}), but is {vs_node.parent_id}"

if __name__ == "__main__":
    try:
        test_nested_pipeline_parentage()
        print("Test Passed!")
    except AssertionError as e:
        print(f"Test Failed: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

