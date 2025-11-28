"""Comprehensive test for JS visualization issues."""
import os
from hypernodes import Pipeline, node
from hypernodes.viz.ui_handler import UIHandler
from hypernodes.viz.js.renderer import JSRenderer
from hypernodes.viz.js.html_generator import generate_widget_html


# Nodes with various type hints (some long)
@node(output_name="cleaned")
def clean_text(passage: str) -> str:
    """Clean text."""
    return passage.strip().lower()


@node(output_name="query")
def extract_query(eval_pair: dict) -> str:
    """Extract query from eval pair."""
    return eval_pair.get("query", "")


@node(output_name="documents")
def retrieve(query: str, num_results: int = 5) -> list[dict[str, str]]:
    """Retrieve documents."""
    return [{"text": f"Doc {i}"} for i in range(num_results)]


@node(output_name="answer")
def generate_answer(query: str, documents: list[dict[str, str]], model_name: str) -> str:
    """Generate answer using LLM."""
    return f"Answer from {model_name}"


@node(output_name=("score", "details"))
def evaluate(answer: str, expected: str) -> tuple[float, dict[str, any]]:
    """Evaluate the answer."""
    return 0.9, {"match": True}


def create_nested_pipeline():
    """Create a pipeline with nested structure for testing."""
    # Inner retrieval pipeline
    retrieval = Pipeline(nodes=[retrieve], name="retrieval")
    
    # RAG pipeline (uses retrieval)
    rag = Pipeline(
        nodes=[
            extract_query,
            retrieval.as_node(name="retrieval_step"),
            generate_answer,
        ],
        name="rag"
    )
    
    # Evaluation pipeline (uses RAG)
    eval_pipeline = Pipeline(
        nodes=[
            rag.as_node(name="rag_pipeline"),
            evaluate,
        ],
        name="evaluation"
    )
    
    return eval_pipeline


def main():
    os.makedirs('outputs', exist_ok=True)
    
    pipeline = create_nested_pipeline()
    
    # Generate HTML with depth=3 to show nested pipelines
    handler = UIHandler(pipeline, depth=3)
    graph_data = handler.get_visualization_data(traverse_collapsed=True)
    
    renderer = JSRenderer()
    react_flow_data = renderer.render(
        graph_data,
        theme='dark',
        separate_outputs=False,  # Test combined mode
        show_types=True,
    )
    
    html_content = generate_widget_html(react_flow_data)
    
    output_path = 'outputs/test_viz_comprehensive.html'
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML saved to {output_path}")
    print(f"Total nodes: {len(react_flow_data['nodes'])}")
    print(f"Total edges: {len(react_flow_data['edges'])}")
    
    # Print node details
    for node in react_flow_data['nodes']:
        node_type = node['data'].get('nodeType', 'UNKNOWN')
        label = node['data'].get('label', 'N/A')
        parent = node.get('parentNode', 'root')
        source_id = node['data'].get('sourceId', None)
        is_output = source_id is not None
        print(f"  - {node_type}: {label} (parent: {parent}){' [OUTPUT]' if is_output else ''}")
    
    print("\nEdges:")
    for edge in react_flow_data['edges']:
        print(f"  - {edge['source']} -> {edge['target']}")


if __name__ == "__main__":
    main()

