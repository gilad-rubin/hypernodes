"""Reproduce the ghost nodes issue from the RAG pipeline example."""

from hypernodes import Pipeline, node
from hypernodes.viz.graph_walker import GraphWalker


def test_rag_pipeline_ghost_nodes():
    """Reproduce the exact issue from the image: ghost nodes from retrieval pipeline."""
    
    # Inner retrieval pipeline (collapsed)
    @node(output_name="retrieved_docs")
    def retrieve(query: str, vector_store: object, top_k: int) -> list:
        return ["doc1", "doc2"]
    
    retrieval_pipeline = Pipeline(nodes=[retrieve])
    
    # Outer RAG pipeline
    @node(output_name="extracted_query")
    def extract_query(eval_pair: dict) -> str:
        return "query"
    
    @node(output_name="answer")
    def generate(query: str, retrieved_docs: list, llm: object) -> str:
        return "answer"
    
    @node(output_name="evaluation_result")
    def evaluate_answer(answer: str, eval_pair: dict, judge: object) -> dict:
        return {"score": 0.9}
    
    # Build the pipeline
    retrieval_node = retrieval_pipeline.as_node(
        input_mapping={"extracted_query": "query"}
    )
    
    rag_pipeline = Pipeline(nodes=[
        extract_query,
        retrieval_node,
        generate,
        evaluate_answer
    ])
    
    # Get visualization with collapsed retrieval pipeline
    walker = GraphWalker(rag_pipeline, expanded_nodes=set(), group_inputs=False)
    graph_data = walker.get_visualization_data()
    
    # Check for ghost nodes
    node_ids = {n.id for n in graph_data.nodes}
    edge_node_ids = set()
    for edge in graph_data.edges:
        edge_node_ids.add(edge.source)
        edge_node_ids.add(edge.target)
    
    ghost_nodes = edge_node_ids - node_ids
    
    print("\n📊 RAG Pipeline Visualization Statistics:")
    print(f"   Nodes in nodes list: {len(node_ids)}")
    print(f"   Nodes in edges: {len(edge_node_ids)}")
    print(f"   Ghost nodes: {len(ghost_nodes)}")
    
    if ghost_nodes:
        print("\n👻 GHOST NODES FOUND:")
        for ghost in ghost_nodes:
            print(f"   - {ghost}")
            # Find edges using this ghost node
            for edge in graph_data.edges:
                if edge.source == ghost or edge.target == ghost:
                    print(f"     Edge: {edge.source} -> {edge.target}")
    
    print("\n📝 All nodes in visualization:")
    for n in graph_data.nodes:
        node_type = type(n).__name__
        node_name = getattr(n, 'label', getattr(n, 'name', n.id))
        print(f"   - {node_name} ({node_type})")
    
    print("\n🔗 All edges:")
    for edge in graph_data.edges:
        # Try to resolve to names
        src_node = next((n for n in graph_data.nodes if n.id == edge.source), None)
        tgt_node = next((n for n in graph_data.nodes if n.id == edge.target), None)
        src_name = getattr(src_node, 'label', getattr(src_node, 'name', edge.source)) if src_node else edge.source
        tgt_name = getattr(tgt_node, 'label', getattr(tgt_node, 'name', edge.target)) if tgt_node else edge.target
        print(f"   - {src_name} -> {tgt_name}")
    
    assert len(ghost_nodes) == 0, f"Found ghost nodes: {ghost_nodes}"


if __name__ == "__main__":
    test_rag_pipeline_ghost_nodes()
