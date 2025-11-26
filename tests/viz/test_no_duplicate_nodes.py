"""Test that visualizations don't create duplicate nodes for nested pipelines."""

from collections import Counter

from hypernodes import Pipeline, node


def test_no_duplicate_nodes_in_expanded_nested_pipeline():
    """Test that expanded nested pipelines don't create duplicate boundary DataNodes."""
    
    @node(output_name='extracted_query')
    def extract_query(query: str) -> str:
        return query
    
    @node(output_name='vector_store')
    def query_vdb(extracted_query: str) -> list:
        return []
    
    @node(output_name='retrieved_docs')
    def retrieve(top_k: int, vector_store: list) -> list:
        return vector_store[:top_k]
    
    # Create nested pipeline
    retrieval_pipeline = Pipeline(nodes=[extract_query, query_vdb, retrieve], name='retrieval')
    
    @node(output_name='answer')
    def generate(retrieved_docs: list) -> str:
        return 'answer'
    
    # Outer pipeline
    outer = Pipeline(nodes=[retrieval_pipeline.as_node(), generate])
    
    # Visualize with depth=2 (expanded), separate_outputs=True for individual output nodes
    result = outer.visualize(depth=2, separate_outputs=True)
    
    # Extract HTML
    if hasattr(result, 'data'):
        html = str(result.data)
    elif hasattr(result, '_repr_html_'):
        html = str(result._repr_html_())
    else:
        html = str(result)
    
    # Extract all node names from <title> elements (excluding edges)
    import re
    nodes = [t for t in re.findall(r'<title>(.*?)</title>', html) 
             if '-&gt;' not in t and 'cluster_' not in t]
    
    # Check for duplicates
    counts = Counter(nodes)
    duplicates = [(name, count) for name, count in counts.items() if count > 1]
    
    # Should have no duplicates
    assert len(duplicates) == 0, (
        f"Found duplicate nodes: {duplicates}. "
        "Nested pipelines should not create duplicate boundary DataNodes."
    )
    
    # Verify expected nodes exist (without duplicates)
    # Note: When expanded (depth=2), the PipelineNode wrapper may not appear as a titled node
    # because the internal structure is fully exposed. That's correct behavior.
    expected_nodes = {
        'extract_query', 'extracted_query',
        'query_vdb', 'vector_store',
        'retrieve', 'retrieved_docs',
        'generate', 'answer',
        'query', 'top_k'  # inputs
    }
    actual_nodes = set(nodes)
    
    # All expected nodes should be present
    missing = expected_nodes - actual_nodes
    assert not missing, f"Missing expected nodes: {missing}"
    
    # Check that we don't have unexpected duplicates (actual may have pipeline wrapper or not)
    # The key is no duplicates, not exact count match
    assert len(actual_nodes) >= len(expected_nodes), (
        f"Expected at least {len(expected_nodes)} unique nodes, got {len(actual_nodes)}"
    )


def test_no_duplicate_nodes_with_output_mapping():
    """Test that output_mapping doesn't create duplicate nodes."""
    
    @node(output_name='doubled')
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name='tripled')
    def triple(doubled: int) -> int:
        return doubled * 3
    
    inner = Pipeline(nodes=[double, triple], name='processor')
    # Map 'tripled' to 'result'
    inner_node = inner.as_node(output_mapping={'tripled': 'result'})
    
    @node(output_name='final')
    def add_ten(result: int) -> int:
        return result + 10
    
    outer = Pipeline(nodes=[inner_node, add_ten])
    
    # Visualize with depth=2 (expanded), separate_outputs=True for individual output nodes
    result = outer.visualize(depth=2, separate_outputs=True)
    
    # Extract HTML
    if hasattr(result, 'data'):
        html = str(result.data)
    elif hasattr(result, '_repr_html_'):
        html = str(result._repr_html_())
    else:
        html = str(result)
    
    # Extract all node names
    import re
    nodes = [t for t in re.findall(r'<title>(.*?)</title>', html) 
             if '-&gt;' not in t and 'cluster_' not in t]
    
    # Check for duplicates
    counts = Counter(nodes)
    duplicates = [(name, count) for name, count in counts.items() if count > 1]
    
    # Should have no duplicates, including for the mapped output
    assert len(duplicates) == 0, (
        f"Found duplicate nodes: {duplicates}. "
        "Output mapping should not create duplicate nodes."
    )
    
    # The inner 'tripled' should be accessible as 'result' in outer scope
    # but should not create duplicate nodes
    assert 'tripled' in nodes, "Inner output 'tripled' should exist"
    # 'result' would only appear if we created a duplicate boundary node (which we shouldn't)
    # The outer scope should use 'tripled' directly
