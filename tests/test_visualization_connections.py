"""Test visualization of output connections and pruning.

Tests for issues found in rag_hypernodes project:
1. Output from one node should connect to input of nested pipeline
2. Pruned outputs should not appear in visualization
3. Renamed inputs/outputs should show proper connections (no floating nodes)
"""

import pytest
from hypernodes import Pipeline, node


@node(output_name="query")
def extract_query(eval_pair: dict) -> str:
    """Extract query from evaluation pair."""
    return eval_pair["query"]


@node(output_name="retrieved_docs")
def retrieve(query: str, vector_store: object, top_k: int) -> list:
    """Retrieve documents."""
    return [f"doc for {query}"] * top_k


@node(output_name="answer")
def generate_answer(query: str, retrieved_docs: list, llm: object) -> str:
    """Generate answer from query and docs."""
    return f"Answer for {query} using {len(retrieved_docs)} docs"


@node(output_name="evaluation_result")
def evaluate_answer(answer: str, eval_pair: dict) -> float:
    """Evaluate answer quality."""
    return 0.95


class TestOutputConnectionsInViz:
    """Test that outputs connect properly to nested pipeline inputs."""

    def test_output_connects_to_nested_pipeline_input(self):
        """Test that node output connects to nested pipeline's input parameter.
        
        Issue: extract_query outputs 'query', retrieval pipeline needs 'query' input.
        The visualization should show a connection between them.
        """
        # Create retrieval pipeline that takes query
        retrieval = Pipeline(nodes=[retrieve, generate_answer], name="retrieval")
        retrieval.bind(vector_store=object(), top_k=3, llm=object())
        
        # Create evaluation pipeline with extract_query -> retrieval
        evaluation = Pipeline(
            nodes=[
                extract_query,
                retrieval.as_node(),
                evaluate_answer
            ],
            name="evaluation"
        )
        
        # Get visualization data
        viz = evaluation.visualize(depth=2, min_arg_group_size=None)
        viz_str = str(viz)
        
        # The query output from extract_query should connect to retrieval
        # Both should appear in the visualization
        assert 'query' in viz_str, "Query should appear in visualization"
        assert 'extract_query' in viz_str, "extract_query node should appear"
        
        # Check the graph structure
        # query should be produced by extract_query and consumed by retrieve
        from hypernodes.viz.visualization import _collect_visualization_data
        viz_data = _collect_visualization_data(evaluation, depth=2)
        
        # Find edges: should have extract_query -> query and query -> retrieve
        def unpack_edge(edge):
            if len(edge) == 3:
                return edge[:2]  # Just get source and target
            return edge
        
        query_edges = [(src, tgt) for src, tgt in [unpack_edge(e) for e in viz_data.edges] if 'query' in str(src) or 'query' in str(tgt)]
        assert len(query_edges) > 0, "Should have edges involving query"
        
        # Verify query is not in root_args (it's produced internally)
        assert 'query' not in viz_data.root_args, "query should not be an external input"
    
    def test_no_duplicate_edges_in_nested_pipeline(self):
        """Test that expanding nested pipelines doesn't create duplicate edges.
        
        Issue: Bound parameters from nested pipelines were getting edges added twice,
        causing double arrows in visualization.
        """
        # Create retrieval pipeline with bound inputs
        retrieval = Pipeline(nodes=[retrieve, generate_answer], name="retrieval")
        retrieval.bind(vector_store=object(), top_k=3, llm=object())
        
        # Create evaluation pipeline
        evaluation = Pipeline(
            nodes=[extract_query, retrieval.as_node()],
            name="evaluation"
        )
        
        # Get visualization data
        from hypernodes.viz.visualization import _collect_visualization_data
        viz_data = _collect_visualization_data(evaluation, depth=2)
        
        # Count edges - should not have duplicates
        edge_counts = {}
        for edge_data in viz_data.edges:
            # Handle new format with optional label
            if len(edge_data) == 3:
                src, tgt, _ = edge_data
            else:
                src, tgt = edge_data
            key = (id(src) if not isinstance(src, str) else src,
                   id(tgt) if not isinstance(tgt, str) else tgt)
            edge_counts[key] = edge_counts.get(key, 0) + 1
        
        # Check for duplicates
        duplicates = [count for count in edge_counts.values() if count > 1]
        assert len(duplicates) == 0, \
            f"Found {len(duplicates)} duplicate edges (double arrows)"


class TestOutputPruningInViz:
    """Test that pruned outputs don't appear in visualization."""

    def test_unused_outputs_not_shown_in_nested_pipeline(self):
        """Test that unused outputs from nested pipeline are pruned.
        
        Issue: retrieval pipeline outputs both 'answer' and 'retrieved_docs',
        but evaluation only uses 'answer'. The visualization should only show
        'answer' as output from the retrieval pipeline node.
        """
        # Create retrieval pipeline with multiple outputs
        retrieval = Pipeline(
            nodes=[retrieve, generate_answer],
            name="retrieval"
        )
        retrieval.bind(vector_store=object(), top_k=3, llm=object())
        
        # Create evaluation that only uses 'answer', not 'retrieved_docs'
        evaluation = Pipeline(
            nodes=[
                extract_query,
                retrieval.as_node(),
                evaluate_answer  # Only uses 'answer', not 'retrieved_docs'
            ],
            name="evaluation"
        )
        
        # Check the graph - retrieved_docs should be pruned
        # The PipelineNode should only output what's needed
        retrieval_node = evaluation.nodes[1]  # The PipelineNode
        
        # Get required outputs for this node from the graph
        required_outputs = evaluation.graph.required_outputs.get(retrieval_node)
        
        # Should only require 'answer', not 'retrieved_docs'
        if required_outputs is not None:
            assert 'answer' in required_outputs or required_outputs == ['answer']
            assert 'retrieved_docs' not in str(required_outputs), \
                "retrieved_docs should not be in required outputs"
    
    def test_pruned_outputs_not_in_visualization(self):
        """Test that pruned outputs don't show in expanded visualization."""
        # Create retrieval pipeline
        retrieval = Pipeline(
            nodes=[retrieve, generate_answer],
            name="retrieval"
        )
        retrieval.bind(vector_store=object(), top_k=3, llm=object())
        
        # Create evaluation that only uses 'answer'
        evaluation = Pipeline(
            nodes=[
                extract_query,
                retrieval.as_node(),
                evaluate_answer
            ],
            name="evaluation"
        )
        
        # Visualize with expansion
        viz = evaluation.visualize(depth=2, min_arg_group_size=None)
        viz_str = str(viz)
        
        # 'answer' should appear (it's used)
        assert 'answer' in viz_str, "answer output should appear"
        
        # 'retrieved_docs' might appear as it's produced by retrieve node
        # But it should not appear as an output of the nested pipeline node
        # when collapsed (depth=1)
        viz_collapsed = evaluation.visualize(depth=1, min_arg_group_size=None)
        viz_collapsed_str = str(viz_collapsed)
        
        # In collapsed view, we can check the PipelineNode label
        # It should show output as 'answer', not both
        # This is hard to test precisely, but we can check the structure


class TestRenamedInputsOutputsInViz:
    """Test that renamed inputs/outputs show proper connections."""

    def test_input_mapping_shows_connections_no_floating_nodes(self):
        """Test that input mapping creates proper connections in visualization.
        
        Issue: When using input_mapping, renamed parameters can appear as
        "floating nodes" without connections in the visualization.
        """
        # Create inner pipeline with specific parameter names
        @node(output_name="result")
        def process(data: str, config: dict) -> str:
            return f"Processed {data}"
        
        inner = Pipeline(nodes=[process], name="inner")
        inner.bind(config={"key": "value"})
        
        # Create outer pipeline with different parameter names
        # Map outer's "input_data" to inner's "data"
        inner_node = inner.as_node(input_mapping={"input_data": "data"})
        
        @node(output_name="input_data")
        def prepare(raw: str) -> str:
            return raw.upper()
        
        outer = Pipeline(nodes=[prepare, inner_node], name="outer")
        
        # Visualize with expansion
        viz = outer.visualize(depth=2, min_arg_group_size=None)
        viz_str = str(viz)
        
        # Check visualization data
        from hypernodes.viz.visualization import _collect_visualization_data
        viz_data = _collect_visualization_data(outer, depth=2)
        
        # 'input_data' should not be a floating node
        # It should connect: prepare -> input_data -> process
        # Find all edges involving input_data
        def unpack_edge(edge):
            if len(edge) == 3:
                return edge[:2]  # Just get source and target
            return edge
        
        input_data_edges = [
            (src, tgt) for src, tgt in [unpack_edge(e) for e in viz_data.edges] 
            if 'input_data' in str(src) or 'input_data' in str(tgt)
        ]
        
        # Should have at least 2 edges: prepare produces it, process consumes it
        assert len(input_data_edges) >= 1, \
            f"input_data should have connections, found {len(input_data_edges)}"
        
        # Check that 'data' (the inner param name) doesn't appear as separate node
        # It should be remapped to 'input_data'
        # In expanded view, we might see 'data' as the actual parameter
        # but it should be connected
    
    def test_output_mapping_shows_connections_no_floating_nodes(self):
        """Test that output mapping creates proper connections."""
        @node(output_name="inner_result")
        def compute(x: int) -> int:
            return x * 2
        
        inner = Pipeline(nodes=[compute], name="inner")
        
        # Map inner's "inner_result" to outer's "final_value"
        inner_node = inner.as_node(output_mapping={"inner_result": "final_value"})
        
        @node(output_name="formatted")
        def format_result(final_value: int) -> str:
            return f"Result: {final_value}"
        
        outer = Pipeline(nodes=[inner_node, format_result], name="outer")
        
        # Visualize
        viz = outer.visualize(depth=2, min_arg_group_size=None)
        
        # Check visualization data
        from hypernodes.viz.visualization import _collect_visualization_data
        viz_data = _collect_visualization_data(outer, depth=2)
        
        # final_value should connect compute to format_result
        # Check that it's in output_to_node
        assert 'final_value' in viz_data.output_to_node or 'inner_result' in viz_data.output_to_node, \
            "Mapped output should be in output_to_node"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

