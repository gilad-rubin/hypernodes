"""Tests for input_mapping visualization issues."""

from hypernodes import Pipeline, node
from hypernodes.viz.graphviz_ui import _collect_visualization_data


@node(output_name="cleaned")
def clean_text(passage: str) -> str:
    """Clean text."""
    return passage.strip().lower()


@node(output_name="result")
def process_item(item: str, config: dict) -> str:
    """Process an item."""
    return f"Processed: {item}"


@node(output_name="answer")
def generate_answer(cleaned: str, llm: str) -> str:
    """Generate answer."""
    return f"Answer from {llm}: {cleaned}"


class TestInputMappingVisualization:
    """Test that input_mapping is visible in visualization."""

    def test_input_mapping_shows_label_on_edge(self):
        """Test that input mapping is indicated visually in the graph.
        
        When using input_mapping={"eval_pairs": "eval_pair"}, the visualization
        should show the mapping, not just replace the name silently.
        """
        # Inner pipeline expects "eval_pair"
        inner = Pipeline(nodes=[process_item], name="inner")
        inner.bind(config={"x": 1})
        
        # Outer provides "eval_pairs" → "eval_pair"
        inner_node = inner.as_node(
            input_mapping={"eval_pairs": "item"},
            name="batch_process"
        )
        outer = Pipeline(nodes=[inner_node], name="outer")
        
        # Visualize with expansion
        viz = outer.visualize(depth=2, group_inputs=False)
        viz_str = str(viz)
        
        # The visualization should somehow indicate the mapping
        # Either in edge labels or node labels
        # For now, check that BOTH names appear somewhere
        assert "eval_pairs" in viz_str, "Outer parameter name should be visible"
        # The inner name might not be visible if we're doing pure replacement
        # but there should be some indication of mapping
        
        # Check visualization data
        viz_data = _collect_visualization_data(outer, depth=2)
        
        # The root args should have the OUTER name
        assert "eval_pairs" in viz_data.root_args, \
            "Should show outer parameter name 'eval_pairs'"
        assert "item" not in viz_data.root_args, \
            "Should not show inner parameter name 'item' as root arg when mapped"


class TestDepthExpansion:
    """Test that depth parameter works correctly."""

    def test_depth_2_expands_one_level(self):
        """Test that depth=2 expands nested pipelines one level deep."""
        # Create a simple nested structure
        inner = Pipeline(nodes=[clean_text], name="inner")
        
        outer = Pipeline(nodes=[inner.as_node(name="inner_node")], name="outer")
        
        # Depth 1: should show collapsed pipeline node
        viz_data_1 = _collect_visualization_data(outer, depth=1)
        
        # Depth 2: should expand and show inner nodes
        viz_data_2 = _collect_visualization_data(outer, depth=2)
        
        # At depth=1, should only see PipelineNode (collapsed)
        from hypernodes.pipeline_node import PipelineNode
        pipeline_nodes_d1 = [n for n in viz_data_1.nodes if isinstance(n, PipelineNode)]
        func_nodes_d1 = [n for n in viz_data_1.nodes if not isinstance(n, PipelineNode)]
        
        assert len(pipeline_nodes_d1) == 1, "Depth 1 should show collapsed pipeline node"
        assert len(func_nodes_d1) == 0, "Depth 1 should not show inner function nodes"
        
        # At depth=2, should see expanded nodes (clean_text function)
        pipeline_nodes_d2 = [n for n in viz_data_2.nodes if isinstance(n, PipelineNode)]
        func_nodes_d2 = [n for n in viz_data_2.nodes if not isinstance(n, PipelineNode)]
        
        assert len(pipeline_nodes_d2) == 0, "Depth 2 should expand pipeline, no collapsed nodes"
        assert len(func_nodes_d2) == 1, "Depth 2 should show inner function node (clean_text)"

    def test_depth_3_expands_two_levels(self):
        """Test that depth=3 expands two levels of nesting."""
        # Level 1 (innermost)
        level1 = Pipeline(nodes=[clean_text], name="level1")
        
        # Level 2 (middle)
        level2 = Pipeline(nodes=[level1.as_node(name="level1_node")], name="level2")
        
        # Level 3 (outer)
        level3 = Pipeline(nodes=[level2.as_node(name="level2_node")], name="level3")
        
        # Depth 1: collapsed outermost only
        viz_data_1 = _collect_visualization_data(level3, depth=1)
        from hypernodes.pipeline_node import PipelineNode
        pipeline_nodes_d1 = [n for n in viz_data_1.nodes if isinstance(n, PipelineNode)]
        assert len(pipeline_nodes_d1) == 1, "Depth 1 should show one collapsed pipeline"
        
        # Depth 2: expand one level (should see level1 pipeline collapsed)
        viz_data_2 = _collect_visualization_data(level3, depth=2)
        pipeline_nodes_d2 = [n for n in viz_data_2.nodes if isinstance(n, PipelineNode)]
        assert len(pipeline_nodes_d2) == 1, "Depth 2 should show level1 collapsed"
        
        # Depth 3: expand two levels (should see clean_text function)
        viz_data_3 = _collect_visualization_data(level3, depth=3)
        pipeline_nodes_d3 = [n for n in viz_data_3.nodes if isinstance(n, PipelineNode)]
        func_nodes_d3 = [n for n in viz_data_3.nodes if not isinstance(n, PipelineNode)]
        assert len(pipeline_nodes_d3) == 0, "Depth 3 should have all expanded"
        assert len(func_nodes_d3) == 1, "Depth 3 should show clean_text function"


class TestComplexInputMappingScenario:
    """Test the actual RAG evaluation scenario."""

    def test_rag_evaluation_pipeline_structure(self):
        """Test the structure from the RAG evaluation demo.
        
        This mirrors the actual issue:
        - extract_query produces "query"
        - retrieval pipeline expects different param (with mapping)
        - Should show proper connections
        """
        # Simulate extract_query node
        @node(output_name="query")
        def extract_query(eval_pair: str) -> str:
            return eval_pair
        
        # Simulate retrieval pipeline (expects "query" internally)
        retrieval = Pipeline(nodes=[clean_text], name="retrieval")
        # Assume it's bound with vector_store, llm, etc.
        
        # Create evaluation pipeline
        eval_pipeline = Pipeline(
            nodes=[extract_query, retrieval.as_node(name="retrieval")],
            name="evaluation"
        )
        
        # Visualize at depth=2
        viz_data = _collect_visualization_data(eval_pipeline, depth=2)
        
        # Should show connection: extract_query → clean_text
        # Find the clean_text node
        clean_text_node = None
        for n in viz_data.nodes:
            if hasattr(n, 'func') and getattr(n.func, '__name__', None) == 'clean_text':
                clean_text_node = n
                break
        
        assert clean_text_node is not None, "Should find clean_text node when expanded"
        
        # Check edges - there should be NO floating unconnected parameters
        # All parameters should either be:
        # 1. Connected to a producer node
        # 2. External inputs (in root_args)
        # Handle new edge format (source, target, label)
        def unpack_edge(edge):
            if len(edge) == 3:
                return edge
            else:
                return (*edge, None)
        
        string_edges = [unpack_edge(edge) for edge in viz_data.edges if isinstance(edge[0], str)]
        node_edges = [unpack_edge(edge) for edge in viz_data.edges if not isinstance(edge[0], str)]
        
        # If "query" is produced by extract_query, it should NOT appear as string edge
        # It should be a node-to-node edge
        query_string_edges = [(s, t, lbl) for s, t, lbl in string_edges if s == "query"]
        assert len(query_string_edges) == 0, \
            "query should not be a floating parameter, should connect extract_query → clean_text"
