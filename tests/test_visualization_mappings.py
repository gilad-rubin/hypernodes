"""Test visualization of input/output mappings and cross-level connections.

Tests for issues found in rag_hypernodes project:
1. Extract query → nested retrieval connection (no floating id boxes)
2. eval_pairs → eval_pair mapping with proper connections
3. No duplicate parameter edges (top_k multiple arrows)
4. Nested bound inputs visible with correct names
"""

import pytest
from hypernodes import Pipeline, node
from hypernodes.viz.graph_serializer import GraphSerializer


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


@node(output_name="metrics")
def compute_metrics(evaluation_results: list) -> dict:
    """Compute aggregate metrics."""
    return {"count": len(evaluation_results), "avg": 0.95}


class TestCrossLevelConnections:
    """Test that outputs connect properly across nested pipeline boundaries."""

    def test_evaluation_pipeline_depth2_connection(self):
        """Test that extract_query output connects to nested retrieval input.
        
        Issue: extract_query outputs 'query', retrieval pipeline needs 'query' input.
        The serializer should recognize that 'query' is produced by extract_query
        and create a node→node edge, NOT a parameter→node edge with id number.
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
        
        # Serialize with depth=2 (expand nested pipeline)
        serializer = GraphSerializer(evaluation)
        graph_data = serializer.serialize(depth=2)
        
        # Find extract_query node
        extract_query_node = None
        for n in graph_data["nodes"]:
            if n["function_name"] == "extract_query":
                extract_query_node = n
                break
        
        assert extract_query_node is not None, "extract_query node should exist"
        
        # Find retrieve node (inside nested pipeline)
        retrieve_node = None
        for n in graph_data["nodes"]:
            if n["function_name"] == "retrieve":
                retrieve_node = n
                break
        
        assert retrieve_node is not None, "retrieve node should exist (nested pipeline expanded)"
        
        # Find edge from extract_query to retrieve
        # It should be a node→node edge, not a parameter edge
        edges_to_retrieve = [
            e for e in graph_data["edges"]
            if e["target"] == retrieve_node["id"]
        ]
        
        # Should have an edge from extract_query (producing 'query')
        has_node_edge = False
        has_param_edge_with_id = False
        
        for edge in edges_to_retrieve:
            if edge["source"] == extract_query_node["id"]:
                has_node_edge = True
            # Check for problematic parameter edges with id numbers
            if isinstance(edge["source"], str) and edge["source"].startswith("input_"):
                param_name = edge["source"].replace("input_", "")
                # If param_name is a number (id), that's the bug
                if param_name.isdigit():
                    has_param_edge_with_id = True
        
        assert has_node_edge, "Should have edge from extract_query to retrieve"
        assert not has_param_edge_with_id, "Should NOT have parameter edge with id number"
        
        # Also check that 'query' is not in root_args of the evaluation pipeline
        root_level = [l for l in graph_data["levels"] if l["level_id"] == "root"][0]
        assert "query" not in root_level["unfulfilled_inputs"], \
            "'query' should not be unfulfilled (it's produced by extract_query)"


class TestInputOutputMapping:
    """Test that input/output mappings show proper connections."""

    def test_metrics_pipeline_eval_pair_mapping(self):
        """Test that input_mapping creates proper connections with labels.
        
        Issue: metrics pipeline uses map_over with input_mapping:
        {"eval_pairs": "eval_pair"}. The visualization should show these
        are connected, not floating separate parameters.
        """
        # Create evaluation pipeline that takes eval_pair
        evaluation = Pipeline(
            nodes=[extract_query, evaluate_answer],
            name="evaluation"
        )
        
        # Create metrics pipeline with mapping
        evaluation_node = evaluation.as_node(
            name="batch_evaluation",
            map_over="eval_pairs",
            input_mapping={"eval_pairs": "eval_pair"},
            output_mapping={"evaluation_result": "evaluation_results"}
        )
        
        metrics = Pipeline(
            nodes=[evaluation_node, compute_metrics],
            name="metrics"
        )
        
        # Serialize with depth=2 (expand nested pipeline)
        serializer = GraphSerializer(metrics)
        graph_data = serializer.serialize(depth=2)
        
        # Check that eval_pairs is in root args (outer name)
        root_level = [l for l in graph_data["levels"] if l["level_id"] == "root"][0]
        assert "eval_pairs" in root_level["unfulfilled_inputs"], \
            "Should use outer parameter name 'eval_pairs'"
        
        # Check that eval_pair is NOT in root args (it's the inner name)
        assert "eval_pair" not in root_level["unfulfilled_inputs"], \
            "Should not show inner parameter name 'eval_pair' as root arg"
        
        # Find extract_query node (inside nested pipeline)
        extract_query_node = None
        for n in graph_data["nodes"]:
            if n["function_name"] == "extract_query":
                extract_query_node = n
                break
        
        assert extract_query_node is not None, "extract_query should exist in expanded view"
        
        # Find edges to extract_query that involve eval_pair
        edges_to_extract = [
            e for e in graph_data["edges"]
            if e["target"] == extract_query_node["id"]
        ]
        
        # Should have an edge for the eval_pair parameter
        eval_pair_edges = [
            e for e in edges_to_extract
            if "eval_pair" in str(e["source"]) or e.get("mapping_label")
        ]
        
        assert len(eval_pair_edges) > 0, \
            "Should have edge(s) connecting eval_pair parameter"
        
        # Check for mapping label
        has_mapping_label = any(
            e.get("mapping_label") and "→" in e["mapping_label"]
            for e in eval_pair_edges
        )
        
        assert has_mapping_label, \
            "Should have mapping label showing 'eval_pairs → eval_pair'"

    def test_output_mapping_connection(self):
        """Test that output_mapping connects properly.
        
        When inner pipeline outputs 'evaluation_result' but outer expects
        'evaluation_results', the connection should work.
        """
        # Create evaluation pipeline
        evaluation = Pipeline(
            nodes=[extract_query, evaluate_answer],
            name="evaluation"
        )
        
        # Map output
        evaluation_node = evaluation.as_node(
            output_mapping={"evaluation_result": "evaluation_results"}
        )
        
        metrics = Pipeline(
            nodes=[evaluation_node, compute_metrics],
            name="metrics"
        )
        
        # Serialize with depth=2
        serializer = GraphSerializer(metrics)
        graph_data = serializer.serialize(depth=2)
        
        # Find evaluate_answer node (produces evaluation_result)
        evaluate_node = None
        for n in graph_data["nodes"]:
            if n["function_name"] == "evaluate_answer":
                evaluate_node = n
                break
        
        # Find compute_metrics node (consumes evaluation_results)
        compute_node = None
        for n in graph_data["nodes"]:
            if n["function_name"] == "compute_metrics":
                compute_node = n
                break
        
        assert evaluate_node is not None
        assert compute_node is not None
        
        # Should have edge from evaluate_answer to compute_metrics
        edge_exists = any(
            e["source"] == evaluate_node["id"] and e["target"] == compute_node["id"]
            for e in graph_data["edges"]
        )
        
        assert edge_exists, \
            "Should have edge from evaluate_answer to compute_metrics through output mapping"


class TestDuplicateEdges:
    """Test that there are no duplicate parameter edges."""

    def test_no_duplicate_parameter_edges(self):
        """Test that bound parameters don't create duplicate edges.
        
        Issue: top_k parameter appearing with multiple arrows in visualization.
        Each parameter should have exactly one input edge.
        """
        # Create retrieval pipeline with bound inputs
        retrieval = Pipeline(nodes=[retrieve, generate_answer], name="retrieval")
        retrieval.bind(vector_store=object(), top_k=3, llm=object())
        
        # Create evaluation pipeline
        evaluation = Pipeline(
            nodes=[extract_query, retrieval.as_node()],
            name="evaluation"
        )
        
        # Serialize with depth=2 (expand nested)
        serializer = GraphSerializer(evaluation)
        graph_data = serializer.serialize(depth=2)
        
        # Find retrieve node
        retrieve_node = None
        for n in graph_data["nodes"]:
            if n["function_name"] == "retrieve":
                retrieve_node = n
                break
        
        assert retrieve_node is not None
        
        # Find all edges targeting retrieve node
        edges_to_retrieve = [
            e for e in graph_data["edges"]
            if e["target"] == retrieve_node["id"]
        ]
        
        # Group by source (to detect duplicates)
        source_counts = {}
        for edge in edges_to_retrieve:
            source = edge["source"]
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Check for duplicates
        duplicates = {src: count for src, count in source_counts.items() if count > 1}
        
        assert len(duplicates) == 0, \
            f"Found duplicate edges to retrieve: {duplicates}"
        
        # Specifically check top_k (it's bound, so should be one edge)
        top_k_edges = [
            e for e in edges_to_retrieve
            if "top_k" in str(e["source"])
        ]
        
        assert len(top_k_edges) <= 1, \
            f"top_k should have at most 1 edge, found {len(top_k_edges)}"


class TestNestedBoundInputs:
    """Test that bound inputs from nested pipelines appear correctly."""

    def test_nested_bound_inputs_visible(self):
        """Test that bound inputs are visible with correct styling flags.
        
        Bound inputs should be marked as is_bound=True in the node's inputs.
        """
        # Create retrieval pipeline with bound inputs
        retrieval = Pipeline(nodes=[retrieve, generate_answer], name="retrieval")
        retrieval.bind(vector_store=object(), top_k=3, llm=object())
        
        # Create evaluation pipeline
        evaluation = Pipeline(
            nodes=[extract_query, retrieval.as_node()],
            name="evaluation"
        )
        
        # Serialize with depth=2
        serializer = GraphSerializer(evaluation)
        graph_data = serializer.serialize(depth=2)
        
        # Find retrieve node
        retrieve_node = None
        for n in graph_data["nodes"]:
            if n["function_name"] == "retrieve":
                retrieve_node = n
                break
        
        assert retrieve_node is not None
        
        # Check that top_k, vector_store, and llm inputs are marked as bound
        inputs = retrieve_node["inputs"]
        
        top_k_input = None
        vector_store_input = None
        
        for inp in inputs:
            if inp["name"] == "top_k":
                top_k_input = inp
            elif inp["name"] == "vector_store":
                vector_store_input = inp
        
        # These are bound in the retrieval pipeline
        assert top_k_input is not None, "top_k should be in inputs"
        assert top_k_input["is_bound"], "top_k should be marked as bound"
        
        assert vector_store_input is not None, "vector_store should be in inputs"
        assert vector_store_input["is_bound"], "vector_store should be marked as bound"


class TestExpandedPipelineParameterEdges:
    """Ensure expanded pipelines don't create orphaned parameter nodes."""

    def _get_pipeline_node(self, graph_data, label_substring="batch"):
        """Helper to find the expanded PipelineNode for assertions."""
        for node in graph_data["nodes"]:
            if node["node_type"] == "PIPELINE" and label_substring in node["label"]:
                return node
        return None

    def test_metrics_depth2_no_param_edges_to_pipeline_node(self):
        """Depth=2 should not render parameter edges to expanded pipeline wrapper."""
        # Build metrics pipeline with mapping (matches notebook scenario)
        retrieval = Pipeline(nodes=[retrieve, generate_answer], name="retrieval")
        retrieval.bind(vector_store=object(), top_k=3, llm=object())
        evaluation = Pipeline(
            nodes=[extract_query, retrieval.as_node(), evaluate_answer],
            name="evaluation",
        )

        evaluation_node = evaluation.as_node(
            name="batch_evaluation",
            map_over="eval_pairs",
            input_mapping={"eval_pairs": "eval_pair"},
            output_mapping={"evaluation_result": "evaluation_results"},
        )
        metrics = Pipeline(nodes=[evaluation_node, compute_metrics], name="metrics")

        graph_data = GraphSerializer(metrics).serialize(depth=2)

        pipeline_node = self._get_pipeline_node(graph_data, label_substring="batch")
        assert pipeline_node is not None, "Expected expanded pipeline node present"
        pipeline_id = pipeline_node["id"]

        param_edges_to_pipeline = [
            e
            for e in graph_data["edges"]
            if e["edge_type"] == "parameter_flow" and e["target"] == pipeline_id
        ]

        # Failing behavior: parameter edges target the expanded wrapper, producing
        # unlabeled white boxes in the visualization. Should route into inner nodes.
        assert (
            len(param_edges_to_pipeline) == 0
        ), "Expanded pipeline should not receive parameter edges directly"

    def test_metrics_depth3_no_param_edges_to_pipeline_node(self):
        """Depth=3 (double expansion) should also avoid pipeline wrapper edges."""
        # Build nested metrics pipeline again to expand inner retrieval layer
        retrieval = Pipeline(nodes=[retrieve, generate_answer], name="retrieval")
        retrieval.bind(vector_store=object(), top_k=3, llm=object())
        evaluation = Pipeline(
            nodes=[extract_query, retrieval.as_node(), evaluate_answer],
            name="evaluation",
        )

        evaluation_node = evaluation.as_node(
            name="batch_evaluation",
            map_over="eval_pairs",
            input_mapping={"eval_pairs": "eval_pair"},
            output_mapping={"evaluation_result": "evaluation_results"},
        )
        metrics = Pipeline(nodes=[evaluation_node, compute_metrics], name="metrics")

        graph_data = GraphSerializer(metrics).serialize(depth=3)

        pipeline_node = self._get_pipeline_node(graph_data, label_substring="batch")
        assert pipeline_node is not None, "Expected expanded pipeline node present"
        pipeline_id = pipeline_node["id"]

        param_edges_to_pipeline = [
            e
            for e in graph_data["edges"]
            if e["edge_type"] == "parameter_flow" and e["target"] == pipeline_id
        ]

        assert (
            len(param_edges_to_pipeline) == 0
        ), "Expanded pipeline should not receive parameter edges directly at depth=3"


class TestExpandedPipelineWrapperEdges:
    """Ensure expanded pipelines never appear as raw numeric nodes."""

    def _get_pipeline_node(self, graph_data, name_substring: str) -> dict:
        for node in graph_data["nodes"]:
            if node["node_type"] == "PIPELINE" and name_substring in node["label"]:
                return node
        return {}

    def _assert_no_edges_touching_wrapper(self, graph_data, wrapper_id: str):
        touching = [
            e
            for e in graph_data["edges"]
            if (e["target"] == wrapper_id or e["source"] == wrapper_id)
            and not str(e["source"]).startswith("input_")
            and not str(e["target"]).startswith("input_")
        ]
        assert (
            len(touching) == 0
        ), f"Edges unexpectedly target expanded wrapper: {touching}"

    def test_evaluation_depth2_no_wrapper_edges(self):
        """Expanded retrieval wrapper should not get edges (prevents numeric ovals)."""
        retrieval = Pipeline(nodes=[retrieve, generate_answer], name="retrieval")
        retrieval.bind(vector_store=object(), top_k=3, llm=object())

        evaluation = Pipeline(
            nodes=[extract_query, retrieval.as_node(), evaluate_answer],
            name="evaluation",
        ).bind(judge=object())

        graph_data = GraphSerializer(evaluation).serialize(depth=2)
        wrapper = self._get_pipeline_node(graph_data, "retrieval")
        assert wrapper, "Expected retrieval pipeline wrapper in expanded view"

        self._assert_no_edges_touching_wrapper(graph_data, wrapper["id"])

    def test_metrics_depth3_no_wrapper_edges(self):
        """Expanded evaluation wrapper should not get edges at deeper expansion."""
        retrieval = Pipeline(nodes=[retrieve, generate_answer], name="retrieval")
        retrieval.bind(vector_store=object(), top_k=3, llm=object())
        evaluation = Pipeline(
            nodes=[extract_query, retrieval.as_node(), evaluate_answer],
            name="evaluation",
        )

        evaluation_node = evaluation.as_node(
            name="batch_evaluation",
            map_over="eval_pairs",
            input_mapping={"eval_pairs": "eval_pair"},
            output_mapping={"evaluation_result": "evaluation_results"},
        )
        metrics = Pipeline(nodes=[evaluation_node, compute_metrics], name="metrics")

        graph_data = GraphSerializer(metrics).serialize(depth=3)
        wrapper = self._get_pipeline_node(graph_data, "evaluation")
        assert wrapper, "Expected evaluation pipeline wrapper in expanded view"

        self._assert_no_edges_touching_wrapper(graph_data, wrapper["id"])


class TestGraphvizInputGrouping:
    """Front-end input grouping controls."""

    def test_group_inputs_separates_bound_and_unbound(self):
        _ = pytest.importorskip("graphviz")

        @node(output_name="out")
        def consume(a, b, bound1, bound2):
            return (a, b, bound1, bound2)

        pipeline = Pipeline(nodes=[consume], name="grouped").bind(bound1=1, bound2=2)

        consumer_id = None
        ser = GraphSerializer(pipeline).serialize(depth=1)
        for n in ser["nodes"]:
            if n["function_name"] == "consume":
                consumer_id = n["id"]
                break
        assert consumer_id, "consumer node id should be present"

        dot = pipeline.visualize(
            engine="graphviz",
            return_type="graphviz",
            group_inputs=True,
            min_arg_group_size=2,
        )
        src = dot.source

        # Separate bound/unbound groups are rendered
        assert f"group_{consumer_id}_unbound" in src
        assert f"group_{consumer_id}_bound" in src

        # Individual grouped inputs are not rendered as standalone nodes
        assert "input_a" not in src and "input_b" not in src
        assert "input_bound1" not in src and "input_bound2" not in src

        # Labels contain the right parameters without mixing
        assert "<TD>a" in src and "<TD>b" in src
        assert "<TD>bound1" in src and "<TD>bound2" in src

    def test_group_inputs_can_disable_grouping(self):
        _ = pytest.importorskip("graphviz")

        @node(output_name="out")
        def consume(a, b, bound1, bound2):
            return (a, b, bound1, bound2)

        pipeline = Pipeline(nodes=[consume], name="grouped").bind(bound1=1, bound2=2)
        dot = pipeline.visualize(
            engine="graphviz",
            return_type="graphviz",
            group_inputs=False,
            min_arg_group_size=2,
        )
        src = dot.source

        assert "group_" not in src
        assert "input_a" in src and "input_b" in src
        assert "input_bound1" in src and "input_bound2" in src


class TestInputLevelPlacement:
    """Ensure inputs are attached to the correct levels for nesting."""

    def test_nested_retrieval_inputs_in_nested_level(self):
        retrieval = Pipeline(nodes=[retrieve, generate_answer], name="retrieval")
        retrieval.bind(vector_store=object(), top_k=3, llm=object())

        evaluation = Pipeline(
            nodes=[extract_query, retrieval.as_node(), evaluate_answer],
            name="evaluation",
        ).bind(judge=object())

        graph_data = GraphSerializer(evaluation).serialize(depth=2)
        input_levels = graph_data.get("input_levels", {})

        # Identify nested level (child of root)
        nested_levels = [
            lvl["level_id"] for lvl in graph_data["levels"] if lvl["parent_level_id"] == "root"
        ]
        assert nested_levels, "Expected a nested level for retrieval pipeline"
        nested_level_id = nested_levels[0]

        assert input_levels.get("vector_store") == nested_level_id
        assert input_levels.get("top_k") == nested_level_id
        assert input_levels.get("llm") == nested_level_id

        # eval_pair is only consumed at root level
        assert input_levels.get("eval_pair") == "root"

    def test_map_over_outer_input_stays_at_root(self):
        """eval_pairs should remain at root even if only consumed inside nested pipeline."""
        retrieval = Pipeline(nodes=[retrieve, generate_answer], name="retrieval")
        retrieval.bind(vector_store=object(), top_k=3, llm=object())

        evaluation = Pipeline(
            nodes=[extract_query, retrieval.as_node(), evaluate_answer],
            name="evaluation",
        )
        evaluation_node = evaluation.as_node(
            name="batch_evaluation",
            map_over="eval_pairs",
            input_mapping={"eval_pairs": "eval_pair"},
            output_mapping={"evaluation_result": "evaluation_results"},
        )
        metrics = Pipeline(nodes=[evaluation_node, compute_metrics], name="metrics")

        graph_data = GraphSerializer(metrics).serialize(depth=2)
        input_levels = graph_data.get("input_levels", {})

        assert input_levels.get("eval_pairs") == "root"


class TestGraphvizExpandedPipelineRendering:
    """Prevent floating expanded pipeline nodes in Graphviz output."""

    def _build_metrics(self):
        retrieval = Pipeline(nodes=[retrieve, generate_answer], name="retrieval")
        retrieval.bind(vector_store=object(), top_k=3, llm=object())
        evaluation = Pipeline(
            nodes=[extract_query, retrieval.as_node(), evaluate_answer],
            name="evaluation",
        )
        evaluation_node = evaluation.as_node(
            name="batch_evaluation",
            map_over="eval_pairs",
            input_mapping={"eval_pairs": "eval_pair"},
            output_mapping={"evaluation_result": "evaluation_results"},
        )
        metrics = Pipeline(nodes=[evaluation_node, compute_metrics], name="metrics")
        return metrics

    def _expanded_pipeline_ids(self, graph_data):
        return [
            n["id"]
            for n in graph_data["nodes"]
            if n["node_type"] == "PIPELINE" and n.get("is_expanded")
        ]

    def _assert_not_rendered(self, dot_source: str, node_ids):
        for pid in node_ids:
            assert f"{pid} [" not in dot_source, f"Expanded pipeline node {pid} rendered"
            assert f"{pid} ->" not in dot_source, f"Edge emits expanded pipeline {pid}"
            assert f"-> {pid}" not in dot_source, f"Edge targets expanded pipeline {pid}"

    def test_no_floating_expanded_nodes_depth2(self):
        _ = pytest.importorskip("graphviz")

        metrics = self._build_metrics()
        graph_data = GraphSerializer(metrics).serialize(depth=2)
        expanded = self._expanded_pipeline_ids(graph_data)
        assert expanded, "Should have expanded pipeline nodes at depth=2"

        dot = metrics.visualize(
            engine="graphviz",
            return_type="graphviz",
            depth=2,
            group_inputs=True,
        )
        self._assert_not_rendered(dot.source, expanded)

    def test_no_floating_expanded_nodes_depth3(self):
        _ = pytest.importorskip("graphviz")

        metrics = self._build_metrics()
        graph_data = GraphSerializer(metrics).serialize(depth=3)
        expanded = self._expanded_pipeline_ids(graph_data)
        assert expanded, "Should have expanded pipeline nodes at depth=3"

        dot = metrics.visualize(
            engine="graphviz",
            return_type="graphviz",
            depth=3,
            group_inputs=True,
        )
        self._assert_not_rendered(dot.source, expanded)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
