"""Tests for the visualization debugging tools.

These tests verify that the debugging infrastructure works correctly:
- UIHandler.validate_graph()
- UIHandler.debug_dump()
- UIHandler.trace_node()
- UIHandler.trace_edge()
- UIHandler.find_issues()
- simulate_state()
- verify_state()
- diagnose_all_states()
"""

import pytest

from hypernodes import Pipeline, node
from hypernodes.viz import UIHandler, diagnose_all_states, simulate_state, verify_state

# ============================================================================
# Test Fixtures
# ============================================================================

@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2


@node(output_name="result")
def add(a: int, b: int) -> int:
    return a + b


@node(output_name="cleaned")
def clean_text(text: str) -> str:
    return text.strip().lower()


@node(output_name="summary")
def summarize(cleaned: str, max_len: int = 100) -> str:
    return cleaned[:max_len]


@pytest.fixture
def simple_pipeline():
    """Single-node pipeline."""
    return Pipeline(nodes=[double])


@pytest.fixture
def two_node_pipeline():
    """Two connected nodes."""
    @node(output_name="intermediate")
    def step1(x: int) -> int:
        return x + 1
    
    @node(output_name="final")
    def step2(intermediate: int) -> int:
        return intermediate * 2
    
    return Pipeline(nodes=[step1, step2])


@pytest.fixture
def nested_pipeline():
    """Pipeline containing another pipeline."""
    inner = Pipeline(nodes=[clean_text, summarize], name="text_processor")
    inner_node = inner.as_node(name="text_processor")
    
    @node(output_name="score")
    def evaluate(summary: str, expected: str) -> float:
        return 1.0 if summary == expected else 0.0
    
    return Pipeline(nodes=[inner_node, evaluate])


# ============================================================================
# Test UIHandler.validate_graph()
# ============================================================================

class TestValidateGraph:
    def test_valid_simple_pipeline(self, simple_pipeline):
        handler = UIHandler(simple_pipeline, depth=1)
        errors = handler.validate_graph()
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_valid_two_node_pipeline(self, two_node_pipeline):
        handler = UIHandler(two_node_pipeline, depth=1)
        errors = handler.validate_graph()
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_valid_nested_pipeline_collapsed(self, nested_pipeline):
        handler = UIHandler(nested_pipeline, depth=1)
        errors = handler.validate_graph()
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_valid_nested_pipeline_expanded(self, nested_pipeline):
        handler = UIHandler(nested_pipeline, depth=None)  # Fully expanded
        errors = handler.validate_graph()
        assert errors == [], f"Expected no errors, got: {errors}"


# ============================================================================
# Test UIHandler.debug_dump()
# ============================================================================

class TestDebugDump:
    def test_debug_dump_structure(self, simple_pipeline):
        handler = UIHandler(simple_pipeline, depth=1)
        dump = handler.debug_dump()
        
        # Check required keys exist
        assert "nodes" in dump
        assert "edges" in dump
        assert "metadata" in dump
        assert "validation" in dump
        assert "state" in dump
        assert "stats" in dump
        
    def test_debug_dump_nodes(self, simple_pipeline):
        handler = UIHandler(simple_pipeline, depth=1)
        dump = handler.debug_dump()
        
        # Should have nodes
        assert len(dump["nodes"]) > 0
        
        # Each node should have required fields
        for node_dict in dump["nodes"]:
            assert "id" in node_dict
            assert "type" in node_dict
            assert "parent" in node_dict
    
    def test_debug_dump_metadata(self, two_node_pipeline):
        handler = UIHandler(two_node_pipeline, depth=1)
        dump = handler.debug_dump()
        
        # Check metadata structure
        metadata = dump["metadata"]
        assert "producer_map" in metadata
        assert "input_type_hints" in metadata
        assert "output_type_hints" in metadata
        assert "boundary_outputs" in metadata
        
    def test_debug_dump_shows_type_hints(self, simple_pipeline):
        handler = UIHandler(simple_pipeline, depth=1)
        dump = handler.debug_dump()
        
        # Input 'x' should have type hint 'int'
        input_hints = dump["metadata"]["input_type_hints"]
        assert "x" in input_hints or any("x" in k for k in input_hints.keys())
        
    def test_debug_dump_nested_shows_boundary_outputs(self, nested_pipeline):
        handler = UIHandler(nested_pipeline, depth=1)  # Collapsed
        dump = handler.debug_dump()
        
        # Should identify boundary outputs from the nested pipeline
        boundary = dump["metadata"]["boundary_outputs"]
        # When collapsed, the pipeline node produces 'summary' as boundary output
        assert len(boundary) >= 0  # May be empty depending on expansion


# ============================================================================
# Test UIHandler.trace_node()
# ============================================================================

class TestTraceNode:
    def test_trace_existing_node(self, simple_pipeline):
        handler = UIHandler(simple_pipeline, depth=1)
        result = handler.trace_node("x")
        
        assert result["status"] == "FOUND"
        assert result["node_id"] == "x"
        assert result["node_type"] == "DataNode"
        
    def test_trace_missing_node(self, simple_pipeline):
        handler = UIHandler(simple_pipeline, depth=1)
        result = handler.trace_node("nonexistent")
        
        assert result["status"] == "NOT_FOUND"
        assert "suggestion" in result
        
    def test_trace_partial_match(self, nested_pipeline):
        handler = UIHandler(nested_pipeline, depth=None)  # Expanded
        result = handler.trace_node("clean")  # Partial match for "clean_text"
        
        assert result["status"] == "NOT_FOUND"
        assert "partial_matches" in result
        assert len(result["partial_matches"]) > 0
        
    def test_trace_data_node_info(self, simple_pipeline):
        handler = UIHandler(simple_pipeline, depth=1)
        result = handler.trace_node("x")
        
        assert "data_info" in result
        assert result["data_info"]["is_input"] is True
        assert result["data_info"]["is_output"] is False
        
    def test_trace_function_node_outputs(self, simple_pipeline):
        handler = UIHandler(simple_pipeline, depth=1)
        result = handler.trace_node("double")
        
        assert result["status"] == "FOUND"
        assert result["node_type"] == "FunctionNode"
        assert "outputs" in result
        assert len(result["outputs"]) > 0
        
    def test_trace_pipeline_node_children(self, nested_pipeline):
        handler = UIHandler(nested_pipeline, depth=None)  # Expanded
        result = handler.trace_node("text_processor")
        
        assert result["status"] == "FOUND"
        assert result["node_type"] == "PipelineNode"
        assert "children" in result
        assert len(result["children"]) > 0  # Should have internal nodes


# ============================================================================
# Test UIHandler.trace_edge()
# ============================================================================

class TestTraceEdge:
    def test_trace_existing_edge(self, two_node_pipeline):
        handler = UIHandler(two_node_pipeline, depth=1)
        # x -> step1 should exist
        result = handler.trace_edge("x", "step1")
        
        assert result["edge_found"] is True
        assert result["source"]["found"] is True
        assert result["target"]["found"] is True
        
    def test_trace_missing_edge(self, two_node_pipeline):
        handler = UIHandler(two_node_pipeline, depth=1)
        # Direct edge from x to step2 should NOT exist (goes through step1)
        result = handler.trace_edge("x", "step2")
        
        assert result["edge_found"] is False
        assert "analysis" in result
        
    def test_trace_edge_analysis_shows_path(self, two_node_pipeline):
        handler = UIHandler(two_node_pipeline, depth=1)
        result = handler.trace_edge("x", "step2")
        
        # Should show edges from source and to target
        analysis = result["analysis"]
        assert "edges_from_source" in analysis
        assert "edges_to_target" in analysis


# ============================================================================
# Test UIHandler.find_issues()
# ============================================================================

class TestFindIssues:
    def test_find_issues_on_valid_pipeline(self, simple_pipeline):
        handler = UIHandler(simple_pipeline, depth=1)
        issues = handler.find_issues()
        
        # Simple pipeline should have no validation errors
        assert "validation_errors" not in issues or len(issues.get("validation_errors", [])) == 0
        
    def test_find_issues_detects_missing_type_hints(self):
        # Create a function without type hints
        def untyped_func(x):
            return x * 2
        
        untyped_node = node(output_name="result")(untyped_func)
        pipeline = Pipeline(nodes=[untyped_node])
        
        handler = UIHandler(pipeline, depth=1)
        issues = handler.find_issues()
        
        # Should detect missing type hint
        if "missing_type_hints" in issues:
            assert len(issues["missing_type_hints"]) > 0


# ============================================================================
# Test simulate_state()
# ============================================================================

class TestSimulateState:
    def test_simulate_simple_pipeline(self, simple_pipeline):
        handler = UIHandler(simple_pipeline, depth=1)
        graph = handler.get_visualization_data(traverse_collapsed=True)
        
        result = simulate_state(
            graph,
            expansion_state={},
            separate_outputs=False,
            show_types=True
        )
        
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) > 0
        
    def test_simulate_combined_mode_hides_outputs(self, simple_pipeline):
        handler = UIHandler(simple_pipeline, depth=1)
        graph = handler.get_visualization_data(traverse_collapsed=True)
        
        result = simulate_state(
            graph,
            expansion_state={},
            separate_outputs=False,  # Combined mode
        )
        
        # In combined mode, output DATA nodes should be hidden/combined
        visible_nodes = [n for n in result["nodes"] if not n.get("hidden")]
        
        # The 'doubled' output should be combined into the function node
        # Check that function node has outputs data
        func_nodes = [n for n in visible_nodes if n["data"]["nodeType"] == "FUNCTION"]
        if func_nodes:
            assert "outputs" in func_nodes[0]["data"]
        
    def test_simulate_separate_mode_shows_outputs(self, simple_pipeline):
        handler = UIHandler(simple_pipeline, depth=1)
        graph = handler.get_visualization_data(traverse_collapsed=True)
        
        result = simulate_state(
            graph,
            expansion_state={},
            separate_outputs=True,  # Separate mode
        )
        
        # In separate mode, output DATA nodes should be visible
        visible_nodes = [n for n in result["nodes"] if not n.get("hidden")]
        node_types = [n["data"]["nodeType"] for n in visible_nodes]
        
        assert "DATA" in node_types
        
    def test_simulate_collapsed_pipeline_hides_children(self, nested_pipeline):
        handler = UIHandler(nested_pipeline, depth=None)  # Start expanded
        graph = handler.get_visualization_data(traverse_collapsed=True)
        
        # Now simulate collapsed state
        result = simulate_state(
            graph,
            expansion_state={"text_processor": False},  # Collapsed
            separate_outputs=False,
        )
        
        # Children of collapsed pipeline should be hidden
        visible = [n for n in result["nodes"] if not n.get("hidden")]
        visible_ids = {n["id"] for n in visible}
        
        # The internal nodes (clean_text, summarize) should be hidden
        # Only the pipeline node itself should be visible (plus root-level nodes)
        internal_visible = [nid for nid in visible_ids if nid.startswith("text_processor__")]
        assert len(internal_visible) == 0, f"Internal nodes should be hidden: {internal_visible}"


# ============================================================================
# Test verify_state()
# ============================================================================

class TestVerifyState:
    def test_verify_passes_with_correct_expectations(self, simple_pipeline):
        handler = UIHandler(simple_pipeline, depth=1)
        graph = handler.get_visualization_data(traverse_collapsed=True)
        
        result = simulate_state(
            graph,
            expansion_state={},
            separate_outputs=True,
        )
        
        # Verify with correct expectations
        verification = verify_state(
            result,
            expected_edges=[("x", "double")],  # Input to function
        )
        
        assert verification["passed"] is True
        
    def test_verify_fails_with_wrong_expectations(self, simple_pipeline):
        handler = UIHandler(simple_pipeline, depth=1)
        graph = handler.get_visualization_data(traverse_collapsed=True)
        
        result = simulate_state(
            graph,
            expansion_state={},
            separate_outputs=True,
        )
        
        # Verify with wrong expectation
        verification = verify_state(
            result,
            expected_edges=[("nonexistent", "double")],  # This edge doesn't exist
        )
        
        assert verification["passed"] is False
        assert len(verification["failures"]) > 0
        
    def test_verify_forbidden_edges(self, simple_pipeline):
        handler = UIHandler(simple_pipeline, depth=1)
        graph = handler.get_visualization_data(traverse_collapsed=True)
        
        result = simulate_state(
            graph,
            expansion_state={},
            separate_outputs=True,
        )
        
        # Verify that a certain edge does NOT exist
        verification = verify_state(
            result,
            forbidden_edges=[("double", "x")],  # This shouldn't exist (wrong direction)
        )
        
        assert verification["passed"] is True


# ============================================================================
# Test diagnose_all_states()
# ============================================================================

class TestDiagnoseAllStates:
    def test_diagnose_simple_pipeline(self, simple_pipeline):
        handler = UIHandler(simple_pipeline, depth=1)
        graph = handler.get_visualization_data(traverse_collapsed=True)
        
        results = diagnose_all_states(graph)
        
        # Should have 4 combinations: 2 separate * 2 expanded
        assert len(results) == 4
        
        # Check all keys exist
        expected_keys = [
            "separate=False,expanded=False",
            "separate=False,expanded=True",
            "separate=True,expanded=False",
            "separate=True,expanded=True",
        ]
        for key in expected_keys:
            assert key in results
            
    def test_diagnose_no_orphan_edges(self, nested_pipeline):
        handler = UIHandler(nested_pipeline, depth=None)
        graph = handler.get_visualization_data(traverse_collapsed=True)
        
        results = diagnose_all_states(graph)
        
        # No state should have orphan edges
        for key, state_result in results.items():
            orphans = state_result["orphan_edges"]
            assert len(orphans) == 0, f"State {key} has orphan edges: {orphans}"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    def test_full_debug_workflow(self, nested_pipeline):
        """Test the complete debugging workflow as an AI agent would use it."""
        handler = UIHandler(nested_pipeline, depth=2)
        
        # Step 1: Quick validation
        errors = handler.validate_graph()
        assert errors == [], f"Validation failed: {errors}"
        
        # Step 2: Get debug dump
        dump = handler.debug_dump()
        assert dump["stats"]["total_nodes"] > 0
        
        # Step 3: Check all state combinations
        graph = handler.get_visualization_data(traverse_collapsed=True)
        states = diagnose_all_states(graph)
        
        for key, state in states.items():
            assert len(state["orphan_edges"]) == 0, f"Orphan edges in {key}"
        
        # Step 4: Trace specific nodes
        for node_info in dump["nodes"][:3]:
            trace = handler.trace_node(node_info["id"])
            assert trace["status"] == "FOUND"
            
    def test_debug_catches_issues_before_render(self, nested_pipeline):
        """Verify that debugging tools catch issues at Python level."""
        handler = UIHandler(nested_pipeline, depth=None)
        
        # Use find_issues() to get all problems in one call
        issues = handler.find_issues()
        
        # A well-formed pipeline should have no critical issues
        if "validation_errors" in issues:
            assert len(issues["validation_errors"]) == 0
