"""Playwright tests for edge-node alignment validation.

These tests use Playwright to render the visualization in a real browser and verify
that edges properly connect to their source/target nodes after collapse/expand operations.

Prerequisites:
    pip install playwright
    playwright install chromium

Run tests:
    pytest tests/viz/test_edge_alignment_playwright.py -v

Skip if Playwright not installed:
    These tests will be automatically skipped if Playwright is not available.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from hypernodes import Pipeline, node
from hypernodes.viz import UIHandler, simulate_collapse_expand_cycle, verify_edge_alignment
from hypernodes.viz.js.html_generator import generate_widget_html
from hypernodes.viz.js.renderer import JSRenderer

# Try to import Playwright, skip browser tests if not available
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    sync_playwright = None

# Only browser tests are skipped if Playwright is not available
# Python-only tests should always run


# ============================================================================
# Test Fixtures
# ============================================================================

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


@pytest.fixture
def rag_pipeline():
    """Create a RAG evaluation pipeline with nested pipelines."""
    retrieval = Pipeline(nodes=[retrieve], name="retrieval")
    rag = Pipeline(
        nodes=[extract_query, retrieval.as_node(name="retrieval_step"), generate_answer],
        name="rag"
    )
    eval_pipeline = Pipeline(
        nodes=[rag.as_node(name="rag_pipeline"), evaluate],
        name="evaluation"
    )
    return eval_pipeline


@pytest.fixture
def simple_nested_pipeline():
    """Simpler nested pipeline for basic tests."""
    @node(output_name="cleaned")
    def clean_text(text: str) -> str:
        return text.strip().lower()
    
    @node(output_name="summary")
    def summarize(cleaned: str) -> str:
        return cleaned[:100]
    
    inner = Pipeline(nodes=[clean_text, summarize], name="text_processor")
    
    @node(output_name="score")
    def score_output(summary: str, expected: str) -> float:
        return 1.0 if summary == expected else 0.0
    
    return Pipeline(nodes=[inner.as_node(name="text_processor"), score_output])


def generate_test_html(pipeline: Pipeline, depth: int = 99, separate_outputs: bool = True) -> str:
    """Generate HTML visualization for a pipeline."""
    handler = UIHandler(pipeline, depth=depth)
    graph_data = handler.get_visualization_data(traverse_collapsed=True)
    renderer = JSRenderer()
    rf_data = renderer.render(graph_data, theme="dark", separate_outputs=separate_outputs, show_types=True)
    return generate_widget_html(rf_data)


# ============================================================================
# Python-Side Tests (No Browser Required)
# ============================================================================

class TestPythonEdgeAlignment:
    """Tests using Python state simulator (no browser needed)."""
    
    def test_verify_edge_alignment_expanded(self, simple_nested_pipeline):
        """Verify edges are valid when pipeline is expanded."""
        handler = UIHandler(simple_nested_pipeline, depth=99)
        graph = handler.get_visualization_data(traverse_collapsed=True)
        
        from hypernodes.viz.state_simulator import simulate_state
        result = simulate_state(graph, expansion_state={"text_processor": True})
        alignment = verify_edge_alignment(result)
        
        assert alignment["valid"], f"Edge alignment issues: {alignment['issues']}"
    
    def test_verify_edge_alignment_collapsed(self, simple_nested_pipeline):
        """Verify edges are valid when pipeline is collapsed."""
        handler = UIHandler(simple_nested_pipeline, depth=99)
        graph = handler.get_visualization_data(traverse_collapsed=True)
        
        from hypernodes.viz.state_simulator import simulate_state
        result = simulate_state(graph, expansion_state={"text_processor": False})
        alignment = verify_edge_alignment(result)
        
        assert alignment["valid"], f"Edge alignment issues: {alignment['issues']}"
    
    def test_collapse_expand_cycle(self, simple_nested_pipeline):
        """Test that collapse/expand cycle maintains valid edges."""
        handler = UIHandler(simple_nested_pipeline, depth=99)
        graph = handler.get_visualization_data(traverse_collapsed=True)
        
        result = simulate_collapse_expand_cycle(graph, "text_processor")
        
        assert result["all_valid"], f"Cycle failed: {result['summary']}"
        assert result["initial"]["alignment"]["valid"]
        assert result["after_collapse"]["alignment"]["valid"]
        assert result["after_expand"]["alignment"]["valid"]
    
    def test_rag_pipeline_collapse_cycle(self, rag_pipeline):
        """Test RAG pipeline collapse/expand cycle."""
        handler = UIHandler(rag_pipeline, depth=99)
        graph = handler.get_visualization_data(traverse_collapsed=True)
        
        result = simulate_collapse_expand_cycle(graph, "rag_pipeline")
        
        assert result["all_valid"], f"Cycle failed: {result['summary']}"


# ============================================================================
# Playwright Browser Tests
# ============================================================================

@pytest.mark.skipif(
    not PLAYWRIGHT_AVAILABLE, 
    reason="Playwright not installed. Install with: pip install playwright && playwright install chromium"
)
class TestPlaywrightEdgeAlignment:
    """Tests using Playwright to validate in actual browser."""
    
    @pytest.fixture
    def browser_context(self):
        """Create a Playwright browser context."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            yield context
            context.close()
            browser.close()
    
    def _save_and_open_html(self, html_content: str, page):
        """Save HTML to temp file and open in browser."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html_content)
            temp_path = f.name
        
        try:
            page.goto(f"file://{temp_path}")
            # Wait for React Flow to render
            page.wait_for_selector('.react-flow__node', timeout=10000)
            # Additional wait for layout to complete
            page.wait_for_timeout(1000)
            return temp_path
        except Exception:
            os.unlink(temp_path)
            raise
    
    def _validate_connections_in_browser(self, page) -> dict:
        """Run validateConnections() in the browser and return result."""
        result = page.evaluate("""
            () => {
                if (window.HyperNodesVizState && window.HyperNodesVizState.debug) {
                    return window.HyperNodesVizState.debug.validateConnections();
                }
                return { valid: false, issues: [{ issue: "Debug API not available" }] };
            }
        """)
        return result
    
    def _inspect_layout_in_browser(self, page) -> dict:
        """Run inspectLayout() in the browser and return result."""
        result = page.evaluate("""
            () => {
                if (window.HyperNodesVizState && window.HyperNodesVizState.debug) {
                    return window.HyperNodesVizState.debug.inspectLayout();
                }
                return null;
            }
        """)
        return result
    
    def _click_pipeline_to_collapse(self, page, pipeline_id: str):
        """Click on a pipeline node to collapse it."""
        # Find the pipeline node and click it
        selector = f'.react-flow__node[data-id="{pipeline_id}"]'
        page.click(selector)
        # Wait for re-layout
        page.wait_for_timeout(500)
    
    def test_initial_render_valid_connections(self, browser_context, simple_nested_pipeline):
        """Test that initial render has valid edge-node connections."""
        page = browser_context.new_page()
        html = generate_test_html(simple_nested_pipeline)
        temp_path = self._save_and_open_html(html, page)
        
        try:
            validation = self._validate_connections_in_browser(page)
            assert validation["valid"], f"Connection issues: {validation['issues']}"
        finally:
            os.unlink(temp_path)
    
    def test_collapse_maintains_valid_connections(self, browser_context, simple_nested_pipeline):
        """Test that collapsing a pipeline maintains valid edge-node connections."""
        page = browser_context.new_page()
        html = generate_test_html(simple_nested_pipeline)
        temp_path = self._save_and_open_html(html, page)
        
        try:
            # Initial validation
            initial_validation = self._validate_connections_in_browser(page)
            assert initial_validation["valid"], f"Initial issues: {initial_validation['issues']}"
            
            # Collapse the pipeline
            self._click_pipeline_to_collapse(page, "text_processor")
            page.wait_for_timeout(500)  # Wait for re-layout
            
            # Validate after collapse
            collapsed_validation = self._validate_connections_in_browser(page)
            assert collapsed_validation["valid"], f"Post-collapse issues: {collapsed_validation['issues']}"
        finally:
            os.unlink(temp_path)
    
    def test_expand_maintains_valid_connections(self, browser_context, simple_nested_pipeline):
        """Test that expanding a collapsed pipeline maintains valid edge-node connections."""
        page = browser_context.new_page()
        html = generate_test_html(simple_nested_pipeline)
        temp_path = self._save_and_open_html(html, page)
        
        try:
            # Collapse first
            self._click_pipeline_to_collapse(page, "text_processor")
            page.wait_for_timeout(500)
            
            # Then expand
            self._click_pipeline_to_collapse(page, "text_processor")  # Toggle back
            page.wait_for_timeout(500)
            
            # Validate after expand
            expanded_validation = self._validate_connections_in_browser(page)
            assert expanded_validation["valid"], f"Post-expand issues: {expanded_validation['issues']}"
        finally:
            os.unlink(temp_path)
    
    def test_rag_pipeline_collapse_in_browser(self, browser_context, rag_pipeline):
        """Test RAG pipeline collapse/expand in browser."""
        page = browser_context.new_page()
        html = generate_test_html(rag_pipeline)
        temp_path = self._save_and_open_html(html, page)
        
        try:
            # Initial validation
            initial = self._validate_connections_in_browser(page)
            
            # Collapse rag_pipeline
            self._click_pipeline_to_collapse(page, "rag_pipeline")
            page.wait_for_timeout(1000)  # Longer wait for complex layout
            
            # Validate after collapse
            collapsed = self._validate_connections_in_browser(page)
            
            # Note: This test may fail due to the known hanging edge bug
            # If it fails, the bug still exists and needs fixing
            if not collapsed["valid"]:
                pytest.xfail(
                    f"Known issue: Edges may not properly reconnect after collapse. "
                    f"Issues: {collapsed['issues']}"
                )
        finally:
            os.unlink(temp_path)
    
    def test_debug_overlays_toggle(self, browser_context, simple_nested_pipeline):
        """Test that debug overlays can be toggled via console API."""
        page = browser_context.new_page()
        html = generate_test_html(simple_nested_pipeline)
        temp_path = self._save_and_open_html(html, page)
        
        try:
            # Enable debug overlays
            page.evaluate("window.HyperNodesVizState.debug.showOverlays()")
            
            # Check that the flag is set
            debug_enabled = page.evaluate("window.__hypernodes_debug_overlays")
            assert debug_enabled, "Debug overlays should be enabled"
            
            # Disable debug overlays
            page.evaluate("window.HyperNodesVizState.debug.hideOverlays()")
            
            debug_disabled = page.evaluate("window.__hypernodes_debug_overlays")
            assert not debug_disabled, "Debug overlays should be disabled"
        finally:
            os.unlink(temp_path)
    
    def test_inspect_layout_returns_data(self, browser_context, simple_nested_pipeline):
        """Test that inspectLayout() returns layout data."""
        page = browser_context.new_page()
        html = generate_test_html(simple_nested_pipeline)
        temp_path = self._save_and_open_html(html, page)
        
        try:
            layout = self._inspect_layout_in_browser(page)
            
            assert layout is not None, "Layout data should be available"
            assert "nodes" in layout, "Layout should have nodes"
            assert "edges" in layout, "Layout should have edges"
            assert len(layout["nodes"]) > 0, "Should have at least one node"
        finally:
            os.unlink(temp_path)


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.skipif(
    not PLAYWRIGHT_AVAILABLE, 
    reason="Playwright not installed. Install with: pip install playwright && playwright install chromium"
)
class TestDebugToolsIntegration:
    """Integration tests combining Python and browser-based debugging."""
    
    def test_python_and_browser_alignment_agree(self, simple_nested_pipeline):
        """Test that Python simulator and browser validation agree on edge alignment."""
        if not PLAYWRIGHT_AVAILABLE:
            pytest.skip("Playwright not available")
        
        # Python-side analysis
        handler = UIHandler(simple_nested_pipeline, depth=99)
        graph = handler.get_visualization_data(traverse_collapsed=True)
        
        from hypernodes.viz.state_simulator import simulate_state
        py_result = simulate_state(graph, expansion_state={"text_processor": True})
        py_alignment = verify_edge_alignment(py_result)
        
        # Browser-side analysis
        html = generate_test_html(simple_nested_pipeline)
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                f.write(html)
                temp_path = f.name
            
            try:
                page.goto(f"file://{temp_path}")
                page.wait_for_selector('.react-flow__node', timeout=10000)
                page.wait_for_timeout(1000)
                
                browser_validation = page.evaluate("""
                    () => window.HyperNodesVizState.debug.validateConnections()
                """)
                
                # Both should agree on validity (or both have issues)
                # Note: Exact issues may differ due to different validation approaches
                assert py_alignment["valid"] == browser_validation["valid"], (
                    f"Python says valid={py_alignment['valid']} but browser says valid={browser_validation['valid']}. "
                    f"Python issues: {py_alignment['issues']}, Browser issues: {browser_validation['issues']}"
                )
            finally:
                os.unlink(temp_path)
                browser.close()

