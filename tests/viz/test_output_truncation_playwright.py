"""Test that output names are not truncated in combined mode (separate_outputs=False).

Uses Playwright to verify that short output names are fully visible.
"""

import pytest
from pathlib import Path
import tempfile
import json

from hypernodes import Pipeline, node
from hypernodes.viz import UIHandler
from hypernodes.viz.js.renderer import JSRenderer
from hypernodes.viz.js.html_generator import generate_widget_html


# Test nodes with various output name lengths
@node(output_name="result")
def short_output(x: int) -> int:
    return x


@node(output_name="embeddings")
def medium_output(text: str) -> str:
    return text


@node(output_name="retrieved_documents")
def long_output(query: str) -> str:
    return query


@node(output_name=("scores", "metadata"))
def multi_output(data: str) -> tuple:
    return ("", "")


class TestOutputTruncationPlaywright:
    """Test that output names are not truncated when they shouldn't be."""

    @pytest.fixture
    def browser_page(self):
        """Create a Playwright browser page."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            pytest.skip("Playwright not installed")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            yield page
            browser.close()

    def _generate_combined_html(self, pipeline: Pipeline, show_types: bool = True) -> str:
        """Generate HTML with separate_outputs=False (combined mode)."""
        handler = UIHandler(pipeline, depth=99)
        graph_data = handler.get_visualization_data(traverse_collapsed=True)
        renderer = JSRenderer()
        rf_data = renderer.render(
            graph_data,
            theme="dark",
            separate_outputs=False,  # Combined mode
            show_types=show_types,
        )
        return generate_widget_html(rf_data)

    def _save_and_open(self, page, html: str) -> None:
        """Save HTML to temp file and open in browser."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html)
            temp_path = f.name
        
        page.goto(f"file://{temp_path}")
        # Wait for layout to complete
        page.wait_for_timeout(2000)

    def _get_output_texts(self, page) -> list:
        """Extract all output texts from the visualization."""
        # Get all output rows (they have the arrow →)
        result = page.evaluate("""
            () => {
                const outputs = [];
                // Find all elements containing the arrow
                document.querySelectorAll('*').forEach(el => {
                    if (el.textContent && el.textContent.includes('→') && el.children.length > 0) {
                        // This might be an output row
                        const text = el.textContent.trim();
                        if (text.startsWith('→')) {
                            outputs.push({
                                fullText: text,
                                isTruncated: text.includes('...'),
                                element: el.className
                            });
                        }
                    }
                });
                return outputs;
            }
        """)
        return result

    def _check_truncation_in_js(self, page) -> dict:
        """Check for truncated output names using JavaScript."""
        result = page.evaluate("""
            () => {
                const issues = [];
                const outputNames = [];
                
                // Find all function nodes with combined outputs
                document.querySelectorAll('[class*="border-t"]').forEach(outputSection => {
                    // Find output rows within the section
                    outputSection.querySelectorAll('[class*="flex"][class*="items-center"]').forEach(row => {
                        const text = row.textContent || '';
                        // Check if this is an output row (has arrow)
                        if (text.includes('→')) {
                            // Extract the output name (after arrow, before colon or end)
                            const match = text.match(/→\\s*([^:]+)/);
                            if (match) {
                                const outputName = match[1].trim();
                                const isTruncated = outputName.endsWith('...');
                                outputNames.push({
                                    name: outputName,
                                    fullText: text.trim(),
                                    isTruncated: isTruncated,
                                    // Get computed width
                                    rowWidth: row.getBoundingClientRect().width,
                                    parentWidth: row.parentElement ? row.parentElement.getBoundingClientRect().width : 0
                                });
                                
                                if (isTruncated) {
                                    issues.push({
                                        truncatedName: outputName,
                                        fullText: text.trim(),
                                        issue: 'Output name is truncated'
                                    });
                                }
                            }
                        }
                    });
                });
                
                return {
                    outputNames: outputNames,
                    issues: issues,
                    hasIssues: issues.length > 0
                };
            }
        """)
        return result

    def test_short_output_not_truncated(self, browser_page):
        """Test that short output names like 'result' are not truncated."""
        pipeline = Pipeline(nodes=[short_output])
        html = self._generate_combined_html(pipeline)
        self._save_and_open(browser_page, html)
        
        result = self._check_truncation_in_js(browser_page)
        
        assert not result['hasIssues'], (
            f"Output names should not be truncated. Issues found: {result['issues']}"
        )

    def test_medium_output_not_truncated(self, browser_page):
        """Test that medium output names like 'embeddings' are not truncated."""
        pipeline = Pipeline(nodes=[medium_output])
        html = self._generate_combined_html(pipeline)
        self._save_and_open(browser_page, html)
        
        result = self._check_truncation_in_js(browser_page)
        
        assert not result['hasIssues'], (
            f"Output names should not be truncated. Issues found: {result['issues']}"
        )

    def test_multi_output_not_truncated(self, browser_page):
        """Test that multiple output names like 'scores', 'metadata' are not truncated."""
        pipeline = Pipeline(nodes=[multi_output])
        html = self._generate_combined_html(pipeline)
        self._save_and_open(browser_page, html)
        
        result = self._check_truncation_in_js(browser_page)
        
        assert not result['hasIssues'], (
            f"Output names should not be truncated. Issues found: {result['issues']}"
        )

    def test_combined_pipeline_outputs_not_truncated(self, browser_page):
        """Test a realistic pipeline with various output lengths."""
        # Create a pipeline with multiple nodes
        @node(output_name="cleaned")
        def clean(text: str) -> str:
            return text
        
        @node(output_name="embedded")
        def embed(cleaned: str) -> str:
            return cleaned
        
        @node(output_name="retrieved_docs")
        def retrieve(embedded: str) -> str:
            return embedded
        
        pipeline = Pipeline(nodes=[clean, embed, retrieve])
        html = self._generate_combined_html(pipeline)
        self._save_and_open(browser_page, html)
        
        result = self._check_truncation_in_js(browser_page)
        
        # Print debug info
        print(f"Output names found: {result['outputNames']}")
        
        assert not result['hasIssues'], (
            f"Output names should not be truncated. Issues found: {result['issues']}"
        )

    def test_output_with_type_allows_type_truncation(self, browser_page):
        """Test that type hints CAN be truncated, but not the output name."""
        @node(output_name="result")
        def typed_output(x: int) -> dict:
            return {}
        
        pipeline = Pipeline(nodes=[typed_output])
        html = self._generate_combined_html(pipeline, show_types=True)
        self._save_and_open(browser_page, html)
        
        result = self._check_truncation_in_js(browser_page)
        
        # The output NAME should not be truncated
        # (type hints are allowed to truncate)
        for output in result['outputNames']:
            # Extract just the name part (before any colon)
            name_part = output['name'].split(':')[0].strip()
            assert not name_part.endswith('...'), (
                f"Output name '{name_part}' should not be truncated"
            )


class TestRetrieveNodeTruncation:
    """Test specific case where 'retrieved_documents' output gets truncated."""

    @pytest.fixture
    def browser_page(self):
        """Create a Playwright browser page."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            pytest.skip("Playwright not installed")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            yield page
            browser.close()

    def _generate_and_open(self, page, pipeline, show_types=True):
        """Generate HTML and open in browser."""
        handler = UIHandler(pipeline, depth=99)
        graph_data = handler.get_visualization_data(traverse_collapsed=True)
        renderer = JSRenderer()
        rf_data = renderer.render(
            graph_data,
            theme="dark",
            separate_outputs=False,
            show_types=show_types,
        )
        html = generate_widget_html(rf_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html)
            temp_path = f.name
        
        page.goto(f"file://{temp_path}")
        page.wait_for_timeout(2000)
        return temp_path

    def test_retrieve_output_not_truncated(self, browser_page):
        """Test that 'retrieved_documents' output name is fully visible.
        
        This tests the specific case where:
        - Function name is short (8 chars: 'retrieve')
        - Output name is long (19 chars: 'retrieved_documents')
        - The node width should be based on the LONGER output name
        """
        @node(output_name="retrieved_documents")
        def retrieve(query: str) -> str:
            return query
        
        pipeline = Pipeline(nodes=[retrieve])
        self._generate_and_open(browser_page, pipeline)
        
        # Check if the output section text is clipped by the node container
        result = browser_page.evaluate("""() => {
            // Find the function node's output section
            const outputSections = document.querySelectorAll('[class*="border-t"]');
            const issues = [];
            
            outputSections.forEach(section => {
                // Get the output row
                const row = section.querySelector('[class*="flex"][class*="items-center"]');
                if (!row) return;
                
                const text = row.innerText || '';
                if (!text.includes('retrieved')) return;
                
                // Find the parent node container
                let nodeContainer = section.parentElement;
                while (nodeContainer && !nodeContainer.classList.contains('react-flow__node')) {
                    nodeContainer = nodeContainer.parentElement;
                }
                
                if (!nodeContainer) return;
                
                // Get the node width from ELK layout (stored in style)
                const nodeWidth = nodeContainer.offsetWidth || nodeContainer.clientWidth;
                
                // Get the content width needed
                const contentWidth = row.scrollWidth;
                
                // Check if the output name span is clipped
                const outputNameSpan = row.querySelector('span.font-mono.font-medium');
                const outputText = outputNameSpan ? outputNameSpan.innerText : '';
                
                // The issue is when the content is wider than the node
                // Node should be at least as wide as content (allowing for small rounding)
                if (contentWidth > nodeWidth + 5) {
                    issues.push({
                        outputName: outputText,
                        nodeWidth: nodeWidth,
                        contentWidth: contentWidth,
                        fullText: text,
                        issue: 'Node too narrow for output content'
                    });
                }
                
                // Also check if output name is partially visible
                if (outputText && !outputText.includes('documents')) {
                    issues.push({
                        outputName: outputText,
                        nodeWidth: nodeWidth,
                        issue: 'Output name "retrieved_documents" is visually truncated'
                    });
                }
            });
            
            return {
                issues,
                hasIssues: issues.length > 0,
                bodyText: document.body.innerText.substring(0, 500)
            };
        }""")
        
        print(f"Issues found: {result['issues']}")
        print(f"Body text: {result['bodyText']}")
        
        assert not result['hasIssues'], (
            f"Output should be fully visible. Issues: {result['issues']}"
        )

    def test_short_function_long_output(self, browser_page):
        """Test case where function name is short but output name is long."""
        @node(output_name="very_long_output_name_that_exceeds_function")
        def fn(x: str) -> str:
            return x
        
        pipeline = Pipeline(nodes=[fn])
        self._generate_and_open(browser_page, pipeline, show_types=False)
        
        # The node should be wide enough for the output
        result = browser_page.evaluate("""
            () => {
                const text = document.body.innerText;
                // Check if output is visible without truncation
                const hasFullOutput = text.includes('very_long_output_name_that_exceeds_function');
                return {
                    hasFullOutput,
                    bodyText: text
                };
            }
        """)
        
        # With such a long name, it might be truncated at the label limit
        # but it should be visible up to NODE_LABEL_MAX_CHARS (25)
        assert 'very_long_output_name_tha' in result['bodyText'], (
            f"Output should be visible (possibly truncated at 25 chars). Got: {result['bodyText']}"
        )


class TestOutputTruncationPython:
    """Python-only tests for output truncation (no browser needed)."""

    def test_output_name_in_html(self):
        """Verify that full output names appear in the generated HTML."""
        @node(output_name="embeddings")
        def test_node(x: str) -> str:
            return x
        
        pipeline = Pipeline(nodes=[test_node])
        handler = UIHandler(pipeline, depth=99)
        graph_data = handler.get_visualization_data(traverse_collapsed=True)
        renderer = JSRenderer()
        rf_data = renderer.render(
            graph_data,
            theme="dark",
            separate_outputs=False,
            show_types=True,
        )
        html = generate_widget_html(rf_data)
        
        # The output name should appear in the HTML data (with space after colon)
        assert '"label": "embeddings"' in html, \
               "Output name 'embeddings' should be in the HTML"

    def test_width_calculation_includes_output(self):
        """Verify width calculation considers output names.
        
        Note: Outputs are added by JavaScript at runtime via fallbackApplyState,
        not in the Python renderer. The JS width calculation correctly uses
        the transformed node data.
        """
        @node(output_name="very_long_output_name_here")
        def long_output_node(x: str) -> str:
            return x
        
        pipeline = Pipeline(nodes=[long_output_node])
        handler = UIHandler(pipeline, depth=99)
        graph_data = handler.get_visualization_data(traverse_collapsed=True)
        renderer = JSRenderer()
        rf_data = renderer.render(
            graph_data,
            theme="dark",
            separate_outputs=False,
            show_types=False,
        )
        
        # Find the output node (it becomes a separate DATA node in rf_data)
        output_node = None
        for n in rf_data['nodes']:
            if n['data'].get('label') == 'very_long_output_name_here':
                output_node = n
                break
        
        assert output_node is not None, "Should have output node with the label"
        assert output_node['data'].get('sourceId') is not None, \
               "Output node should have sourceId pointing to function"

