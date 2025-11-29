"""
Tests for VSCode notebook compatibility of visualization widgets.

These tests ensure the generated HTML works in VSCode's restricted iframe environment.
See skills/viz/js/VSCODE_NOTEBOOK_COMPATIBILITY.md for details.
"""

import html as html_module
import re

import pytest

from hypernodes import Pipeline, node
from hypernodes.viz.js.html_generator import generate_widget_html
from hypernodes.viz.js.renderer import JSRenderer
from hypernodes.viz.ui_handler import UIHandler
from hypernodes.viz.visualization_widget import PipelineWidget


@pytest.fixture
def simple_pipeline():
    """Create a simple test pipeline."""
    @node(output_name="y")
    def add_one(x: int) -> int:
        return x + 1

    @node(output_name="z")
    def double(y: int) -> int:
        return y * 2

    return Pipeline([add_one, double])


@pytest.fixture
def nested_pipeline():
    """Create a nested pipeline for more complex testing."""
    @node(output_name="cleaned")
    def clean(text: str) -> str:
        return text.strip()

    @node(output_name="result")
    def process(cleaned: str) -> str:
        return cleaned.upper()

    inner = Pipeline([clean, process], name="inner")

    @node(output_name="final")
    def finalize(result: str) -> str:
        return f"[{result}]"

    return Pipeline([inner.as_node(), finalize], name="outer")


class TestNoModuleScripts:
    """Ensure no ES module scripts are in the generated HTML.
    
    Module scripts don't execute in VSCode notebook iframe srcdoc contexts
    due to CSP restrictions.
    """

    def test_html_generator_no_module_scripts(self, simple_pipeline):
        """Generated HTML should not contain <script type="module">."""
        handler = UIHandler(simple_pipeline, depth=99)
        graph_data = handler.get_visualization_data(traverse_collapsed=True)
        renderer = JSRenderer()
        rf_data = renderer.render(graph_data, theme="dark")
        html = generate_widget_html(rf_data)

        # Check for module script tags
        module_pattern = r'<script[^>]*type\s*=\s*["\']module["\'][^>]*>'
        matches = re.findall(module_pattern, html, re.IGNORECASE)
        
        assert len(matches) == 0, (
            f"Found {len(matches)} module script(s) in HTML. "
            "Module scripts don't work in VSCode notebook iframes. "
            "Use regular scripts with DOMContentLoaded instead."
        )

    def test_widget_no_module_scripts(self, simple_pipeline):
        """PipelineWidget HTML should not contain module scripts after unescaping."""
        widget = PipelineWidget(simple_pipeline)
        decoded = html_module.unescape(widget.value)

        module_pattern = r'<script[^>]*type\s*=\s*["\']module["\'][^>]*>'
        matches = re.findall(module_pattern, decoded, re.IGNORECASE)
        
        assert len(matches) == 0, (
            "Found module scripts in widget HTML. "
            "This will cause blank visualization in VSCode notebooks."
        )


class TestDOMContentLoaded:
    """Ensure scripts wait for DOM to be ready.
    
    Regular scripts execute immediately, but may reference DOM elements
    that come later in the HTML (like the graph-data script element).
    """

    def test_html_has_dom_content_loaded(self, simple_pipeline):
        """Generated HTML should wrap main script in DOMContentLoaded."""
        handler = UIHandler(simple_pipeline, depth=99)
        graph_data = handler.get_visualization_data(traverse_collapsed=True)
        renderer = JSRenderer()
        rf_data = renderer.render(graph_data, theme="dark")
        html = generate_widget_html(rf_data)

        assert "DOMContentLoaded" in html, (
            "HTML should use DOMContentLoaded event listener. "
            "Without it, scripts may execute before DOM elements exist, "
            "causing 'Cannot read properties of null' errors."
        )

    def test_widget_has_dom_content_loaded(self, simple_pipeline):
        """Widget HTML should have DOMContentLoaded after unescaping."""
        widget = PipelineWidget(simple_pipeline)
        decoded = html_module.unescape(widget.value)

        assert "DOMContentLoaded" in decoded, (
            "Widget HTML must use DOMContentLoaded for VSCode compatibility."
        )


class TestIIFEPattern:
    """Ensure scripts use IIFE pattern for proper scoping.
    
    IIFE (Immediately Invoked Function Expression) provides:
    - Strict mode
    - Variable scope isolation
    - Similar behavior to module scripts
    """

    def test_html_has_iife(self, simple_pipeline):
        """Generated HTML should use IIFE pattern."""
        handler = UIHandler(simple_pipeline, depth=99)
        graph_data = handler.get_visualization_data(traverse_collapsed=True)
        renderer = JSRenderer()
        rf_data = renderer.render(graph_data, theme="dark")
        html = generate_widget_html(rf_data)

        assert "(function()" in html, "HTML should use IIFE pattern"
        assert "'use strict'" in html, "IIFE should enable strict mode"

    def test_widget_has_iife(self, simple_pipeline):
        """Widget HTML should have IIFE after unescaping."""
        widget = PipelineWidget(simple_pipeline)
        decoded = html_module.unescape(widget.value)

        assert "(function()" in decoded, "Widget should use IIFE"
        assert "'use strict'" in decoded, "Widget IIFE should use strict mode"


class TestSrcdocAttribute:
    """Ensure widget uses srcdoc attribute, not base64 data URI.
    
    Base64 data URIs may be blocked by CSP in some environments.
    """

    def test_widget_uses_srcdoc(self, simple_pipeline):
        """Widget should use srcdoc attribute for iframe."""
        widget = PipelineWidget(simple_pipeline)
        
        assert 'srcdoc=' in widget.value, (
            "Widget should use srcdoc attribute, not data: URI. "
            "Data URIs may be blocked in VSCode notebooks."
        )
        
        # Should NOT use base64 data URI
        assert 'src="data:text/html;base64' not in widget.value, (
            "Widget should not use base64 data URI for iframe src."
        )


class TestGraphDataElement:
    """Ensure graph data element is properly placed and accessible."""

    def test_graph_data_element_exists(self, simple_pipeline):
        """HTML should contain graph-data script element."""
        handler = UIHandler(simple_pipeline, depth=99)
        graph_data = handler.get_visualization_data(traverse_collapsed=True)
        renderer = JSRenderer()
        rf_data = renderer.render(graph_data, theme="dark")
        html = generate_widget_html(rf_data)

        assert 'id="graph-data"' in html, "HTML must contain graph-data element"
        assert 'type="application/json"' in html, "graph-data should be JSON type"

    def test_graph_data_has_valid_json(self, simple_pipeline):
        """Graph data element should contain valid JSON."""
        import json
        
        handler = UIHandler(simple_pipeline, depth=99)
        graph_data = handler.get_visualization_data(traverse_collapsed=True)
        renderer = JSRenderer()
        rf_data = renderer.render(graph_data, theme="dark")
        html = generate_widget_html(rf_data)

        # Extract JSON from script tag
        match = re.search(
            r'<script[^>]*id="graph-data"[^>]*>(.*?)</script>',
            html,
            re.DOTALL
        )
        assert match, "Could not find graph-data script content"
        
        json_str = match.group(1)
        try:
            data = json.loads(json_str)
            assert "nodes" in data, "Graph data should have nodes"
            assert "edges" in data, "Graph data should have edges"
        except json.JSONDecodeError as e:
            pytest.fail(f"Graph data is not valid JSON: {e}")


class TestNestedPipeline:
    """Test compatibility with nested pipelines."""

    def test_nested_pipeline_no_module_scripts(self, nested_pipeline):
        """Nested pipeline HTML should not have module scripts."""
        widget = PipelineWidget(nested_pipeline, depth=99)
        decoded = html_module.unescape(widget.value)

        module_pattern = r'<script[^>]*type\s*=\s*["\']module["\'][^>]*>'
        matches = re.findall(module_pattern, decoded, re.IGNORECASE)
        
        assert len(matches) == 0

    def test_nested_pipeline_has_dom_content_loaded(self, nested_pipeline):
        """Nested pipeline HTML should have DOMContentLoaded."""
        widget = PipelineWidget(nested_pipeline, depth=99)
        decoded = html_module.unescape(widget.value)

        assert "DOMContentLoaded" in decoded


class TestFullCompatibilityCheck:
    """Comprehensive compatibility check combining all requirements."""

    def test_full_vscode_compatibility(self, simple_pipeline):
        """Run all VSCode compatibility checks on a widget."""
        widget = PipelineWidget(simple_pipeline)
        decoded = html_module.unescape(widget.value)

        issues = []

        # Check 1: No module scripts
        if re.search(r'<script[^>]*type=["\']module["\']', decoded):
            issues.append("Contains module scripts (blocked in VSCode)")

        # Check 2: Has DOMContentLoaded
        if "DOMContentLoaded" not in decoded:
            issues.append("Missing DOMContentLoaded (causes null errors)")

        # Check 3: Has IIFE
        if "(function()" not in decoded:
            issues.append("Missing IIFE pattern")

        # Check 4: Uses strict mode
        if "'use strict'" not in decoded:
            issues.append("Missing 'use strict'")

        # Check 5: Uses srcdoc
        if "srcdoc=" not in widget.value:
            issues.append("Not using srcdoc attribute")

        # Check 6: Has graph-data
        if 'id="graph-data"' not in decoded:
            issues.append("Missing graph-data element")

        assert len(issues) == 0, (
            "VSCode compatibility issues found:\n" +
            "\n".join(f"  - {issue}" for issue in issues)
        )


# Utility function that can be used for debugging
def check_vscode_compatibility(pipeline) -> dict:
    """
    Check a pipeline's visualization for VSCode compatibility.
    
    Returns a dict with check results and any issues found.
    
    Usage:
        from tests.viz.test_vscode_compatibility import check_vscode_compatibility
        result = check_vscode_compatibility(my_pipeline)
        if not result["compatible"]:
            print("Issues:", result["issues"])
    """
    widget = PipelineWidget(pipeline)
    decoded = html_module.unescape(widget.value)

    checks = {
        "no_module_scripts": not bool(
            re.search(r'<script[^>]*type=["\']module["\']', decoded)
        ),
        "has_dom_content_loaded": "DOMContentLoaded" in decoded,
        "has_iife": "(function()" in decoded,
        "has_strict_mode": "'use strict'" in decoded,
        "uses_srcdoc": "srcdoc=" in widget.value,
        "has_graph_data": 'id="graph-data"' in decoded,
    }

    issues = []
    for check, passed in checks.items():
        if not passed:
            issues.append(check)

    return {
        "compatible": len(issues) == 0,
        "checks": checks,
        "issues": issues,
    }
