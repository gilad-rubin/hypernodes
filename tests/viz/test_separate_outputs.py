"""Tests for the separate_outputs parameter in visualization."""

import re
from collections import Counter

import pytest

from hypernodes import Pipeline, node


@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2


@node(output_name="tripled")
def triple(doubled: int) -> int:
    return doubled * 3


@node(output_name="final")
def add_ten(tripled: int) -> int:
    return tripled + 10


def get_svg_titles(result) -> list:
    """Extract <title> elements from SVG output."""
    if hasattr(result, "data"):
        html = str(result.data)
    elif hasattr(result, "_repr_html_"):
        html = str(result._repr_html_())
    else:
        html = str(result)
    
    # Extract titles, filter out edge titles (contain ->)
    return [t for t in re.findall(r"<title>(.*?)</title>", html)
            if "-&gt;" not in t and "cluster_" not in t and t != "G"]


class TestSeparateOutputsFalse:
    """Tests for combined mode (separate_outputs=False, the default)."""

    def test_combined_mode_is_default(self):
        """Test that combined mode is the default."""
        pipeline = Pipeline(nodes=[double, triple])
        
        # Default call
        result = pipeline.visualize()
        titles = get_svg_titles(result)
        
        # Should only have function names and inputs, not output names
        assert "double" in titles
        assert "triple" in titles
        assert "x" in titles
        # Output names should NOT be separate titles in combined mode
        assert "doubled" not in titles
        assert "tripled" not in titles

    def test_combined_mode_explicit_false(self):
        """Test explicit separate_outputs=False."""
        pipeline = Pipeline(nodes=[double, triple])
        
        result = pipeline.visualize(separate_outputs=False)
        titles = get_svg_titles(result)
        
        # Same as default - outputs combined with functions
        assert "double" in titles
        assert "triple" in titles
        assert "doubled" not in titles
        assert "tripled" not in titles

    def test_combined_mode_chain(self):
        """Test combined mode with a longer chain."""
        pipeline = Pipeline(nodes=[double, triple, add_ten])
        
        result = pipeline.visualize(separate_outputs=False)
        titles = get_svg_titles(result)
        
        # All functions present
        assert "double" in titles
        assert "triple" in titles
        assert "add_ten" in titles
        
        # No separate output nodes
        assert "doubled" not in titles
        assert "tripled" not in titles
        assert "final" not in titles


class TestSeparateOutputsTrue:
    """Tests for separate mode (separate_outputs=True)."""

    def test_separate_mode_shows_outputs(self):
        """Test that separate_outputs=True shows output nodes."""
        pipeline = Pipeline(nodes=[double, triple])
        
        result = pipeline.visualize(separate_outputs=True)
        titles = get_svg_titles(result)
        
        # Should have both function names AND output names
        assert "double" in titles
        assert "triple" in titles
        assert "x" in titles
        assert "doubled" in titles
        assert "tripled" in titles

    def test_separate_mode_chain(self):
        """Test separate mode with a longer chain."""
        pipeline = Pipeline(nodes=[double, triple, add_ten])
        
        result = pipeline.visualize(separate_outputs=True)
        titles = get_svg_titles(result)
        
        # All functions
        assert "double" in titles
        assert "triple" in titles
        assert "add_ten" in titles
        
        # All outputs as separate nodes
        assert "doubled" in titles
        assert "tripled" in titles
        assert "final" in titles

    def test_separate_mode_no_duplicates(self):
        """Test that separate mode doesn't create duplicate nodes."""
        pipeline = Pipeline(nodes=[double, triple, add_ten])
        
        result = pipeline.visualize(separate_outputs=True)
        titles = get_svg_titles(result)
        
        counts = Counter(titles)
        duplicates = [(name, count) for name, count in counts.items() if count > 1]
        
        assert len(duplicates) == 0, f"Found duplicates: {duplicates}"


class TestSeparateOutputsNested:
    """Tests for separate_outputs with nested pipelines."""

    def test_nested_pipeline_combined_mode(self):
        """Test combined mode with nested pipeline."""
        inner = Pipeline(nodes=[double, triple], name="inner")
        outer = Pipeline(nodes=[inner.as_node()])
        
        # Combined mode at depth=2 (expanded)
        result = outer.visualize(depth=2, separate_outputs=False)
        titles = get_svg_titles(result)
        
        # Functions should be visible
        assert "double" in titles
        assert "triple" in titles
        
        # Outputs should NOT be separate nodes
        assert "doubled" not in titles
        # Note: In combined mode, outputs are part of the function boxes

    def test_nested_pipeline_separate_mode(self):
        """Test separate mode with nested pipeline."""
        inner = Pipeline(nodes=[double, triple], name="inner")
        outer = Pipeline(nodes=[inner.as_node()])
        
        # Separate mode at depth=2 (expanded)
        result = outer.visualize(depth=2, separate_outputs=True)
        titles = get_svg_titles(result)
        
        # Functions
        assert "double" in titles
        assert "triple" in titles
        
        # Outputs as separate nodes
        assert "doubled" in titles
        assert "tripled" in titles

    def test_nested_with_output_mapping_separate_mode(self):
        """Test separate mode with output mapping."""
        inner = Pipeline(nodes=[double, triple], name="processor")
        inner_node = inner.as_node(output_mapping={"tripled": "result"})
        
        @node(output_name="done")
        def finalize(result: int) -> str:
            return f"Result: {result}"
        
        outer = Pipeline(nodes=[inner_node, finalize])
        
        result = outer.visualize(depth=2, separate_outputs=True)
        titles = get_svg_titles(result)
        
        # Inner outputs should exist
        assert "doubled" in titles
        assert "tripled" in titles
        
        # Mapped output should also exist
        assert "result" in titles
        
        # Final output
        assert "done" in titles


class TestSeparateOutputsEdgeCases:
    """Edge cases for separate_outputs parameter."""

    def test_single_node_combined(self):
        """Test single node pipeline in combined mode."""
        pipeline = Pipeline(nodes=[double])
        
        result = pipeline.visualize(separate_outputs=False)
        titles = get_svg_titles(result)
        
        assert "double" in titles
        assert "x" in titles
        assert "doubled" not in titles

    def test_single_node_separate(self):
        """Test single node pipeline in separate mode."""
        pipeline = Pipeline(nodes=[double])
        
        result = pipeline.visualize(separate_outputs=True)
        titles = get_svg_titles(result)
        
        assert "double" in titles
        assert "x" in titles
        assert "doubled" in titles

    def test_bound_inputs_both_modes(self):
        """Test that bound inputs appear in both modes."""
        pipeline = Pipeline(nodes=[double]).bind(x=5)
        
        # Combined mode
        result_combined = pipeline.visualize(separate_outputs=False)
        titles_combined = get_svg_titles(result_combined)
        assert "x" in titles_combined
        
        # Separate mode
        result_separate = pipeline.visualize(separate_outputs=True)
        titles_separate = get_svg_titles(result_separate)
        assert "x" in titles_separate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

