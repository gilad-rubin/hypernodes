"""
Visualization tests adapted from scripts.

Validates that nested pipelines render with different detail based on depth.
Skips gracefully if graphviz is not installed in the environment.
"""

import pytest

from hypernodes import Pipeline, node


@pytest.mark.skipif(pytest.importorskip("graphviz", reason="graphviz not installed") is None, reason="graphviz not installed")
def test_visualization_depth_graph_has_more_nodes_when_expanded():
    @node(output_name="cleaned")
    def clean_text(text: str) -> str:
        return text.strip()

    @node(output_name="uppercased")
    def uppercase_text(cleaned: str) -> str:
        return cleaned.upper()

    inner_pipeline = Pipeline(nodes=[clean_text, uppercase_text]).with_name("preprocessing")

    @node(output_name="result")
    def add_prefix(uppercased: str, prefix: str) -> str:
        return f"{prefix}: {uppercased}"

    outer_pipeline = Pipeline(nodes=[inner_pipeline, add_prefix]).with_name("main_pipeline")

    # Depth=1: collapsed view, Depth=2: expanded inner pipeline
    g1 = outer_pipeline.visualize(depth=1, return_type="graphviz")
    g2 = outer_pipeline.visualize(depth=2, return_type="graphviz")

    # Heuristic: expanded graph has more DOT body lines
    assert hasattr(g1, "body") and hasattr(g2, "body")
    assert len(g2.body) > len(g1.body)

