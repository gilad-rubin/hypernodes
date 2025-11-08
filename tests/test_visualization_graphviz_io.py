"""
Visualization basic checks: object type and file output.

Adapted from scripts visualization snippets. Keeps assertions light but useful.
"""

import pytest

from hypernodes import Pipeline, node


graphviz = pytest.importorskip("graphviz", reason="graphviz not installed")


def test_visualize_returns_graphviz_and_writes_file(tmp_path):
    @node(output_name="a")
    def a_fn(x: int) -> int:
        return x + 1

    @node(output_name="b")
    def b_fn(a: int) -> int:
        return a * 2

    p = Pipeline(nodes=[a_fn, b_fn]).with_name("simple")

    g = p.visualize(return_type="graphviz")
    assert isinstance(g, graphviz.Digraph)

    out_file = tmp_path / "dag.svg"
    p.visualize(filename=str(out_file), return_type="graphviz")
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_visualize_nested_pipeline_as_node_no_error():
    @node(output_name="cleaned")
    def clean_text(text: str) -> str:
        return text.strip().lower()

    @node(output_name="upper")
    def to_upper(cleaned: str) -> str:
        return cleaned.upper()

    inner = Pipeline(nodes=[clean_text, to_upper]).with_name("inner")
    inner_node = inner.as_node(name="inner_node")

    @node(output_name="final")
    def add_prefix(upper: str, prefix: str) -> str:
        return f"{prefix}:{upper}"

    outer = Pipeline(nodes=[inner_node, add_prefix]).with_name("outer")

    # Ensure both collapsed and expanded renders work without exceptions
    g1 = outer.visualize(depth=1, return_type="graphviz")
    g2 = outer.visualize(depth=2, return_type="graphviz")
    assert isinstance(g1, graphviz.Digraph)
    assert isinstance(g2, graphviz.Digraph)
    # Expanded graph likely larger
    assert len(g2.body) >= len(g1.body)

