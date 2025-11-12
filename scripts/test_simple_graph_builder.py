#!/usr/bin/env python3
"""Test SimpleGraphBuilder without NetworkX dependency."""

from hypernodes import node, Pipeline
from hypernodes.graph_builder import SimpleGraphBuilder


@node(output_name="cleaned")
def clean(text: str) -> str:
    return text.strip().lower()


@node(output_name="word_count")
def count(cleaned: str) -> int:
    return len(cleaned.split())


@node(output_name="result")
def format_result(cleaned: str, word_count: int) -> str:
    return f"{cleaned} ({word_count} words)"


def test_simple_builder():
    """Test basic graph building."""
    print("=" * 60)
    print("TEST: Simple Graph Builder")
    print("=" * 60)
    
    nodes = [clean, count, format_result]
    builder = SimpleGraphBuilder()
    result = builder.build_graph(nodes)
    
    print(f"\n✓ Available outputs: {result.available_output_names}")
    print(f"✓ Root args: {result.root_args}")
    print(f"✓ Execution order: {[n.func.__name__ for n in result.execution_order]}")
    print(f"\n✓ Dependencies:")
    for node, deps in result.dependencies.items():
        dep_names = [d.func.__name__ for d in deps]
        print(f"  {node.func.__name__} -> {dep_names if deps else '[]'}")


def test_nested_pipeline():
    """Test with nested pipeline."""
    print("\n" + "=" * 60)
    print("TEST: Nested Pipeline")
    print("=" * 60)
    
    # Inner pipeline
    inner = Pipeline(
        nodes=[clean, count],
        engine=None,  # Just for structure
    )
    
    # Wrap as node
    inner_node = inner.as_node(
        input_mapping={"input_text": "text"},
        output_mapping={"cleaned": "processed", "word_count": "wc"},
    )
    
    # Outer pipeline node
    outer_nodes = [inner_node, format_result]
    builder = SimpleGraphBuilder()
    
    try:
        result = builder.build_graph(outer_nodes)
        print(f"\n✓ Available outputs: {result.available_output_names}")
        print(f"✓ Root args: {result.root_args}")
        print(f"✓ Execution order has {len(result.execution_order)} nodes")
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    test_simple_builder()
    test_nested_pipeline()
