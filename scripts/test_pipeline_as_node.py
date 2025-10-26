"""Test script for pipeline.with_name() and direct pipeline nesting."""

from hypernodes import node, Pipeline


@node(output_name="cleaned")
def clean_text(text: str) -> str:
    """Clean text by stripping whitespace."""
    return text.strip()


@node(output_name="uppercased")
def uppercase_text(cleaned: str) -> str:
    """Convert text to uppercase."""
    return cleaned.upper()


@node(output_name="result")
def add_prefix(uppercased: str, prefix: str) -> str:
    """Add prefix to text."""
    return f"{prefix}: {uppercased}"


def test_with_name():
    """Test that with_name() works."""
    print("\n=== Test 1: with_name() ===")
    
    inner_pipeline = Pipeline(
        nodes=[clean_text, uppercase_text]
    ).with_name("preprocessing")
    
    assert inner_pipeline.name == "preprocessing"
    print(f"✓ Pipeline name set: {inner_pipeline.name}")
    print(f"✓ Pipeline repr: {inner_pipeline}")


def test_direct_nesting():
    """Test that Pipeline can be used directly as a node."""
    print("\n=== Test 2: Direct Pipeline Nesting ===")
    
    inner_pipeline = Pipeline(
        nodes=[clean_text, uppercase_text]
    ).with_name("preprocessing")
    
    # Use pipeline directly without calling .as_node()
    outer_pipeline = Pipeline(
        nodes=[inner_pipeline, add_prefix]
    )
    
    # Run the pipeline
    result = outer_pipeline.run(inputs={"text": "  hello world  ", "prefix": "OUTPUT"})
    
    print(f"✓ Direct nesting works")
    print(f"  Input: '  hello world  '")
    print(f"  Result: {result}")
    assert result["result"] == "OUTPUT: HELLO WORLD"
    print(f"✓ Result correct: {result['result']}")


def test_visualization_depth():
    """Test that visualization with depth=2 shows inner pipeline contents."""
    print("\n=== Test 3: Visualization with depth=2 ===")
    
    inner_pipeline = Pipeline(
        nodes=[clean_text, uppercase_text]
    ).with_name("preprocessing")
    
    outer_pipeline = Pipeline(
        nodes=[inner_pipeline, add_prefix]
    ).with_name("main_pipeline")
    
    # Visualize with depth=1 (collapsed)
    print("✓ Creating visualization with depth=1 (collapsed)...")
    viz1 = outer_pipeline.visualize(depth=1, return_type="graphviz")
    print(f"  Nodes in graph: {len(viz1.body)} lines")
    
    # Visualize with depth=2 (expanded)
    print("✓ Creating visualization with depth=2 (expanded)...")
    viz2 = outer_pipeline.visualize(depth=2, return_type="graphviz")
    print(f"  Nodes in graph: {len(viz2.body)} lines")
    
    # The expanded version should have more nodes
    assert len(viz2.body) > len(viz1.body), "Expanded visualization should have more nodes"
    print("✓ Expanded visualization has more nodes than collapsed")


def test_equivalence():
    """Test that direct nesting is equivalent to using as_node()."""
    print("\n=== Test 4: Equivalence Test ===")
    
    inner_pipeline = Pipeline(
        nodes=[clean_text, uppercase_text]
    ).with_name("preprocessing")
    
    # Method 1: Direct nesting
    outer1 = Pipeline(
        nodes=[inner_pipeline, add_prefix]
    )
    
    # Method 2: Using as_node()
    node_pipe = inner_pipeline.as_node()
    outer2 = Pipeline(
        nodes=[node_pipe, add_prefix]
    )
    
    # Both should produce the same result
    inputs = {"text": "  test  ", "prefix": "RESULT"}
    result1 = outer1.run(inputs=inputs)
    result2 = outer2.run(inputs=inputs)
    
    print(f"✓ Direct nesting result: {result1}")
    print(f"✓ as_node() result: {result2}")
    assert result1 == result2, "Both methods should produce the same result"
    print("✓ Both methods produce identical results")


if __name__ == "__main__":
    test_with_name()
    test_direct_nesting()
    test_visualization_depth()
    test_equivalence()
    
    print("\n" + "="*50)
    print("✓ All tests passed!")
    print("="*50)
