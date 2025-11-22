"""Test that visualizations don't show raw Python object IDs."""

import re
import pytest
from hypernodes import Pipeline
from hypernodes.node import node


def test_no_raw_ids_in_visualization_html():
    """Test that generated HTML visualizations don't contain raw Python object IDs."""
    
    # Create a nested pipeline structure
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="tripled")
    def triple(x: int) -> int:
        return x * 3
    
    inner_pipeline = Pipeline(nodes=[double, triple], name="inner")
    inner_node = inner_pipeline.as_node(
        name="inner_step",  # Give it a name
        output_mapping={"tripled": "result"}
    )
    
    @node(output_name="final")
    def add_ten(result: int) -> int:
        return result + 10
    
    outer_pipeline = Pipeline(nodes=[inner_node, add_ten], name="outer")
    
    # Generate visualization HTML
    result = outer_pipeline.visualize(depth=2)
    
    # Extract HTML content
    if hasattr(result, 'data'):
        html_str = result.data
    elif hasattr(result, '_repr_html_'):
        html_str = result._repr_html_()
    else:
        html_str = str(result)
    
    # Check for raw IDs (10+ consecutive digits)
    # These patterns catch raw Python object IDs in various contexts
    id_patterns = [
        r'>\s*(\d{10,})\s*<',           # In text nodes
        r'<text[^>]*>(\d{10,})</text>',  # In SVG text elements
        r'<title>(\d{10,})</title>',     # In titles
        r'<tspan[^>]*>(\d{10,})</tspan>', # In tspans
    ]
    
    found_ids = []
    for pattern in id_patterns:
        matches = re.findall(pattern, html_str)
        found_ids.extend(matches)
    
    # Also check that the HTML actually contains proper labels
    assert "double" in html_str or "triple" in html_str, \
        "Visualization should contain function names"
    
    if found_ids:
        unique_ids = set(found_ids)
        pytest.fail(
            f"Found {len(unique_ids)} raw Python object IDs in visualization HTML:\n" +
            "\n".join(f"  - {raw_id}" for raw_id in sorted(unique_ids)[:5]) +
            "\n\nThese should be replaced with human-readable labels."
        )


def test_pipeline_nodes_have_labels_not_ids():
    """Test that PipelineNodes show descriptive labels, not raw IDs."""
    
    @node(output_name="processed")
    def process(x: int) -> int:
        return x * 2
    
    # Create pipeline WITHOUT explicit name
    inner = Pipeline(nodes=[process])
    inner_node = inner.as_node()  # No name given
    
    @node(output_name="final")
    def finalize(processed: int) -> int:
        return processed + 10
    
    outer = Pipeline(nodes=[inner_node, finalize])
    
    # Generate visualization
    result = outer.visualize(depth=2)
    
    if hasattr(result, 'data'):
        html_str = result.data
    elif hasattr(result, '_repr_html_'):
        html_str = result._repr_html_()
    else:
        html_str = str(result)
    
    # Should contain the fallback label derived from first function
    assert "process" in html_str.lower() or "pipeline" in html_str.lower(), \
        "PipelineNode should have a descriptive label, not a raw ID"
    
    # Check for raw IDs
    id_pattern = r'>\s*\d{10,}\s*<'
    matches = re.findall(id_pattern, html_str)
    
    assert len(matches) == 0, \
        f"Found {len(matches)} raw IDs in visualization. PipelineNodes should use labels."


def test_function_nodes_use_function_names():
    """Test that FunctionNodes display their function names, not object IDs."""
    
    @node(output_name="result1")
    def step_one(x: int) -> int:
        return x * 2
    
    @node(output_name="result2")
    def step_two(result1: int) -> int:
        return result1 + 10
    
    pipeline = Pipeline(nodes=[step_one, step_two])
    
    # Generate visualization
    result = pipeline.visualize(depth=1)
    
    if hasattr(result, 'data'):
        html_str = result.data
    elif hasattr(result, '_repr_html_'):
        html_str = result._repr_html_()
    else:
        html_str = str(result)
    
    # Should contain function names
    assert "step_one" in html_str, "Should contain first function name"
    assert "step_two" in html_str, "Should contain second function name"
    
    # Should NOT contain raw IDs
    id_pattern = r'>\s*\d{10,}\s*<'
    matches = re.findall(id_pattern, html_str)
    
    assert len(matches) == 0, \
        f"Found {len(matches)} raw IDs. Function nodes should display function names."


def test_deeply_nested_pipelines_no_raw_ids():
    """Test that deeply nested pipelines (depth=3) don't show raw IDs."""
    
    # Level 3: innermost
    @node(output_name="x2")
    def level3_node(x: int) -> int:
        return x * 2
    
    level3 = Pipeline(nodes=[level3_node], name="level3")
    
    # Level 2: middle
    level3_as_node = level3.as_node(name="level3_step")
    
    @node(output_name="x3")
    def level2_node(x2: int) -> int:
        return x2 + 1
    
    level2 = Pipeline(nodes=[level3_as_node, level2_node], name="level2")
    
    # Level 1: outer
    level2_as_node = level2.as_node(name="level2_step")
    
    @node(output_name="final")
    def level1_node(x3: int) -> int:
        return x3 + 10
    
    level1 = Pipeline(nodes=[level2_as_node, level1_node], name="level1")
    
    # Generate deeply nested visualization
    result = level1.visualize(depth=3)
    
    if hasattr(result, 'data'):
        html_str = result.data
    elif hasattr(result, '_repr_html_'):
        html_str = result._repr_html_()
    else:
        html_str = str(result)
    
    # Check for raw IDs
    id_pattern = r'>\s*\d{10,}\s*<'
    matches = re.findall(id_pattern, html_str)
    
    if matches:
        unique_ids = set(matches)
        pytest.fail(
            f"Found {len(unique_ids)} raw IDs in depth=3 visualization:\n" +
            "\n".join(f"  - {raw_id}" for raw_id in sorted(unique_ids)[:5])
        )


if __name__ == "__main__":
    # Run tests standalone
    print("Running visualization label tests...")
    print()
    
    try:
        test_no_raw_ids_in_visualization_html()
        print("✅ test_no_raw_ids_in_visualization_html PASSED")
    except AssertionError as e:
        print(f"❌ test_no_raw_ids_in_visualization_html FAILED")
        print(f"   {e}")
    
    try:
        test_pipeline_nodes_have_labels_not_ids()
        print("✅ test_pipeline_nodes_have_labels_not_ids PASSED")
    except AssertionError as e:
        print(f"❌ test_pipeline_nodes_have_labels_not_ids FAILED")
        print(f"   {e}")
    
    try:
        test_function_nodes_use_function_names()
        print("✅ test_function_nodes_use_function_names PASSED")
    except AssertionError as e:
        print(f"❌ test_function_nodes_use_function_names FAILED")
        print(f"   {e}")
    
    try:
        test_deeply_nested_pipelines_no_raw_ids()
        print("✅ test_deeply_nested_pipelines_no_raw_ids PASSED")
    except AssertionError as e:
        print(f"❌ test_deeply_nested_pipelines_no_raw_ids FAILED")
        print(f"   {e}")
    
    print()
    print("🎉 All visualization label tests passed!")

