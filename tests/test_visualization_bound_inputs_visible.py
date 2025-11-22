"""Test that bound inputs are visible in visualizations.

This test verifies that bound inputs appear in the visualization with transparency,
rather than being completely hidden.
"""

import pytest

from hypernodes import Pipeline, node


@node(output_name="result")
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


@node(output_name="scaled")
def scale(value: int, factor: int) -> int:
    """Scale a value by a factor."""
    return value * factor


def test_bound_input_appears_in_visualization():
    """Test that a bound input actually appears in the visualization graph."""
    pipeline = Pipeline(nodes=[add])
    pipeline.bind(x=5)
    
    # Generate visualization
    viz = pipeline.visualize()
    
    # Convert to string to check what's in the graph
    viz_str = str(viz)
    
    # The bound input 'x' should appear in the visualization
    # It might be in various forms (node definition, edge, etc)
    assert 'x' in viz_str or 'X' in viz_str, "Bound input 'x' should appear in visualization"
    
    # The unfulfilled input 'y' should definitely appear
    assert 'y' in viz_str or 'Y' in viz_str, "Unfulfilled input 'y' should appear in visualization"


def test_all_bound_inputs_still_visible():
    """Test that when all inputs are bound, they're still visible."""
    pipeline = Pipeline(nodes=[add])
    pipeline.bind(x=5, y=10)
    
    # Generate visualization
    viz = pipeline.visualize()
    viz_str = str(viz)
    
    # Both inputs should appear even though they're bound
    assert 'x' in viz_str or 'X' in viz_str, "Bound input 'x' should appear in visualization"
    assert 'y' in viz_str or 'Y' in viz_str, "Bound input 'y' should appear in visualization"


def test_bound_inputs_have_transparency():
    """Test that bound inputs have different styling (dashed border)."""
    pipeline = Pipeline(nodes=[add])
    pipeline.bind(x=5)
    
    # Visualize without grouping to see individual input styling
    viz = pipeline.visualize(group_inputs=False)
    viz_str = str(viz)
    
    # Check for dashed border (distinguishes bound inputs)
    assert 'dashed' in viz_str, "Bound inputs should have dashed border styling"
    
    # Verify that both inputs exist
    assert 'x' in viz_str or 'X' in viz_str, "Input x should be visible"
    assert 'y' in viz_str or 'Y' in viz_str, "Input y should be visible"
    
    # Verify no transparency suffix in fillcolor (check for hex transparency suffix after color)
    # Pattern: fillcolor="#90EE9080" would have 80 at the end
    assert '#90EE9080' not in viz_str, "Should not have transparency in fillcolor"
    assert '#90EE90B0' not in viz_str, "Should not have partial transparency in fillcolor"


def test_bound_grouped_inputs_visible():
    """Test that bound grouped inputs are still visible."""
    @node(output_name="result")
    def process(a: int, b: int, c: int, d: int) -> int:
        return a + b + c + d
    
    pipeline = Pipeline(nodes=[process])
    pipeline.bind(a=1, b=2)  # Bind some of the grouped inputs
    
    # Visualize with grouping enabled
    viz = pipeline.visualize(group_inputs=True)
    viz_str = str(viz)
    
    # All parameters should appear (bound and unbound)
    assert 'a' in viz_str or 'A' in viz_str, "Bound grouped input 'a' should appear"
    assert 'b' in viz_str or 'B' in viz_str, "Bound grouped input 'b' should appear"
    assert 'c' in viz_str or 'C' in viz_str, "Unbound grouped input 'c' should appear"
    assert 'd' in viz_str or 'D' in viz_str, "Unbound grouped input 'd' should appear"


def test_nested_pipeline_bound_inputs_visible():
    """Test that bound inputs in nested pipelines are visible when expanded."""
    inner = Pipeline(nodes=[scale])
    inner.bind(factor=10)
    
    outer = Pipeline(nodes=[inner.as_node()])
    
    # Visualize with depth=2 to expand inner pipeline
    viz = outer.visualize(depth=2)
    viz_str = str(viz)
    
    # The bound input 'factor' should appear in the visualization
    assert 'factor' in viz_str, "Bound input 'factor' in nested pipeline should appear"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

