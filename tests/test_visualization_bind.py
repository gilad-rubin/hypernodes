"""Test visualization with bound inputs.

Tests for the visualization feature when using .bind() to fulfill inputs.
"""

import pytest

from hypernodes import Pipeline, node


@node(output_name="result")
def simple_add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


@node(output_name="scaled")
def scale(value: int, factor: int) -> int:
    """Scale a value by a factor."""
    return value * factor


@node(output_name="processed")
def process_with_config(data: str, config: dict, threshold: float) -> str:
    """Process data with multiple parameters."""
    return f"{data}_{config}_{threshold}"


@node(output_name="output")
def combine(result: int, scaled: int) -> int:
    """Combine two results."""
    return result + scaled


class TestVisualizationWithBind:
    """Test that visualization works correctly with bound inputs."""

    def test_visualize_simple_pipeline_no_binding(self):
        """Test basic visualization without any binding."""
        pipeline = Pipeline(nodes=[simple_add])
        
        # Should not raise any errors
        viz = pipeline.visualize()
        assert viz is not None

    def test_visualize_pipeline_with_partial_binding(self):
        """Test visualization when some inputs are bound."""
        pipeline = Pipeline(nodes=[simple_add])
        pipeline.bind(x=5)
        
        # Should visualize successfully
        viz = pipeline.visualize()
        assert viz is not None

    def test_visualize_pipeline_with_full_binding(self):
        """Test visualization when all inputs are bound."""
        pipeline = Pipeline(nodes=[simple_add])
        pipeline.bind(x=5, y=10)
        
        # Should visualize successfully
        viz = pipeline.visualize()
        assert viz is not None

    def test_visualize_multi_node_pipeline_with_binding(self):
        """Test visualization of multi-node pipeline with bound inputs."""
        pipeline = Pipeline(nodes=[simple_add, scale, combine])
        pipeline.bind(y=10, factor=2)
        
        # Should visualize successfully
        viz = pipeline.visualize()
        assert viz is not None

    def test_visualize_with_grouped_inputs_and_binding(self):
        """Test visualization with grouped inputs when some are bound."""
        pipeline = Pipeline(nodes=[process_with_config])
        pipeline.bind(threshold=0.5)
        
        # Should visualize successfully with grouped inputs
        # group_inputs=True means config and threshold might be grouped
        viz = pipeline.visualize(group_inputs=True)
        assert viz is not None

    def test_visualize_all_grouped_inputs_bound(self):
        """Test when all grouped inputs are bound."""
        pipeline = Pipeline(nodes=[process_with_config])
        pipeline.bind(config={"key": "value"}, threshold=0.5)
        
        # Should visualize successfully
        viz = pipeline.visualize(group_inputs=True)
        assert viz is not None

    def test_visualize_nested_pipeline_with_binding(self):
        """Test visualization of nested pipeline where inner has bound inputs."""
        inner = Pipeline(nodes=[scale])
        inner.bind(factor=10)
        
        outer = Pipeline(nodes=[inner.as_node()])
        
        # Should visualize successfully
        viz = outer.visualize()
        assert viz is not None

    def test_visualize_nested_expanded_with_binding(self):
        """Test visualization with depth=2 when nested pipeline has bindings."""
        inner = Pipeline(nodes=[scale])
        inner.bind(factor=10)
        
        outer = Pipeline(nodes=[inner.as_node()])
        
        # Should visualize successfully when expanded
        viz = outer.visualize(depth=2)
        assert viz is not None

    def test_visualize_after_bind_unbind_cycle(self):
        """Test visualization after binding and unbinding."""
        pipeline = Pipeline(nodes=[simple_add])
        pipeline.bind(x=5, y=10)
        pipeline.unbind("x")
        
        # Should visualize successfully
        viz = pipeline.visualize()
        assert viz is not None

    def test_visualize_different_orientations_with_binding(self):
        """Test visualization with different orientations when inputs are bound."""
        pipeline = Pipeline(nodes=[simple_add])
        pipeline.bind(x=5)
        
        for orient in ["TB", "LR", "BT", "RL"]:
            viz = pipeline.visualize(orient=orient)
            assert viz is not None

    def test_visualize_with_binding_and_styles(self):
        """Test visualization with bound inputs using different styles."""
        pipeline = Pipeline(nodes=[simple_add])
        pipeline.bind(x=5)
        
        # Test with different styles
        for style_name in ["default", "minimal", "dark"]:
            viz = pipeline.visualize(style=style_name)
            assert viz is not None


class TestNestedPipelineVisualization:
    """Test visualization of nested pipelines with input binding."""

    def test_nested_pipeline_fully_bound(self):
        """Test nested pipeline that is fully self-contained via binding."""
        inner = Pipeline(nodes=[simple_add])
        inner.bind(x=5, y=10)
        
        # Inner is fully bound, outer should accept no inputs for inner_node
        outer = Pipeline(nodes=[inner.as_node()])
        
        # Should visualize without errors
        viz = outer.visualize()
        assert viz is not None

    def test_nested_with_input_mapping_and_binding(self):
        """Test nested pipeline with input mapping when some inputs are bound."""
        inner = Pipeline(nodes=[scale])
        inner.bind(factor=10)
        
        # Map outer's "val" to inner's "value"
        inner_node = inner.as_node(input_mapping={"val": "value"})
        outer = Pipeline(nodes=[inner_node])
        
        # Should visualize correctly
        viz = outer.visualize()
        assert viz is not None

    def test_deeply_nested_with_bindings(self):
        """Test deeply nested pipelines with bindings at multiple levels."""
        # Level 1: innermost
        level1 = Pipeline(nodes=[scale])
        level1.bind(factor=2)
        
        # Level 2: middle
        level2 = Pipeline(nodes=[level1.as_node()])
        level2.bind(value=100)
        
        # Level 3: outer
        level3 = Pipeline(nodes=[level2.as_node()])
        
        # Should visualize correctly
        viz = level3.visualize(depth=None)  # Fully expand
        assert viz is not None


class TestVisualizationInputRecognition:
    """Test that visualization correctly recognizes bound vs unbound inputs."""

    def test_bound_inputs_reflected_in_visualization(self):
        """Test that bound inputs are visually distinguished."""
        pipeline = Pipeline(nodes=[simple_add])
        pipeline.bind(x=5)
        
        # Generate visualization
        viz = pipeline.visualize()
        
        # The visualization should complete without errors
        # Bound inputs should have different styling (tested visually)
        assert viz is not None
        
        # Check that pipeline correctly reports bound/unbound status
        assert pipeline.bound_inputs == {"x": 5}
        assert pipeline.unfulfilled_args == ("y",)

    def test_no_inputs_needed_visualization(self):
        """Test visualization when all inputs are bound (no external inputs needed)."""
        pipeline = Pipeline(nodes=[simple_add])
        pipeline.bind(x=5, y=10)
        
        # No external inputs needed
        assert pipeline.unfulfilled_args == ()
        
        # Should still visualize correctly
        viz = pipeline.visualize()
        assert viz is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

