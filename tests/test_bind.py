"""Tests for Pipeline.bind() and .unbind() functionality."""

import pytest
from hypernodes import Pipeline, node


@node(output_name="result")
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


@node(output_name="doubled")
def double(result: int) -> int:
    """Double a number."""
    return result * 2


@node(output_name="sum")
def sum_list(numbers: list) -> int:
    """Sum a list of numbers."""
    return sum(numbers)


@node(output_name="scaled")
def scale(value: int, factor: int) -> int:
    """Scale a value by a factor."""
    return value * factor


class TestBind:
    """Test bind() functionality."""

    def test_basic_bind_and_run(self):
        """Test basic bind and run without passing inputs."""
        pipeline = Pipeline(nodes=[add])
        pipeline.bind(x=5, y=10)
        result = pipeline.run()
        assert result == {"result": 15}

    def test_bind_then_override(self):
        """Test that provided inputs override bound inputs."""
        pipeline = Pipeline(nodes=[add])
        pipeline.bind(x=5, y=10)
        result = pipeline.run(inputs={"x": 100})
        assert result == {"result": 110}  # x=100 (override), y=10 (bound)

    def test_multiple_bind_calls_merge(self):
        """Test that multiple bind() calls merge their inputs."""
        pipeline = Pipeline(nodes=[add])
        pipeline.bind(x=5)
        pipeline.bind(y=10)
        result = pipeline.run()
        assert result == {"result": 15}

    def test_multiple_bind_calls_override_same_key(self):
        """Test that later bind() calls override earlier values for the same key."""
        pipeline = Pipeline(nodes=[add])
        pipeline.bind(x=5, y=10)
        pipeline.bind(x=20)  # Override x
        result = pipeline.run()
        assert result == {"result": 30}  # x=20, y=10

    def test_bind_with_chaining(self):
        """Test that bind() supports method chaining."""
        pipeline = Pipeline(nodes=[add]).bind(x=5, y=10)
        result = pipeline.run()
        assert result == {"result": 15}

    def test_bind_partial_inputs(self):
        """Test binding only some inputs and providing others at runtime."""
        pipeline = Pipeline(nodes=[add])
        pipeline.bind(x=5)
        result = pipeline.run(inputs={"y": 10})
        assert result == {"result": 15}

    def test_run_without_bind_still_works(self):
        """Test that run() still works normally without bind()."""
        pipeline = Pipeline(nodes=[add])
        result = pipeline.run(inputs={"x": 5, "y": 10})
        assert result == {"result": 15}

    def test_bind_validates_at_runtime(self):
        """Test that validation happens at run() time, not bind() time."""
        pipeline = Pipeline(nodes=[add])
        # This should not raise - validation is lazy
        pipeline.bind(invalid_key=999)
        # But this should raise when we try to run
        with pytest.raises(ValueError, match="Missing required input"):
            pipeline.run()

    def test_bind_missing_required_input_raises(self):
        """Test that missing required inputs still raise errors."""
        pipeline = Pipeline(nodes=[add])
        pipeline.bind(x=5)  # Only bind x, not y
        with pytest.raises(ValueError, match="Missing required input.*y"):
            pipeline.run()  # Should raise because y is missing

    def test_bind_with_pipeline_chain(self):
        """Test binding with a multi-node pipeline."""
        pipeline = Pipeline(nodes=[add, double])
        pipeline.bind(x=5, y=10)
        result = pipeline.run()
        assert result == {"result": 15, "doubled": 30}


class TestUnbind:
    """Test unbind() functionality."""

    def test_unbind_specific_key(self):
        """Test unbinding a specific key."""
        pipeline = Pipeline(nodes=[add])
        pipeline.bind(x=5, y=10)
        pipeline.unbind("x")
        result = pipeline.run(inputs={"x": 20})
        assert result == {"result": 30}  # x=20 (provided), y=10 (still bound)

    def test_unbind_multiple_keys(self):
        """Test unbinding multiple specific keys."""
        pipeline = Pipeline(nodes=[add])
        pipeline.bind(x=5, y=10)
        pipeline.unbind("x", "y")
        result = pipeline.run(inputs={"x": 1, "y": 2})
        assert result == {"result": 3}

    def test_unbind_all(self):
        """Test unbinding all bound inputs."""
        pipeline = Pipeline(nodes=[add])
        pipeline.bind(x=5, y=10)
        pipeline.unbind()  # No arguments = clear all
        result = pipeline.run(inputs={"x": 1, "y": 2})
        assert result == {"result": 3}

    def test_unbind_nonexistent_key(self):
        """Test that unbinding a non-existent key doesn't raise."""
        pipeline = Pipeline(nodes=[add])
        pipeline.bind(x=5, y=10)
        pipeline.unbind("nonexistent")  # Should not raise
        result = pipeline.run()
        assert result == {"result": 15}  # Original bindings unchanged

    def test_unbind_without_bind(self):
        """Test that unbind() doesn't raise if bind() was never called."""
        pipeline = Pipeline(nodes=[add])
        pipeline.unbind()  # Should not raise
        result = pipeline.run(inputs={"x": 5, "y": 10})
        assert result == {"result": 15}

    def test_unbind_with_chaining(self):
        """Test that unbind() supports method chaining."""
        pipeline = Pipeline(nodes=[add]).bind(x=5, y=10).unbind("x")
        result = pipeline.run(inputs={"x": 20})
        assert result == {"result": 30}


class TestBindWithMap:
    """Test bind() with map operations."""

    def test_bind_with_map(self):
        """Test that bind works with map operations."""
        pipeline = Pipeline(nodes=[scale])
        pipeline.bind(factor=10)
        results = pipeline.map(inputs={"value": [1, 2, 3]}, map_over="value")
        assert results == [{"scaled": 10}, {"scaled": 20}, {"scaled": 30}]

    def test_bind_map_override(self):
        """Test that provided inputs override bound inputs in map."""
        pipeline = Pipeline(nodes=[scale])
        pipeline.bind(factor=10)
        results = pipeline.map(
            inputs={"value": [1, 2], "factor": [100, 200]}, 
            map_over=["value", "factor"]
        )
        assert results == [{"scaled": 100}, {"scaled": 400}]

    def test_bind_with_map_zip(self):
        """Test bind with zip mode map."""
        pipeline = Pipeline(nodes=[add])
        pipeline.bind(y=1000)
        results = pipeline.map(
            inputs={"x": [1, 2, 3]}, 
            map_over="x",
            map_mode="zip"
        )
        assert results == [
            {"result": 1001},
            {"result": 1002},
            {"result": 1003}
        ]

    def test_bind_with_map_product(self):
        """Test bind with product mode map."""
        pipeline = Pipeline(nodes=[add])
        pipeline.bind(y=100)
        results = pipeline.map(
            inputs={"x": [1, 2]}, 
            map_over="x",
            map_mode="product"
        )
        assert results == [{"result": 101}, {"result": 102}]

    def test_map_without_inputs_uses_only_bound(self):
        """Test that map can work with only bound inputs if map_over is specified."""
        pipeline = Pipeline(nodes=[scale])
        pipeline.bind(value=[1, 2, 3], factor=10)
        results = pipeline.map(map_over="value")
        assert results == [{"scaled": 10}, {"scaled": 20}, {"scaled": 30}]


class TestBoundInputsProperty:
    """Test bound_inputs property."""

    def test_bound_inputs_empty_initially(self):
        """Test that bound_inputs returns empty dict before bind()."""
        pipeline = Pipeline(nodes=[add])
        assert pipeline.bound_inputs == {}

    def test_bound_inputs_after_bind(self):
        """Test that bound_inputs returns bound values."""
        pipeline = Pipeline(nodes=[add])
        pipeline.bind(x=5, y=10)
        assert pipeline.bound_inputs == {"x": 5, "y": 10}

    def test_bound_inputs_is_copy(self):
        """Test that modifying returned dict doesn't affect internal state."""
        pipeline = Pipeline(nodes=[add])
        pipeline.bind(x=5, y=10)
        bound = pipeline.bound_inputs
        bound["x"] = 999  # Try to modify
        assert pipeline.bound_inputs == {"x": 5, "y": 10}  # Unchanged


class TestUnfulfilledArgsProperty:
    """Test unfulfilled_args property."""

    def test_unfulfilled_args_all_initially(self):
        """Test that unfulfilled_args equals root_args initially."""
        pipeline = Pipeline(nodes=[add])
        assert pipeline.unfulfilled_args == tuple(pipeline.graph.root_args)
        assert set(pipeline.unfulfilled_args) == {"x", "y"}

    def test_unfulfilled_args_after_partial_bind(self):
        """Test that only unbound params are returned."""
        pipeline = Pipeline(nodes=[add])
        pipeline.bind(x=5)
        assert pipeline.unfulfilled_args == ("y",)

    def test_unfulfilled_args_after_full_bind(self):
        """Test that empty tuple returned when all bound."""
        pipeline = Pipeline(nodes=[add])
        pipeline.bind(x=5, y=10)
        assert pipeline.unfulfilled_args == ()

    def test_unfulfilled_args_after_unbind(self):
        """Test that params return to unfulfilled after unbind."""
        pipeline = Pipeline(nodes=[add])
        pipeline.bind(x=5, y=10)
        assert pipeline.unfulfilled_args == ()
        
        pipeline.unbind("x")
        assert pipeline.unfulfilled_args == ("x",)
        
        pipeline.unbind()
        assert set(pipeline.unfulfilled_args) == {"x", "y"}


class TestPipelineNodeBoundProperties:
    """Test bound_inputs and unfulfilled_args on PipelineNode."""

    def test_pipeline_node_bound_inputs(self):
        """Test that PipelineNode exposes inner pipeline's bound_inputs."""
        inner = Pipeline(nodes=[add]).bind(x=5, y=10)
        inner_node = inner.as_node()
        assert inner_node.bound_inputs == {"x": 5, "y": 10}

    def test_pipeline_node_unfulfilled_args_empty_when_fully_bound(self):
        """Test that unfulfilled_args is empty when inner is fully bound."""
        inner = Pipeline(nodes=[add]).bind(x=5, y=10)
        inner_node = inner.as_node()
        assert inner_node.unfulfilled_args == ()

    def test_pipeline_node_unfulfilled_args_with_input_mapping(self):
        """Test that unfulfilled_args applies reverse mapping."""
        inner = Pipeline(nodes=[add]).bind(x=5)
        inner_node = inner.as_node(
            input_mapping={"outer_y": "y"}
        )
        assert inner_node.unfulfilled_args == ("outer_y",)

    def test_pipeline_node_unfulfilled_args_with_map_over(self):
        """Test that map_over params are included in unfulfilled_args."""
        inner = Pipeline(nodes=[scale]).bind(factor=10)
        inner_node = inner.as_node(map_over="values", input_mapping={"values": "value"})
        assert inner_node.unfulfilled_args == ("values",)


class TestValidationWithBoundInputs:
    """Test that validation uses unfulfilled_args."""

    def test_validation_uses_unfulfilled_args(self):
        """Test that run() validates only unfulfilled params."""
        pipeline = Pipeline(nodes=[add])
        pipeline.bind(x=5)
        
        # Should only require y, not x
        result = pipeline.run(inputs={"y": 10})
        assert result == {"result": 15}

    def test_nested_pipeline_validation_respects_inner_bindings(self):
        """Test that outer pipeline doesn't ask for bound inner params."""
        inner = Pipeline(nodes=[add]).bind(x=5, y=10)
        inner_node = inner.as_node()
        outer = Pipeline(nodes=[inner_node])
        
        # Outer should not require x or y since inner has them bound
        result = outer.run()
        assert result == {"result": 15}

    def test_nested_pipeline_partial_binding(self):
        """Test nested pipeline with partial binding."""
        inner = Pipeline(nodes=[add]).bind(x=5)
        inner_node = inner.as_node()
        outer = Pipeline(nodes=[inner_node])
        
        # Outer should only require y
        result = outer.run(inputs={"y": 10})
        assert result == {"result": 15}


class TestReprWithBoundInputs:
    """Test __repr__ includes bound input information."""

    def test_repr_shows_bound_inputs(self):
        """Test that __repr__ includes bound input info."""
        pipeline = Pipeline(nodes=[add]).bind(x=5, y=10)
        repr_str = repr(pipeline)
        assert "bound=" in repr_str
        # Should show bound inputs

    def test_repr_shows_unfulfilled_args(self):
        """Test that __repr__ shows what's still needed."""
        pipeline = Pipeline(nodes=[add]).bind(x=5)
        repr_str = repr(pipeline)
        assert "needs=" in repr_str
        assert "y" in repr_str


class TestBindEdgeCases:
    """Test edge cases and integration scenarios."""

    def test_bind_empty_dict(self):
        """Test binding with no inputs (should work but not change anything)."""
        pipeline = Pipeline(nodes=[add])
        pipeline.bind()
        result = pipeline.run(inputs={"x": 5, "y": 10})
        assert result == {"result": 15}

    def test_bind_with_output_name_selection(self):
        """Test that bind works with selective output_name."""
        pipeline = Pipeline(nodes=[add, double])
        pipeline.bind(x=5, y=10)
        result = pipeline.run(output_name="result")
        assert result == {"result": 15}
        assert "doubled" not in result

    def test_bind_survives_multiple_runs(self):
        """Test that bound inputs persist across multiple runs."""
        pipeline = Pipeline(nodes=[add])
        pipeline.bind(x=5, y=10)
        
        result1 = pipeline.run()
        result2 = pipeline.run()
        result3 = pipeline.run(inputs={"x": 100})
        
        assert result1 == {"result": 15}
        assert result2 == {"result": 15}
        assert result3 == {"result": 110}

    def test_bind_with_none_value(self):
        """Test binding None as a value (should work)."""
        @node(output_name="result")
        def handle_none(x: int, y=None) -> int:
            return x if y is None else x + y
        
        pipeline = Pipeline(nodes=[handle_none])
        pipeline.bind(y=None)
        result = pipeline.run(inputs={"x": 5})
        assert result == {"result": 5}

    def test_bind_and_unbind_interleaved(self):
        """Test complex bind/unbind patterns."""
        pipeline = Pipeline(nodes=[add])
        
        pipeline.bind(x=1, y=2)
        assert pipeline.run() == {"result": 3}
        
        pipeline.bind(x=10)  # Override x
        assert pipeline.run() == {"result": 12}
        
        pipeline.unbind("x")  # Remove x binding
        result = pipeline.run(inputs={"x": 5})
        assert result == {"result": 7}  # x=5 (provided), y=2 (bound)
        
        pipeline.unbind()  # Clear all
        result = pipeline.run(inputs={"x": 100, "y": 200})
        assert result == {"result": 300}

