"""Tests for selective output execution feature.

This module tests the ability to specify which outputs to compute,
allowing pipelines to execute only the minimal set of nodes needed.
"""

import pytest

from hypernodes import Pipeline, node
from hypernodes.engine import HypernodesEngine


# Test nodes for building pipelines
@node(output_name="a")
def node_a(x: int) -> int:
    """First node - always needed."""
    return x + 1


@node(output_name="b")
def node_b(a: int) -> int:
    """Second node - depends on a."""
    return a * 2


@node(output_name="c")
def node_c(b: int) -> int:
    """Third node - depends on b."""
    return b + 10


@node(output_name="d")
def node_d(a: int) -> int:
    """Fourth node - depends on a (parallel branch)."""
    return a * 3


@node(output_name="e")
def node_e(c: int, d: int) -> int:
    """Fifth node - depends on c and d (merge point)."""
    return c + d


class TestSelectiveOutputBasic:
    """Test basic selective output functionality."""

    def test_select_single_early_output(self):
        """Test selecting a single output early in the pipeline."""
        # Pipeline: x -> a -> b -> c
        #                 \-> d
        pipeline = Pipeline(nodes=[node_a, node_b, node_c, node_d])

        # Request only 'a' - should execute only node_a
        result = pipeline.run(inputs={"x": 5}, output_name="a")

        assert result == {"a": 6}
        assert "b" not in result
        assert "c" not in result
        assert "d" not in result

    def test_select_single_middle_output(self):
        """Test selecting a single output in the middle of the pipeline."""
        pipeline = Pipeline(nodes=[node_a, node_b, node_c, node_d])

        # Request only 'b' - should execute node_a and node_b
        result = pipeline.run(inputs={"x": 5}, output_name="b")

        assert result == {"b": 12}
        assert "a" not in result
        assert "c" not in result
        assert "d" not in result

    def test_select_single_leaf_output(self):
        """Test selecting a leaf output (requires all dependencies)."""
        pipeline = Pipeline(nodes=[node_a, node_b, node_c, node_d])

        # Request only 'c' - should execute node_a, node_b, node_c
        result = pipeline.run(inputs={"x": 5}, output_name="c")

        assert result == {"c": 22}
        assert "a" not in result
        assert "b" not in result
        assert "d" not in result

    def test_select_multiple_outputs_list(self):
        """Test selecting multiple outputs as a list."""
        pipeline = Pipeline(nodes=[node_a, node_b, node_c, node_d])

        # Request 'a' and 'd' - should execute node_a and node_d
        result = pipeline.run(inputs={"x": 5}, output_name=["a", "d"])

        assert result == {"a": 6, "d": 18}
        assert "b" not in result
        assert "c" not in result

    def test_select_outputs_with_shared_dependencies(self):
        """Test selecting outputs that share dependencies."""
        pipeline = Pipeline(nodes=[node_a, node_b, node_c, node_d])

        # Request 'b' and 'd' - both depend on 'a'
        result = pipeline.run(inputs={"x": 5}, output_name=["b", "d"])

        assert result == {"b": 12, "d": 18}
        assert "a" not in result
        assert "c" not in result

    def test_select_none_returns_all_outputs(self):
        """Test that output_name=None returns all outputs (default behavior)."""
        pipeline = Pipeline(nodes=[node_a, node_b, node_c, node_d])

        result = pipeline.run(inputs={"x": 5}, output_name=None)

        assert result == {"a": 6, "b": 12, "c": 22, "d": 18}

    def test_select_all_outputs_explicitly(self):
        """Test explicitly requesting all outputs."""
        pipeline = Pipeline(nodes=[node_a, node_b, node_c, node_d])

        result = pipeline.run(inputs={"x": 5}, output_name=["a", "b", "c", "d"])

        assert result == {"a": 6, "b": 12, "c": 22, "d": 18}

    def test_invalid_output_name_raises_error(self):
        """Test that requesting a non-existent output raises an error."""
        pipeline = Pipeline(nodes=[node_a, node_b])

        with pytest.raises(ValueError, match="not found in pipeline"):
            pipeline.run(inputs={"x": 5}, output_name="nonexistent")

    def test_invalid_output_name_in_list_raises_error(self):
        """Test that requesting a non-existent output in a list raises an error."""
        pipeline = Pipeline(nodes=[node_a, node_b])

        with pytest.raises(ValueError, match="not found in pipeline"):
            pipeline.run(inputs={"x": 5}, output_name=["a", "nonexistent"])


class TestSelectiveOutputMap:
    """Test selective output with map operations."""

    def test_map_with_single_output(self):
        """Test map operation with single output selection."""
        pipeline = Pipeline(nodes=[node_a, node_b, node_c])

        # Map over x, but only request 'b'
        result = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x", output_name="b")

        assert result == {"b": [4, 6, 8]}
        assert "a" not in result
        assert "c" not in result

    def test_map_with_multiple_outputs(self):
        """Test map operation with multiple output selection."""
        pipeline = Pipeline(nodes=[node_a, node_b, node_c])

        # Map over x, request 'a' and 'c'
        result = pipeline.map(
            inputs={"x": [1, 2, 3]}, map_over="x", output_name=["a", "c"]
        )

        assert result == {"a": [2, 3, 4], "c": [14, 16, 18]}
        assert "b" not in result

    def test_map_with_none_returns_all(self):
        """Test map with output_name=None returns all outputs."""
        pipeline = Pipeline(nodes=[node_a, node_b])

        result = pipeline.map(inputs={"x": [1, 2]}, map_over="x", output_name=None)

        assert result == {"a": [2, 3], "b": [4, 6]}


class TestSelectiveOutputComplexDependencies:
    """Test selective output with complex dependency graphs."""

    def test_diamond_dependency_select_merge_point(self):
        """Test selecting output at merge point of diamond dependency."""
        # Pipeline: x -> a -> b -> c
        #                 \-> d /
        #                      e
        pipeline = Pipeline(nodes=[node_a, node_b, node_c, node_d, node_e])

        # Request only 'e' - should execute a, b, c, d, e
        result = pipeline.run(inputs={"x": 5}, output_name="e")

        assert result == {"e": 40}  # c=22, d=18, e=22+18=40
        assert len(result) == 1

    def test_diamond_dependency_select_one_branch(self):
        """Test selecting output from one branch of diamond."""
        pipeline = Pipeline(nodes=[node_a, node_b, node_c, node_d, node_e])

        # Request only 'c' - should execute a, b, c (not d or e)
        result = pipeline.run(inputs={"x": 5}, output_name="c")

        assert result == {"c": 22}
        assert "d" not in result
        assert "e" not in result


class TestSelectiveOutputExecutionModes:
    """Test selective output with different execution modes."""

    @pytest.mark.parametrize("node_execution", ["sequential", "async", "threaded"])
    def test_different_execution_modes(self, node_execution):
        """Test selective output works with different execution modes."""
        engine = HypernodesEngine(node_executor=node_execution)
        pipeline = Pipeline(nodes=[node_a, node_b, node_c], backend=engine)

        result = pipeline.run(inputs={"x": 5}, output_name="b")

        assert result == {"b": 12}
        assert "c" not in result


class TestSelectiveOutputCaching:
    """Test that selective output works correctly with caching."""

    def test_selective_output_with_cache(self):
        """Test that caching works correctly with selective outputs."""
        import shutil
        import tempfile

        from hypernodes import DiskCache

        # Create temporary cache directory
        cache_dir = tempfile.mkdtemp()

        try:
            cache = DiskCache(path=cache_dir)
            pipeline = Pipeline(nodes=[node_a, node_b, node_c], cache=cache)

            # First run - compute only 'b'
            result1 = pipeline.run(inputs={"x": 5}, output_name="b")
            assert result1 == {"b": 12}

            # Second run - should use cache for 'b'
            result2 = pipeline.run(inputs={"x": 5}, output_name="b")
            assert result2 == {"b": 12}

            # Third run - request different output 'c'
            result3 = pipeline.run(inputs={"x": 5}, output_name="c")
            assert result3 == {"c": 22}

        finally:
            shutil.rmtree(cache_dir)


class TestSelectiveOutputNestedPipelines:
    """Test selective output with nested pipelines."""

    def test_nested_pipeline_selective_output(self):
        """Test selective output works with nested pipelines."""
        # Inner pipeline
        inner = Pipeline(nodes=[node_a, node_b])

        # Outer pipeline uses inner as a node
        @node(output_name="final")
        def final_node(b: int) -> int:
            return b + 100

        outer = Pipeline(nodes=[inner.as_node(), final_node])

        # Request only 'b' from outer pipeline
        result = outer.run(inputs={"x": 5}, output_name="b")

        assert result == {"b": 12}
        assert "final" not in result


# Note: ModalBackend tests would require Modal setup
# These are placeholder tests that can be run when Modal is configured
class TestSelectiveOutputModalBackend:
    """Test selective output with ModalBackend."""

    @pytest.mark.skip(reason="Requires Modal configuration")
    def test_modal_backend_selective_output(self):
        """Test that selective output works with ModalBackend."""
        # This test would require Modal setup
        # Keeping as placeholder for future testing
        pass

    @pytest.mark.skip(reason="Requires Modal configuration")
    def test_modal_backend_map_selective_output(self):
        """Test that selective output works with ModalBackend.map()."""
        # This test would require Modal setup
        # Keeping as placeholder for future testing
        pass
