"""Test selective input validation based on output_name."""

import pytest

from hypernodes import Pipeline, node


def test_selective_inputs_with_output_name():
    """When output_name is specified, only validate inputs needed for that output."""

    @node(output_name="a_out")
    def a(x: int) -> int:
        return x + 1

    @node(output_name="b_out")
    def b(a_out: int, y: int) -> int:
        return a_out + y

    pipeline = Pipeline(nodes=[a, b])

    # This should work - we only need 'x' to compute 'a_out'
    result = pipeline.run(inputs={"x": 1}, output_name="a_out")
    assert result == {"a_out": 2}

    # This should fail - we need both 'x' and 'y' to compute 'b_out'
    with pytest.raises(ValueError, match="Missing required input"):
        pipeline.run(inputs={"x": 1}, output_name="b_out")

    # This should work - we have both inputs needed for 'b_out'
    result = pipeline.run(inputs={"x": 1, "y": 10}, output_name="b_out")
    assert result == {"b_out": 12}


def test_selective_inputs_multiple_outputs():
    """Test with multiple requested outputs."""

    @node(output_name="a_out")
    def a(x: int) -> int:
        return x + 1

    @node(output_name="b_out")
    def b(y: int) -> int:
        return y * 2

    @node(output_name="c_out")
    def c(a_out: int, b_out: int, z: int) -> int:
        return a_out + b_out + z

    pipeline = Pipeline(nodes=[a, b, c])

    # Requesting only a_out and b_out should only need x and y
    result = pipeline.run(inputs={"x": 1, "y": 2}, output_name=["a_out", "b_out"])
    assert result == {"a_out": 2, "b_out": 4}

    # Requesting c_out should need all inputs
    with pytest.raises(ValueError, match="Missing required input"):
        pipeline.run(inputs={"x": 1, "y": 2}, output_name="c_out")

    result = pipeline.run(inputs={"x": 1, "y": 2, "z": 3}, output_name="c_out")
    assert result == {"c_out": 9}


def test_selective_inputs_chain():
    """Test with a chain of dependencies."""

    @node(output_name="a")
    def step_a(x: int) -> int:
        return x + 1

    @node(output_name="b")
    def step_b(a: int) -> int:
        return a * 2

    @node(output_name="c")
    def step_c(b: int, y: int) -> int:
        return b + y

    pipeline = Pipeline(nodes=[step_a, step_b, step_c])

    # Getting 'a' should only need 'x'
    result = pipeline.run(inputs={"x": 5}, output_name="a")
    assert result == {"a": 6}

    # Getting 'b' should only need 'x' (not y)
    result = pipeline.run(inputs={"x": 5}, output_name="b")
    assert result == {"b": 12}

    # Getting 'c' should need both x and y
    with pytest.raises(ValueError, match="Missing required input"):
        pipeline.run(inputs={"x": 5}, output_name="c")

    result = pipeline.run(inputs={"x": 5, "y": 3}, output_name="c")
    assert result == {"c": 15}


def test_selective_inputs_with_map():
    """Test selective inputs work with map operation."""

    @node(output_name="a")
    def step_a(x: int) -> int:
        return x + 1

    @node(output_name="b")
    def step_b(a: int, y: int) -> int:
        return a + y

    pipeline = Pipeline(nodes=[step_a, step_b])

    # Map over x, only requesting 'a' - should not need 'y'
    results = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x", output_name="a")
    assert results == [{"a": 2}, {"a": 3}, {"a": 4}]

    # Map over x, requesting 'b' - should need 'y'
    with pytest.raises(ValueError, match="Missing required input"):
        pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x", output_name="b")

    # With y provided, should work
    results = pipeline.map(
        inputs={"x": [1, 2, 3], "y": 10}, map_over="x", output_name="b"
    )
    assert results == [{"b": 12}, {"b": 13}, {"b": 14}]
