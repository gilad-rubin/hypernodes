"""Tests for fluent API with typed inputs (with_inputs and with_map_inputs)."""

import pytest

from hypernodes import Pipeline, node


def test_with_inputs_single_execution():
    """Test with_inputs for single execution."""

    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    @node(output_name="sum")
    def add(doubled: int, y: int) -> int:
        return doubled + y

    pipeline = Pipeline(nodes=[double, add])

    # Single execution
    result = pipeline.with_inputs(x=5, y=10).run()

    assert result["doubled"] == 10
    assert result["sum"] == 20


def test_with_inputs_rejects_lists():
    """Test that with_inputs raises TypeError for list inputs."""

    @node(output_name="result")
    def process(x: int) -> int:
        return x * 2

    pipeline = Pipeline(nodes=[process])

    # Should raise TypeError for lists when type is not List
    with pytest.raises(TypeError, match="received a list"):
        pipeline.with_inputs(x=[1, 2, 3])


def test_with_inputs_with_output_name():
    """Test with_inputs with output selection."""

    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    @node(output_name="tripled")
    def triple(x: int) -> int:
        return x * 3

    pipeline = Pipeline(nodes=[double, triple])

    # Select specific output
    result = pipeline.with_inputs(x=5).with_output_name("doubled").run()

    assert "doubled" in result
    assert "tripled" not in result
    assert result["doubled"] == 10


def test_with_map_inputs_single_param():
    """Test with_map_inputs for mapping over single parameter."""

    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    @node(output_name="sum")
    def add(doubled: int, y: int) -> int:
        return doubled + y

    pipeline = Pipeline(nodes=[double, add])

    # Map over x, broadcast y
    results = pipeline.with_map_inputs(x=[1, 2, 3], y=10).map(map_over="x")

    assert len(results) == 3
    assert results[0] == {"doubled": 2, "sum": 12}
    assert results[1] == {"doubled": 4, "sum": 14}
    assert results[2] == {"doubled": 6, "sum": 16}


def test_with_map_inputs_multiple_params_zip():
    """Test with_map_inputs for mapping over multiple parameters (zip mode)."""

    @node(output_name="sum")
    def add(x: int, y: int) -> int:
        return x + y

    pipeline = Pipeline(nodes=[add])

    # Zip mode (parallel iteration)
    results = pipeline.with_map_inputs(x=[1, 2, 3], y=[10, 20, 30]).map(map_over=["x", "y"])

    assert len(results) == 3
    assert results[0] == {"sum": 11}
    assert results[1] == {"sum": 22}
    assert results[2] == {"sum": 33}


def test_with_map_inputs_multiple_params_product():
    """Test with_map_inputs for mapping over multiple parameters (product mode)."""

    @node(output_name="product")
    def multiply(x: int, y: int) -> int:
        return x * y

    pipeline = Pipeline(nodes=[multiply])

    # Product mode (all combinations)
    results = pipeline.with_map_inputs(x=[2, 3], y=[10, 100]).map(
        map_over=["x", "y"], map_mode="product"
    )

    assert len(results) == 4
    assert results[0] == {"product": 20}  # 2 * 10
    assert results[1] == {"product": 200}  # 2 * 100
    assert results[2] == {"product": 30}  # 3 * 10
    assert results[3] == {"product": 300}  # 3 * 100


def test_with_map_inputs_validates_list_for_mapped_param():
    """Test that with_map_inputs validates mapped parameters are lists."""

    @node(output_name="result")
    def process(x: int) -> int:
        return x * 2

    pipeline = Pipeline(nodes=[process])

    # Should raise TypeError if mapped param is not a list
    with pytest.raises(TypeError, match="not a list"):
        pipeline.with_map_inputs(x=5).map(map_over="x")


def test_with_map_inputs_validates_non_mapped_not_list():
    """Test that with_map_inputs validates non-mapped parameters are not lists."""

    @node(output_name="sum")
    def add(x: int, y: int) -> int:
        return x + y

    pipeline = Pipeline(nodes=[add])

    # Should raise TypeError if non-mapped param is a list
    with pytest.raises(TypeError, match="not in map_over"):
        pipeline.with_map_inputs(x=[1, 2, 3], y=[10, 20, 30]).map(map_over="x")


def test_with_map_inputs_run_raises_error():
    """Test that calling .run() on with_map_inputs raises error."""

    @node(output_name="result")
    def process(x: int) -> int:
        return x * 2

    pipeline = Pipeline(nodes=[process])

    # Should raise RuntimeError
    with pytest.raises(RuntimeError, match="Cannot call .run()"):
        pipeline.with_map_inputs(x=[1, 2, 3]).run()


def test_with_inputs_map_raises_error():
    """Test that calling .map() on with_inputs raises error."""

    @node(output_name="result")
    def process(x: int) -> int:
        return x * 2

    pipeline = Pipeline(nodes=[process])

    # Should raise RuntimeError
    with pytest.raises(RuntimeError, match="only be called on with_map_inputs"):
        pipeline.with_inputs(x=5).map(map_over="x")


def test_with_map_inputs_with_output_name():
    """Test with_map_inputs with output selection."""

    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    @node(output_name="tripled")
    def triple(x: int) -> int:
        return x * 3

    pipeline = Pipeline(nodes=[double, triple])

    # Select specific output
    results = pipeline.with_map_inputs(x=[1, 2, 3]).with_output_name("doubled").map(map_over="x")

    assert len(results) == 3
    assert all("doubled" in r for r in results)
    assert all("tripled" not in r for r in results)


def test_with_map_inputs_broadcast_scalar():
    """Test that scalar values are properly broadcast in map execution."""

    @node(output_name="result")
    def add_constant(x: int, constant: int) -> int:
        return x + constant

    pipeline = Pipeline(nodes=[add_constant])

    # constant should be broadcast to all iterations
    results = pipeline.with_map_inputs(x=[1, 2, 3], constant=100).map(map_over="x")

    assert len(results) == 3
    assert results[0] == {"result": 101}
    assert results[1] == {"result": 102}
    assert results[2] == {"result": 103}


def test_fluent_api_chaining():
    """Test chaining multiple fluent methods."""

    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    @node(output_name="sum")
    def add(doubled: int, y: int) -> int:
        return doubled + y

    pipeline = Pipeline(nodes=[double, add])

    # Chain with_output_name() before map()
    results = (
        pipeline.with_map_inputs(x=[1, 2, 3], y=10).with_output_name("sum").map(map_over="x")
    )

    assert len(results) == 3
    assert all("sum" in r for r in results)
    assert all("doubled" not in r for r in results)


def test_complex_type_handling():
    """Test with_inputs/with_map_inputs with complex types."""
    from typing import List

    @node(output_name="total")
    def sum_items(items: List[int]) -> int:
        return sum(items)

    pipeline = Pipeline(nodes=[sum_items])

    # Single execution with list (the type IS list)
    result = pipeline.with_inputs(items=[1, 2, 3, 4, 5]).run()
    assert result["total"] == 15

    # Map execution - each iteration gets a list
    results = pipeline.with_map_inputs(items=[[1, 2], [3, 4], [5]]).map(map_over="items")
    assert len(results) == 3
    assert results[0]["total"] == 3
    assert results[1]["total"] == 7
    assert results[2]["total"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

