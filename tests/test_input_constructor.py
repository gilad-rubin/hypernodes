"""Tests for input constructor function."""

import pytest

from hypernodes import Pipeline, node


def test_input_constructor_simple():
    """Test generating input constructor for simple pipeline."""

    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    @node(output_name="sum")
    def add(doubled: int, y: int) -> int:
        return doubled + y

    pipeline = Pipeline(nodes=[double, add], name="Math")

    # Get the input constructor
    make_input = pipeline.get_input_constructor()

    # Use it to create inputs
    inputs = make_input(x=5, y=10)

    # Verify it creates the correct dict
    assert inputs == {"x": 5, "y": 10}

    # Verify it works with the pipeline
    result = pipeline.run(inputs=inputs)
    assert result["doubled"] == 10
    assert result["sum"] == 20


def test_input_constructor_type_annotations():
    """Test that input constructor has correct type annotations."""

    @node(output_name="result")
    def process(x: int, name: str, factor: float) -> str:
        return f"{name}: {x * factor}"

    pipeline = Pipeline(nodes=[process])

    make_input = pipeline.get_input_constructor()

    # Check that function has type annotations
    assert "x" in make_input.__annotations__
    assert "name" in make_input.__annotations__
    assert "factor" in make_input.__annotations__
    assert make_input.__annotations__["x"] == int
    assert make_input.__annotations__["name"] == str
    assert make_input.__annotations__["factor"] == float


def test_input_constructor_with_pipeline():
    """Test using input constructor with actual pipeline execution."""

    @node(output_name="message")
    def greet(name: str, greeting: str) -> str:
        return f"{greeting}, {name}!"

    pipeline = Pipeline(nodes=[greet])

    make_input = pipeline.get_input_constructor()

    # Create inputs using constructor
    inputs = make_input(name="World", greeting="Hello")

    # Run pipeline
    result = pipeline.run(inputs=inputs)
    assert result["message"] == "Hello, World!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

