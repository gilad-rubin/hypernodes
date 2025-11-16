"""Tests for typed interface (TypedDict) generation."""

from typing import get_type_hints

import pytest

from hypernodes import Pipeline, node


def test_get_input_type_simple():
    """Test generating TypedDict for simple pipeline inputs."""

    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    @node(output_name="sum")
    def add(doubled: int, y: int) -> int:
        return doubled + y

    pipeline = Pipeline(nodes=[double, add], name="Math")

    # Get the input type
    InputType = pipeline.get_input_type()

    # Verify class name
    assert InputType.__name__ == "MathInput"

    # Verify it has the correct fields
    annotations = InputType.__annotations__
    assert "x" in annotations
    assert "y" in annotations
    assert annotations["x"] == int
    assert annotations["y"] == int


def test_get_output_type_simple():
    """Test generating TypedDict for simple pipeline outputs."""

    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    @node(output_name="sum")
    def add(doubled: int, y: int) -> int:
        return doubled + y

    pipeline = Pipeline(nodes=[double, add], name="Math")

    # Get the output type
    OutputType = pipeline.get_output_type()

    # Verify class name
    assert OutputType.__name__ == "MathOutput"

    # Verify it has the correct fields
    annotations = OutputType.__annotations__
    assert "doubled" in annotations
    assert "sum" in annotations
    assert annotations["doubled"] == int
    assert annotations["sum"] == int


def test_typed_interface_usage():
    """Test using the typed interface with actual pipeline execution."""

    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    @node(output_name="sum")
    def add(doubled: int, y: int) -> int:
        return doubled + y

    pipeline = Pipeline(nodes=[double, add])

    # Get types
    InputType = pipeline.get_input_type()
    OutputType = pipeline.get_output_type()

    # Create typed input
    inputs: InputType = {"x": 5, "y": 10}

    # Run pipeline
    result = pipeline.run(inputs=inputs)

    # Type the result
    typed_result: OutputType = result

    # Verify results
    assert typed_result["doubled"] == 10
    assert typed_result["sum"] == 20


def test_unnamed_pipeline():
    """Test TypedDict generation for unnamed pipeline."""

    @node(output_name="result")
    def process(x: int) -> int:
        return x * 2

    pipeline = Pipeline(nodes=[process])  # No name

    InputType = pipeline.get_input_type()
    OutputType = pipeline.get_output_type()

    # Should use default name
    assert InputType.__name__ == "PipelineInput"
    assert OutputType.__name__ == "PipelineOutput"


def test_multiple_inputs():
    """Test TypedDict with multiple input types."""

    @node(output_name="result")
    def process(x: int, name: str, factor: float) -> str:
        return f"{name}: {x * factor}"

    pipeline = Pipeline(nodes=[process])

    InputType = pipeline.get_input_type()

    annotations = InputType.__annotations__
    assert annotations["x"] == int
    assert annotations["name"] == str
    assert annotations["factor"] == float


def test_output_type_string():
    """Test TypedDict with string output type."""

    @node(output_name="message")
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    pipeline = Pipeline(nodes=[greet])

    OutputType = pipeline.get_output_type()

    annotations = OutputType.__annotations__
    assert annotations["message"] == str


def test_complex_types():
    """Test TypedDict with complex types like List."""
    from typing import List

    @node(output_name="items")
    def get_items(count: int) -> List[int]:
        return list(range(count))

    pipeline = Pipeline(nodes=[get_items])

    InputType = pipeline.get_input_type()
    OutputType = pipeline.get_output_type()

    # Verify types are preserved
    assert InputType.__annotations__["count"] == int
    assert OutputType.__annotations__["items"] == List[int]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

