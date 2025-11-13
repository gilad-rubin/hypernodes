"""Tests for DaftBackend - automatic conversion of HyperNodes to Daft.

These tests verify that HyperNodes pipelines can be executed using Daft
with automatic conversion to Daft UDFs and DataFrame operations.
"""

import pytest

try:
    import daft
    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False

from hypernodes import node, Pipeline

if DAFT_AVAILABLE:
    from hypernodes.engines import DaftEngine

pytestmark = pytest.mark.skipif(not DAFT_AVAILABLE, reason="Daft not installed")


def test_daft_backend_single_node():
    """Test DaftBackend with a single node."""
    @node(output_name="result")
    def add_one(x: int) -> int:
        return x + 1
    
    pipeline = Pipeline(nodes=[add_one], engine=DaftEngine())
    result = pipeline.run(inputs={"x": 5})
    
    assert result == {"result": 6}


def test_daft_backend_two_sequential_nodes():
    """Test DaftBackend with two sequential nodes."""
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="result")
    def add_one(doubled: int) -> int:
        return doubled + 1
    
    pipeline = Pipeline(nodes=[double, add_one], engine=DaftEngine())
    result = pipeline.run(inputs={"x": 5})
    
    assert result == {"doubled": 10, "result": 11}


def test_daft_backend_diamond_pattern():
    """Test DaftBackend with diamond dependency pattern."""
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="tripled")
    def triple(x: int) -> int:
        return x * 3
    
    @node(output_name="result")
    def add(doubled: int, tripled: int) -> int:
        return doubled + tripled
    
    pipeline = Pipeline(nodes=[double, triple, add], engine=DaftEngine())
    result = pipeline.run(inputs={"x": 5})
    
    assert result == {"doubled": 10, "tripled": 15, "result": 25}


def test_daft_backend_map_single_parameter():
    """Test DaftBackend with map over single parameter."""
    @node(output_name="result")
    def add_one(x: int) -> int:
        return x + 1
    
    pipeline = Pipeline(nodes=[add_one], engine=DaftEngine())
    results = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
    
    assert results == {"result": [2, 3, 4]}


def test_daft_backend_map_two_sequential_nodes():
    """Test DaftBackend with map over two sequential nodes."""
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="result")
    def add_one(doubled: int) -> int:
        return doubled + 1
    
    pipeline = Pipeline(nodes=[double, add_one], engine=DaftEngine())
    results = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
    
    assert results == {"doubled": [2, 4, 6], "result": [3, 5, 7]}


def test_daft_backend_map_with_fixed_parameter():
    """Test DaftBackend with map having fixed and varying parameters."""
    @node(output_name="result")
    def multiply(x: int, factor: int) -> int:
        return x * factor
    
    pipeline = Pipeline(nodes=[multiply], engine=DaftEngine())
    results = pipeline.map(inputs={"x": [1, 2, 3], "factor": 10}, map_over="x")
    
    assert results == {"result": [10, 20, 30]}


def test_daft_backend_empty_map():
    """Test DaftBackend with empty map input."""
    @node(output_name="result")
    def add_one(x: int) -> int:
        return x + 1
    
    pipeline = Pipeline(nodes=[add_one], engine=DaftEngine())
    results = pipeline.map(inputs={"x": []}, map_over="x")
    
    assert results == {"result": []}


def test_daft_backend_multiple_inputs():
    """Test DaftBackend with multiple independent inputs."""
    @node(output_name="sum")
    def add(x: int, y: int) -> int:
        return x + y
    
    @node(output_name="product")
    def multiply(x: int, y: int) -> int:
        return x * y
    
    @node(output_name="result")
    def combine(sum: int, product: int) -> int:
        return sum + product
    
    pipeline = Pipeline(nodes=[add, multiply, combine], engine=DaftEngine())
    result = pipeline.run(inputs={"x": 5, "y": 3})
    
    assert result == {"sum": 8, "product": 15, "result": 23}


def test_daft_backend_nested_pipeline():
    """Test DaftBackend with nested pipeline."""
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="incremented")
    def add_one(doubled: int) -> int:
        return doubled + 1
    
    inner_pipeline = Pipeline(nodes=[double, add_one])
    
    @node(output_name="result")
    def square(incremented: int) -> int:
        return incremented ** 2
    
    outer_pipeline = Pipeline(nodes=[inner_pipeline, square], engine=DaftEngine())
    result = outer_pipeline.run(inputs={"x": 5})
    
    assert result == {"doubled": 10, "incremented": 11, "result": 121}


def test_daft_backend_selective_output():
    """Test DaftBackend with selective output."""
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="tripled")
    def triple(x: int) -> int:
        return x * 3
    
    @node(output_name="result")
    def add(doubled: int, tripled: int) -> int:
        return doubled + tripled
    
    pipeline = Pipeline(nodes=[double, triple, add], engine=DaftEngine())
    
    # Only request final result
    result = pipeline.run(inputs={"x": 5}, output_name="result")
    assert result == {"result": 25}
    
    # Request multiple outputs
    result = pipeline.run(inputs={"x": 5}, output_name=["doubled", "result"])
    assert result == {"doubled": 10, "result": 25}


def test_daft_backend_string_operations():
    """Test DaftBackend with string operations."""
    @node(output_name="cleaned")
    def clean_text(text: str) -> str:
        return text.strip().lower()
    
    @node(output_name="tokens")
    def tokenize(cleaned: str) -> list:
        return cleaned.split()
    
    @node(output_name="count")
    def count_tokens(tokens: list) -> int:
        return len(tokens)
    
    pipeline = Pipeline(nodes=[clean_text, tokenize, count_tokens], engine=DaftEngine())
    result = pipeline.run(inputs={"text": "  Hello World  "})
    
    assert result["cleaned"] == "hello world"
    assert result["tokens"] == ["hello", "world"]
    assert result["count"] == 2


def test_daft_backend_map_string_operations():
    """Test DaftBackend with map over string operations."""
    @node(output_name="cleaned")
    def clean_text(text: str) -> str:
        return text.strip().lower()
    
    @node(output_name="count")
    def count_chars(cleaned: str) -> int:
        return len(cleaned)
    
    pipeline = Pipeline(nodes=[clean_text, count_chars], engine=DaftEngine())
    results = pipeline.map(
        inputs={"text": ["  Hello  ", "  World  ", "  Test  "]},
        map_over="text"
    )
    
    assert results["cleaned"] == ["hello", "world", "test"]
    assert results["count"] == [5, 5, 4]
