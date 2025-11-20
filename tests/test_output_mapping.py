"""Test that output_mapping correctly filters outputs from nested pipelines."""

import pytest

from hypernodes import Pipeline, node


def test_output_mapping_filters_outputs():
    """Test that output_mapping only exposes mapped outputs."""
    
    @node(output_name="a")
    def make_a(x: int) -> int:
        return x * 2
    
    @node(output_name="b")
    def make_b(x: int) -> int:
        return x * 3
    
    @node(output_name="c")
    def make_c(x: int) -> int:
        return x * 4
    
    inner = Pipeline(nodes=[make_a, make_b, make_c])
    
    # Only expose 'a' as 'mapped'
    wrapped = inner.as_node(output_mapping={"a": "mapped"})
    
    # Should only expose 'mapped', not 'b' or 'c'
    assert wrapped.output_name == "mapped"


def test_output_mapping_computes_only_required_outputs():
    """Test that inner pipeline only computes the outputs that are exposed via mapping."""
    execution_log = []
    
    @node(output_name="result1")
    def compute_result1(x: int) -> int:
        execution_log.append("result1")
        return x * 2
    
    @node(output_name="result2")
    def compute_result2(x: int) -> int:
        execution_log.append("result2")
        return x * 3
    
    @node(output_name="result3")
    def compute_result3(x: int) -> int:
        execution_log.append("result3")
        return x * 4
    
    inner = Pipeline(nodes=[compute_result1, compute_result2, compute_result3])
    
    # Only expose result1 as "mapped"
    wrapped = inner.as_node(output_mapping={"result1": "mapped"})
    
    @node(output_name="final")
    def use_mapped(mapped: int) -> int:
        execution_log.append("final")
        return mapped + 100
    
    outer = Pipeline(nodes=[wrapped, use_mapped])
    
    # Execute
    result = outer.run(inputs={"x": 5})
    
    # Check that only result1 and final were computed
    assert execution_log == ["result1", "final"]
    assert result == {"mapped": 10, "final": 110}


def test_output_mapping_with_multiple_exposed_outputs():
    """Test that output_mapping can expose multiple outputs."""
    
    @node(output_name="a")
    def make_a(x: int) -> int:
        return x * 2
    
    @node(output_name="b")
    def make_b(x: int) -> int:
        return x * 3
    
    @node(output_name="c")
    def make_c(x: int) -> int:
        return x * 4
    
    inner = Pipeline(nodes=[make_a, make_b, make_c])
    
    # Expose 'a' as 'first' and 'b' as 'second'
    wrapped = inner.as_node(output_mapping={"a": "first", "b": "second"})
    
    # Should expose both outputs
    assert wrapped.output_name == ("first", "second")
    
    # Execute
    result = wrapped(x=5)
    
    # Should only return the mapped outputs
    assert result == {"first": 10, "second": 15}
    # Should NOT include 'c'
    assert "c" not in result


def test_output_mapping_without_mapping_returns_all():
    """Test that without output_mapping, all outputs are exposed."""
    
    @node(output_name="a")
    def make_a(x: int) -> int:
        return x * 2
    
    @node(output_name="b")
    def make_b(x: int) -> int:
        return x * 3
    
    inner = Pipeline(nodes=[make_a, make_b])
    
    # No output mapping
    wrapped = inner.as_node()
    
    # Should expose all outputs
    assert wrapped.output_name == ("a", "b")
    
    # Execute
    result = wrapped(x=5)
    
    # Should return all outputs
    assert result == {"a": 10, "b": 15}


def test_output_mapping_with_selective_execution():
    """Test that selective execution works correctly with output_mapping."""
    execution_log = []
    
    @node(output_name="a")
    def make_a(x: int) -> int:
        execution_log.append("a")
        return x * 2
    
    @node(output_name="b")
    def make_b(a: int) -> int:
        execution_log.append("b")
        return a * 3
    
    @node(output_name="c")
    def make_c(b: int) -> int:
        execution_log.append("c")
        return b * 4
    
    inner = Pipeline(nodes=[make_a, make_b, make_c])
    
    # Expose all outputs with different names
    wrapped = inner.as_node(output_mapping={"a": "first", "b": "second", "c": "third"})
    
    @node(output_name="result")
    def use_first(first: int) -> int:
        execution_log.append("result")
        return first + 100
    
    outer = Pipeline(nodes=[wrapped, use_first])
    
    # Execute - should only compute 'first' (mapped from 'a')
    result = outer.run(inputs={"x": 5})
    
    # Should only execute 'a' and 'result', not 'b' or 'c'
    assert execution_log == ["a", "result"]
    assert result == {"first": 10, "result": 110}

