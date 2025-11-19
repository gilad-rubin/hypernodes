"""Comprehensive tests for Daft operations using real Daft engine."""

import pytest
import sys
import asyncio
from typing import List, Dict

# Skip if daft is not installed
daft = pytest.importorskip("daft")

from hypernodes import Pipeline, node, stateful
from hypernodes.integrations.daft.engine import DaftEngine
from hypernodes.integrations.daft.operations import (
    FunctionNodeOperation,
    DualNodeOperation,
    BatchNodeOperation,
    PipelineNodeOperation,
    ExecutionContext
)
from hypernodes.pipeline_node import PipelineNode

# ===== Test Definitions (Module Level) =====

@node(output_name="y")
async def async_node_func(x: int) -> int:
    await asyncio.sleep(0.01)
    return x * 2

@stateful
class StatefulModel:
    def __init__(self, offset: int):
        self.offset = offset
        
    def predict(self, x: int) -> int:
        return x + self.offset
        
@node(output_name="y")
def predict_node_func(x: int, model: StatefulModel) -> int:
    return model.predict(x)


# ===== Type Inference Tests =====

def test_type_inference_string_annotation():
    """Test that string annotations (PEP 563) are evaluated correctly."""
    
    class MyType:
        pass
        
    def my_func(x) -> "MyType":
        return MyType()
        
    # We need to make sure MyType is available in the function's globals
    my_func.__globals__["MyType"] = MyType
    
    @node(output_name="out")
    def node_func(x):
        return my_func(x)
    
    # Access the operation directly to test inference logic
    op = FunctionNodeOperation(node_func)
    
    dtype = op._infer_daft_return_type(my_func)
    # Should default to Python type for custom classes
    assert dtype == daft.DataType.python()


def test_type_inference_list_return():
    """Test that list return types are correctly inferred."""
    
    def list_func(x) -> List[int]:
        return [x, x + 1]
    
    @node(output_name="out")
    def node_func(x):
        return list_func(x)
        
    op = FunctionNodeOperation(node_func)
    
    dtype = op._infer_daft_return_type(list_func)
    # Should be List[Python] because Daft UDFs usually return Python types inside lists
    # or we might map it to List(Int64) if we were stricter, but current logic uses python
    assert dtype == daft.DataType.list(daft.DataType.python())


# ===== DualNode Tests =====

def test_dual_node_execution():
    """Test that DualNode works correctly in both run and map modes."""
    
    @node(output_name="y")
    def singular_func(x: int) -> int:
        return x + 1
        
    def batch_func(x: List[int]) -> List[int]:
        return [v + 10 for v in x]
        
    # Manually attach batch function (simulating DualNode creation)
    singular_func.singular = singular_func.func
    singular_func.batch = batch_func
    singular_func.is_dual_node = True
    
    pipeline = Pipeline(nodes=[singular_func])
    engine = DaftEngine()
    
    # 1. Run (Singular)
    result = engine.run(pipeline, {"x": 1})
    assert result["y"] == 2
    
    # 2. Map (Batch)
    results = engine.map(pipeline, {"x": [1, 2]}, map_over="x")
    # Batch function adds 10
    assert results[0]["y"] == 11
    assert results[1]["y"] == 12


# ===== Batch Operation Tests =====

def test_batch_node_constant_unwrapping():
    """Test that batch operations unwrap constants correctly."""
    
    @node(output_name="z")
    def add(x: int, y: int) -> int:
        return x + y
        
    pipeline = Pipeline(nodes=[add])
    engine = DaftEngine(use_batch_udf=True)
    
    # Map over x, but y is constant
    inputs = {"x": [1, 2, 3], "y": 10}
    results = engine.map(pipeline, inputs, map_over="x")
    
    assert len(results) == 3
    assert results[0]["z"] == 11
    assert results[1]["z"] == 12
    assert results[2]["z"] == 13


def test_batch_udf_not_used_for_list_return():
    """Test that list-returning nodes don't use batch UDF."""
    
    @node(output_name="items")
    def list_func(x: int) -> List[int]:
        return [x, x + 1]
    
    # Check engine logic
    engine = DaftEngine(use_batch_udf=True)
    engine._is_map_context = True
    
    should_batch = engine._should_use_batch_udf(list_func)
    assert should_batch is False
    
    # Verify execution works
    pipeline = Pipeline(nodes=[list_func])
    results = engine.map(pipeline, {"x": [1]}, map_over="x")
    assert results[0]["items"] == [1, 2]


# ===== Async Node Tests =====

@pytest.mark.asyncio
async def test_async_node_handling():
    """Test that async functions are correctly executed."""
    
    pipeline = Pipeline(nodes=[async_node_func])
    engine = DaftEngine()
    
    # Run
    result = engine.run(pipeline, {"x": 5})
    assert result["y"] == 10
    
    # Map
    results = engine.map(pipeline, {"x": [1, 2]}, map_over="x")
    assert results[0]["y"] == 2
    assert results[1]["y"] == 4


# ===== Stateful UDF Tests =====

def test_stateful_udf_execution():
    """Test that stateful parameters work correctly."""
        
    model = StatefulModel(offset=10)
    pipeline = Pipeline(nodes=[predict_node_func])
    engine = DaftEngine()
    
    # Run
    result = engine.run(pipeline, {"x": 5, "model": model})
    assert result["y"] == 15
    
    # Map
    results = engine.map(pipeline, {"x": [1, 2], "model": model}, map_over="x")
    assert results[0]["y"] == 11
    assert results[1]["y"] == 12


# ===== Code Generation Tests =====

def test_codegen_simple_node():
    """Test code generation for a simple node."""
    
    @node(output_name="y")
    def add_one(x: int) -> int:
        return x + 1
    
    pipeline = Pipeline(nodes=[add_one])
    engine = DaftEngine()
    
    code = engine.generate_code(pipeline, {"x": 1})
    
    assert 'with_column("y"' in code
    assert 'daft.col("x")' in code
    assert "@daft.func" in code
    assert "def add_one(x: int) -> int:" in code


# ===== Pipeline Operation Tests =====

def test_nested_pipeline_execution():
    """Test execution of nested pipelines."""
    
    @node(output_name="inner")
    def inner_func(x: int) -> int:
        return x * 2
        
    inner_pipeline = Pipeline(nodes=[inner_func])
    
    @node(output_name="outer")
    def outer_func(inner: int) -> int:
        return inner + 1
        
    pipeline_node = PipelineNode(pipeline=inner_pipeline)
    outer_pipeline = Pipeline(nodes=[pipeline_node, outer_func])
    
    engine = DaftEngine()
    
    # Run
    result = engine.run(outer_pipeline, {"x": 5})
    assert result["outer"] == 11  # (5 * 2) + 1
    
    # Map
    results = engine.map(outer_pipeline, {"x": [1, 2]}, map_over="x")
    assert results[0]["outer"] == 3  # (1 * 2) + 1
    assert results[1]["outer"] == 5  # (2 * 2) + 1


def test_nested_pipeline_with_map_over():
    """Test execution of nested pipeline with internal map_over."""
    
    @node(output_name="y")
    def add_one(x: int) -> int:
        return x + 1
        
    inner_pipeline = Pipeline(nodes=[add_one])
    
    # Map over 'items' inside the nested pipeline
    pipeline_node = PipelineNode(pipeline=inner_pipeline, map_over="items", input_mapping={"items": "x"})
    
    engine = DaftEngine()
    outer_pipeline = Pipeline(nodes=[pipeline_node])
    
    # Input is a list of lists
    inputs = {"items": [[1, 2], [3, 4]]}
    
    # We map over the outer list
    results = engine.map(outer_pipeline, inputs, map_over="items")
    
    # Result should be list of dicts, where 'y' is a list (aggregated from inner map)
    assert len(results) == 2
    assert results[0]["y"] == [2, 3]
    assert results[1]["y"] == [4, 5]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
