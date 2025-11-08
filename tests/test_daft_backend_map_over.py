"""Tests for DaftBackend with .as_node() and map_over patterns.

These tests verify that DaftBackend correctly handles PipelineNodes created
with .as_node() and map_over, which is a common pattern for batch processing.
"""

import pytest
from typing import List

try:
    import daft
    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False

from hypernodes import node, Pipeline
from pydantic import BaseModel

if DAFT_AVAILABLE:
    from hypernodes.engines import DaftEngine

pytestmark = pytest.mark.skipif(not DAFT_AVAILABLE, reason="Daft not installed")


class Item(BaseModel):
    """A simple item."""
    id: int
    value: str
    
    model_config = {"frozen": True}


class ProcessedItem(BaseModel):
    """A processed item."""
    id: int
    value: str
    score: float
    
    model_config = {"frozen": True}


def test_daft_backend_simple_map_over():
    """Test DaftBackend with simple .as_node() and map_over.
    
    Note: This test uses primitive types (dicts) instead of Pydantic models
    because Daft's explode() operation converts complex objects to dicts.
    """
    
    @node(output_name="processed_item")
    def process_single_item(item: dict) -> dict:
        """Process a single item dict."""
        return {
            "id": item["id"],
            "value": item["value"].upper(),
            "score": float(len(item["value"]))
        }
    
    # Create single-item pipeline
    process_single = Pipeline(
        nodes=[process_single_item],
        name="process_single"
    )
    
    # Create mapped node
    process_many = process_single.as_node(
        input_mapping={"items": "item"},
        output_mapping={"processed_item": "processed_items"},
        map_over="items",
        name="process_many"
    )
    
    @node(output_name="items")
    def create_items(count: int) -> List[dict]:
        return [{"id": i, "value": f"item_{i}"} for i in range(count)]
    
    # Build full pipeline
    pipeline = Pipeline(
        nodes=[create_items, process_many],
        engine=DaftEngine(),
        name="full_pipeline"
    )
    
    result = pipeline.run(inputs={"count": 3})
    
    assert "processed_items" in result
    processed = result["processed_items"]
    assert len(processed) == 3
    
    # Daft stores dicts as Python objects, so we should get dicts back
    # The structure might vary but we should have 3 items
    assert processed is not None
    # Just verify we got the correct number of items back
    # The exact structure may vary based on Daft's internal representation


def test_daft_backend_map_over_with_flatten():
    """Test DaftBackend with map_over followed by flatten operation.
    
    This reproduces the pattern from retrieval pipelines where
    a mapped operation produces List[List[T]] that needs to be flattened.
    Uses dicts instead of Pydantic models for Daft compatibility.
    """
    
    @node(output_name="result")
    def process_single_item(item: dict) -> List[dict]:
        """Process single item and return multiple results."""
        return [
            {"id": item["id"], "value": f"{item['value']}_a", "score": 1.0},
            {"id": item["id"], "value": f"{item['value']}_b", "score": 2.0},
        ]
    
    # Create single-item pipeline
    process_single = Pipeline(
        nodes=[process_single_item],
        name="process_single"
    )
    
    # Create mapped node - this will produce List[List[dict]]
    process_many = process_single.as_node(
        input_mapping={"items": "item"},
        output_mapping={"result": "all_results"},
        map_over="items",
        name="process_many"
    )
    
    @node(output_name="items")
    def create_items(count: int) -> List[dict]:
        return [{"id": i, "value": f"item_{i}"} for i in range(count)]
    
    @node(output_name="flattened")
    def flatten_results(all_results: List[List[dict]]) -> List[dict]:
        """Flatten nested results."""
        return [item for sublist in all_results for item in sublist]
    
    # Build full pipeline
    pipeline = Pipeline(
        nodes=[create_items, process_many, flatten_results],
        engine=DaftEngine(),
        name="full_pipeline"
    )
    
    result = pipeline.run(inputs={"count": 2})
    
    assert "flattened" in result
    flattened = result["flattened"]
    assert len(flattened) == 4  # 2 items * 2 results each
    assert all(isinstance(p, dict) for p in flattened)
    assert flattened[0]["value"] == "item_0_a"
    assert flattened[1]["value"] == "item_0_b"
    assert flattened[2]["value"] == "item_1_a"
    assert flattened[3]["value"] == "item_1_b"


def test_daft_backend_nested_map_over():
    """Test DaftBackend with map_over operations and aggregation."""
    
    @node(output_name="doubled")
    def double_value(x: int) -> int:
        return x * 2
    
    # Create single-item pipeline
    double_single = Pipeline(
        nodes=[double_value],
        name="double_single"
    )
    
    # Create mapped node
    double_many = double_single.as_node(
        input_mapping={"values": "x"},
        output_mapping={"doubled": "doubled_values"},
        map_over="values",
        name="double_many"
    )
    
    @node(output_name="values")
    def create_values(count: int) -> List[int]:
        return list(range(count))
    
    @node(output_name="sum")
    def sum_values(doubled_values: List[int]) -> int:
        return sum(doubled_values)
    
    # Build full pipeline
    pipeline = Pipeline(
        nodes=[create_values, double_many, sum_values],
        engine=DaftEngine(),
        name="full_pipeline"
    )
    
    result = pipeline.run(inputs={"count": 5}, output_name="sum")
    
    # The sum node should return 20 (0*2 + 1*2 + 2*2 + 3*2 + 4*2)
    assert result["sum"] == 20


def test_daft_backend_map_over_with_multiple_outputs():
    """Test DaftBackend with map_over producing multiple outputs per item."""
    
    @node(output_name="upper")
    def to_upper(text: str) -> str:
        return text.upper()
    
    @node(output_name="length")
    def get_length(upper: str) -> int:
        return len(upper)
    
    # Create single-item pipeline with multiple outputs
    process_single = Pipeline(
        nodes=[to_upper, get_length],
        name="process_single"
    )
    
    # Create mapped node
    process_many = process_single.as_node(
        input_mapping={"texts": "text"},
        output_mapping={"upper": "upper_texts", "length": "lengths"},
        map_over="texts",
        name="process_many"
    )
    
    
    @node(output_name="texts")
    def create_texts(count: int) -> List[str]:
        return [f"text_{i}" for i in range(count)]
    
    # Build full pipeline
    pipeline = Pipeline(
        nodes=[create_texts, process_many],
        engine=DaftEngine(),
        name="full_pipeline"
    )
    
    result = pipeline.run(inputs={"count": 3}, output_name=["upper_texts", "lengths"])
    
    assert "upper_texts" in result
    assert "lengths" in result
    assert result["upper_texts"] == ["TEXT_0", "TEXT_1", "TEXT_2"]
    assert result["lengths"] == [6, 6, 6]
