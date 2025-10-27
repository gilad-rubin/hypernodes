#!/usr/bin/env python3
"""
Minimal test case for DaftBackend with mapped pipelines.

This script tests the DaftBackend with:
1. Simple nodes
2. Pipelines with .as_node() (no map_over)
3. Pipelines with .as_node(map_over=...) - should fail gracefully
"""

from typing import List
from hypernodes import Pipeline, node
from hypernodes.daft_backend import DaftBackend
from pydantic import BaseModel


# ==================== Data Models ====================
class Item(BaseModel):
    """A simple item."""
    id: str
    value: int
    
    model_config = {"frozen": True}


class ProcessedItem(BaseModel):
    """A processed item."""
    id: str
    value: int
    doubled: int
    
    model_config = {"frozen": True}


# ==================== Simple Nodes ====================
@node(output_name="items")
def create_items(count: int) -> List[Item]:
    """Create a list of items."""
    return [Item(id=f"item_{i}", value=i) for i in range(count)]


@node(output_name="processed_item")
def process_item(item: Item) -> ProcessedItem:
    """Process a single item."""
    return ProcessedItem(
        id=item.id,
        value=item.value,
        doubled=item.value * 2
    )


@node(output_name="result")
def sum_items(processed_items: List[ProcessedItem]) -> int:
    """Sum all doubled values."""
    return sum(p.doubled for p in processed_items)


# ==================== Test 1: Simple Pipeline (No Mapping) ====================
def test_simple_pipeline():
    """Test a simple pipeline with no mapping."""
    print("\n" + "="*60)
    print("TEST 1: Simple Pipeline (No Mapping)")
    print("="*60)
    
    @node(output_name="x2")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="result")
    def add(x2: int, y: int) -> int:
        return x2 + y
    
    pipeline = Pipeline(
        nodes=[double, add],
        backend=DaftBackend(show_plan=False),
        name="simple_test"
    )
    
    inputs = {"x": 5, "y": 3}
    result = pipeline.run(inputs=inputs, output_name="result")
    
    print(f"Inputs: {inputs}")
    print(f"Result: {result}")
    assert result["result"] == 13, f"Expected 13, got {result['result']}"
    print("✅ Test passed!")


# ==================== Test 2: Pipeline with .as_node() (No map_over) ====================
def test_pipeline_as_node_no_map():
    """Test a pipeline converted to node without map_over."""
    print("\n" + "="*60)
    print("TEST 2: Pipeline as Node (No map_over)")
    print("="*60)
    
    # Inner pipeline that processes a single item
    process_single = Pipeline(
        nodes=[process_item],
        name="process_single"
    )
    
    # Convert to node WITHOUT map_over
    process_node = process_single.as_node(
        input_mapping={"item": "item"},
        output_mapping={"processed_item": "processed_item"},
        name="process_node"
    )
    
    # Outer pipeline
    pipeline = Pipeline(
        nodes=[
            create_items,
            # Note: This will fail because we need to map over items
            # but we're testing the conversion itself
        ],
        backend=DaftBackend(show_plan=False),
        name="test_as_node_no_map"
    )
    
    try:
        # This should work for the node structure itself
        print("Creating pipeline with .as_node() (no map_over)")
        print(f"Process node: {process_node}")
        print(f"Pipeline: {pipeline}")
        print("✅ Node creation successful")
    except Exception as e:
        print(f"❌ Failed: {e}")
        raise


# ==================== Test 3: Pipeline with .as_node(map_over=...) ====================
def test_pipeline_as_node_with_map():
    """Test that DaftBackend properly rejects .as_node(map_over=...)."""
    print("\n" + "="*60)
    print("TEST 3: Pipeline as Node (WITH map_over) - Should Fail")
    print("="*60)
    
    # Inner pipeline that processes a single item
    process_single = Pipeline(
        nodes=[process_item],
        name="process_single"
    )
    
    # Convert to node WITH map_over - this should fail in DaftBackend
    process_mapped = process_single.as_node(
        input_mapping={"items": "item"},
        output_mapping={"processed_item": "processed_items"},
        map_over="items",
        name="process_mapped"
    )
    
    # Outer pipeline
    pipeline = Pipeline(
        nodes=[
            create_items,
            process_mapped,  # This will fail with DaftBackend
            sum_items,
        ],
        backend=DaftBackend(show_plan=False),
        name="test_as_node_with_map"
    )
    
    inputs = {"count": 3}
    
    try:
        print("Attempting to run pipeline with map_over...")
        result = pipeline.run(inputs=inputs, output_name="result")
        print(f"❌ Should have failed but got: {result}")
        assert False, "Expected NotImplementedError but pipeline succeeded"
    except NotImplementedError as e:
        print(f"✅ Correctly raised NotImplementedError: {e}")
    except Exception as e:
        print(f"⚠️  Got different error: {type(e).__name__}: {e}")
        raise


# ==================== Test 4: Manual Daft Implementation ====================
def test_manual_daft_implementation():
    """Test how we SHOULD handle mapping in Daft - using native DataFrame operations."""
    print("\n" + "="*60)
    print("TEST 4: Manual Daft Implementation (Correct Approach)")
    print("="*60)
    
    import daft
    
    # Create data
    items_data = [
        {"id": "item_0", "value": 0},
        {"id": "item_1", "value": 1},
        {"id": "item_2", "value": 2},
    ]
    
    # Create DataFrame
    df = daft.from_pydict({
        "id": [item["id"] for item in items_data],
        "value": [item["value"] for item in items_data],
    })
    
    # Apply transformation using Daft UDF
    @daft.func
    def double_value(value: int) -> int:
        return value * 2
    
    df = df.with_column("doubled", double_value(df["value"]))
    
    # Collect results
    result = df.collect()
    print(f"Result DataFrame:\n{result}")
    print("✅ Manual Daft implementation works!")
    
    return result


# ==================== Main ====================
if __name__ == "__main__":
    print("Testing DaftBackend with various pipeline configurations...")
    print("This will help identify where the bug is.")
    
    # Test 1: Simple pipeline
    test_simple_pipeline()
    
    # Test 2: Pipeline as node (no map)
    test_pipeline_as_node_no_map()
    
    # Test 3: Pipeline as node (with map) - should fail gracefully
    test_pipeline_as_node_with_map()
    
    # Test 4: Manual Daft implementation
    test_manual_daft_implementation()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
    print("\nKey Findings:")
    print("1. DaftBackend works for simple pipelines")
    print("2. DaftBackend should reject .as_node(map_over=...)")
    print("3. For mapping, use native Daft DataFrame operations")
    print("4. Use LocalBackend for .as_node(map_over=...)")
