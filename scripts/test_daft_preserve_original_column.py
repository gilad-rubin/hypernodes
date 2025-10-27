#!/usr/bin/env python3
"""
Test that DaftBackend preserves the original mapped column for downstream nodes.

This reproduces the issue where build_bm25_index needs the original `passages` 
list, even after encode_passages_mapped has transformed it to `encoded_passages`.
"""

from typing import List
from hypernodes import Pipeline, node
from hypernodes.daft_backend import DaftBackend


@node(output_name="items")
def create_items(count: int) -> List[dict]:
    """Create a list of items."""
    return [{"id": i, "value": f"item_{i}"} for i in range(count)]


@node(output_name="processed")
def process_item(item: dict) -> dict:
    """Process a single item."""
    return {"id": item["id"], "upper": item["value"].upper()}


@node(output_name="index")
def build_index_from_original(items: List[dict]) -> str:
    """Build an index from the ORIGINAL items list.
    
    This simulates build_bm25_index which needs the original passages,
    not the encoded ones.
    """
    return f"Index of {len(items)} items"


# Single-item pipeline
process_single = Pipeline(nodes=[process_item], name="process_single")

# Mapped node
process_many = process_single.as_node(
    input_mapping={"items": "item"},
    output_mapping={"processed": "all_processed"},
    map_over="items",
    name="process_many"
)

# Full pipeline - both process_many and build_index_from_original need 'items'
pipeline = Pipeline(
    nodes=[
        create_items,
        process_many,  # Transforms items -> all_processed
        build_index_from_original,  # Still needs original 'items'!
    ],
    backend=DaftBackend(),
    name="test_preserve_original"
)

print("Testing that original mapped column is preserved...")
print("=" * 70)

try:
    result = pipeline.run(inputs={"count": 3})
    print(f"✅ Success! Result: {result}")
    print(f"   - all_processed: {result.get('all_processed')}")
    print(f"   - index: {result.get('index')}")
    
    assert "all_processed" in result
    assert "index" in result
    assert result["index"] == "Index of 3 items"
    
    print("\n✅ All assertions passed!")
    print("The original 'items' column was preserved for build_index_from_original")
    
except ValueError as e:
    print(f"❌ Failed with ValueError: {e}")
    print("\nThis means the original 'items' column was not preserved")
    print("after the map operation.")
    raise
