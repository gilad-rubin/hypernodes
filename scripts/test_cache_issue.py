#!/usr/bin/env python3
"""Test to verify caching issue with PipelineNode."""

from hypernodes import Pipeline, node, DiskCache
from pydantic import BaseModel


class Item(BaseModel):
    value: int
    model_config = {"frozen": True}


@node(output_name="doubled")
def double_item(item: Item) -> Item:
    """Double a single item."""
    print(f"  Processing item: {item.value}")
    return Item(value=item.value * 2)


# Single-item pipeline
single_pipeline = Pipeline(nodes=[double_item], name="single_item")

# Convert to mapped node
mapped_node = single_pipeline.as_node(
    input_mapping={"items": "item"},
    output_mapping={"doubled": "doubled_items"},
    map_over="items",
    name="mapped_items",
)

# Full pipeline with caching
full_pipeline = Pipeline(
    nodes=[mapped_node],
    cache=DiskCache(path=".cache/test_cache"),
    name="full_pipeline",
)

# Test data
items = [Item(value=i) for i in range(3)]

print("=" * 60)
print("FIRST RUN")
print("=" * 60)
results1 = full_pipeline.run(inputs={"items": items})
print(f"Results: {results1}")

print("\n" + "=" * 60)
print("SECOND RUN (should use cache)")
print("=" * 60)
results2 = full_pipeline.run(inputs={"items": items})
print(f"Results: {results2}")

print("\n" + "=" * 60)
print("CACHE CHECK")
print("=" * 60)
import os
if os.path.exists(".cache/test_cache"):
    print("✓ Cache directory exists")
    if os.path.exists(".cache/test_cache/meta.json"):
        import json
        with open(".cache/test_cache/meta.json") as f:
            meta = json.load(f)
        print(f"✓ Cache has {len(meta)} entries")
        print(f"  Cache keys: {list(meta.keys())[:3]}...")  # Show first 3
    else:
        print("✗ meta.json doesn't exist")
else:
    print("✗ Cache directory doesn't exist")
