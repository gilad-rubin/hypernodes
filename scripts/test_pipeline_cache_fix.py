#!/usr/bin/env python3
"""Test that PipelineNode caching works properly after the fix."""

from hypernodes import Pipeline, node, DiskCache
from pydantic import BaseModel


class Item(BaseModel):
    value: int
    model_config = {"frozen": True}


call_count = {"encode": 0, "process": 0}


@node(output_name="encoded")
def encode_item(item: Item) -> Item:
    """Encode a single item."""
    call_count["encode"] += 1
    print(f"  [COMPUTE] Encoding item: {item.value}")
    return Item(value=item.value * 2)


@node(output_name="processed")
def process_item(encoded: Item) -> Item:
    """Process an encoded item."""
    call_count["process"] += 1
    print(f"  [COMPUTE] Processing item: {encoded.value}")
    return Item(value=encoded.value + 1)


# Single-item pipeline
single_pipeline = Pipeline(
    nodes=[encode_item, process_item],
    name="single_item",
)

# Convert to mapped node
mapped_node = single_pipeline.as_node(
    input_mapping={"items": "item"},
    output_mapping={"processed": "processed_items"},
    map_over="items",
    name="mapped_items",
)

# Full pipeline with caching
full_pipeline = Pipeline(
    nodes=[mapped_node],
    cache=DiskCache(path=".cache/test_pipeline_cache_fix"),
    name="full_pipeline",
)

# Test data
items = [Item(value=i) for i in [1, 2, 3]]

print("=" * 70)
print("FIRST RUN - should compute all items")
print("=" * 70)
call_count = {"encode": 0, "process": 0}
results1 = full_pipeline.run(inputs={"items": items})
print(f"\nResults: {results1}")
print(f"Call counts: encode={call_count['encode']}, process={call_count['process']}")

print("\n" + "=" * 70)
print("SECOND RUN - should use cache (NO compute messages)")
print("=" * 70)
call_count = {"encode": 0, "process": 0}
results2 = full_pipeline.run(inputs={"items": items})
print(f"\nResults: {results2}")
print(f"Call counts: encode={call_count['encode']}, process={call_count['process']}")
print(f"\n✓ SUCCESS: Cache is working!" if call_count['encode'] == 0 else f"\n✗ FAIL: Still computing")

print("\n" + "=" * 70)
print("THIRD RUN WITH ONE NEW ITEM - should compute only new item")
print("=" * 70)
call_count = {"encode": 0, "process": 0}
items_new = [Item(value=i) for i in [1, 2, 4]]  # 1,2 cached, 4 is new
results3 = full_pipeline.run(inputs={"items": items_new})
print(f"\nResults: {results3}")
print(f"Call counts: encode={call_count['encode']}, process={call_count['process']}")
expected_calls = 2  # encode_item and process_item for item 4
print(f"\n✓ SUCCESS: Partial cache working!" if call_count['encode'] == 1 and call_count['process'] == 1 else f"\n✗ FAIL: Expected 1 encode + 1 process, got {call_count['encode']} encode + {call_count['process']} process")
