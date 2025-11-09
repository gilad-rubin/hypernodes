#!/usr/bin/env python3
"""
Test Modal with nested pipelines using .as_node() - the pattern from test_modal.py

This is the missing piece! test_modal.py uses Pipeline.as_node() to create
nested pipelines, and those need to be serializable too.
"""

from pathlib import Path
from typing import List
from pydantic import BaseModel

import modal
from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine


# ==================== Modal Setup ====================
app = modal.App("hypernodes-nested-test")

hypernodes_dir = Path("/Users/giladrubin/python_workspace/hypernodes/src/hypernodes")

modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTHONUNBUFFERED": "1", "PYTHONPATH": "/root"})
    .uv_pip_install("numpy", "pydantic", "pyarrow", "daft")
    .add_local_dir(str(hypernodes_dir), remote_path="/root/hypernodes")
)


# ==================== Pydantic Models ====================
class Item(BaseModel):
    id: str
    value: int
    model_config = {"frozen": True}


class ProcessedItem(BaseModel):
    id: str
    original_value: int
    processed_value: int
    model_config = {"frozen": True}


# ==================== Single Item Nodes ====================
@node(output_name="processed_item")
def process_single_item(item: Item, multiplier: int) -> ProcessedItem:
    """Process a single item."""
    return ProcessedItem(
        id=item.id,
        original_value=item.value,
        processed_value=item.value * multiplier
    )


# ==================== Create Nested Pipeline with .as_node() ====================
# This mimics the pattern from test_modal.py where pipelines are defined at module level
# and then used with .as_node()

# Step 1: Create inner pipeline for processing single items
process_single = Pipeline(
    nodes=[process_single_item],
    name="process_single"
)

# Step 2: Convert to a node that can process lists using .as_node() with map_over
# THIS is the pattern that test_modal.py uses!
process_items_mapped = process_single.as_node(
    input_mapping={"items": "item"},  # items (list) -> item (single)
    output_mapping={"processed_item": "processed_items"},  # processed_item -> processed_items
    map_over="items",
    name="process_items_mapped"
)


# ==================== Nodes for Full Pipeline ====================
@node(output_name="items")
def create_items(count: int) -> List[Item]:
    """Create list of items."""
    return [Item(id=f"item_{i}", value=i * 10) for i in range(count)]


@node(output_name="total")
def sum_processed(processed_items: List[ProcessedItem]) -> int:
    """Sum processed values."""
    return sum(item.processed_value for item in processed_items)


# ==================== Full Pipeline ====================
# Build pipeline using the mapped nested pipeline node
# This is the EXACT pattern from test_modal.py!
pipeline = Pipeline(
    nodes=[
        create_items,
        process_items_mapped,  # <- Nested pipeline as a node!
        sum_processed
    ],
    name="full_pipeline_with_nested"
)


# ==================== Modal Function ====================
@app.function(image=modal_image, timeout=600)
def run_pipeline(pipeline: Pipeline, inputs: dict) -> dict:
    """Run pipeline with DaftEngine."""
    from hypernodes.engines import DaftEngine
    
    print("Running pipeline with nested .as_node() using DaftEngine...")
    print(f"Inputs: {list(inputs.keys())}")
    
    engine = DaftEngine(debug=True)
    pipeline = pipeline.with_engine(engine)
    
    print("Executing pipeline...")
    result = pipeline.run(inputs=inputs)
    
    print(f"Success! Result keys: {list(result.keys())}")
    return result


# ==================== Test Runner ====================
@app.local_entrypoint()
def main():
    """Run test with nested pipeline."""
    print("=" * 60)
    print("Testing: Nested Pipeline with .as_node() + Modal")
    print("=" * 60)
    
    try:
        inputs = {
            "count": 3,
            "multiplier": 5
        }
        
        print("\nCalling Modal run_pipeline.local()...")
        print(f"Pipeline structure:")
        print(f"  - create_items")
        print(f"  - process_items_mapped (nested pipeline with .as_node())")
        print(f"  - sum_processed")
        
        result = run_pipeline.local(pipeline, inputs)
        
        print(f"\n✓ Result: {result}")
        print(f"Total: {result['total']}")
        
        expected = 0 * 5 + 10 * 5 + 20 * 5  # 0 + 50 + 100 = 150
        if result['total'] == expected:
            print(f"\n✓✓✓ SUCCESS! Nested .as_node() works with Modal + DaftEngine!")
        else:
            print(f"\n✗ FAILED: Expected {expected}, got {result['total']}")
            
    except Exception as e:
        print(f"\n✗ FAILED with exception:")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

