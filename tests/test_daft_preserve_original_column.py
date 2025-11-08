"""
Daft: preserve original mapped column for downstream nodes.

Adapted from scripts/test_daft_preserve_original_column.py.
Ensures that after a mapped .as_node() operation creates a new output,
the original input list is still available for other downstream nodes.
"""

import pytest
from typing import List

try:
    import daft  # noqa: F401
    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False

from hypernodes import Pipeline, node

if DAFT_AVAILABLE:
    from hypernodes.engines import DaftEngine

pytestmark = pytest.mark.skipif(not DAFT_AVAILABLE, reason="Daft not installed")


@node(output_name="items")
def create_items(count: int) -> List[dict]:
    return [{"id": i, "value": f"item_{i}"} for i in range(count)]


@node(output_name="processed")
def process_item(item: dict) -> dict:
    return {"id": item["id"], "upper": item["value"].upper()}


@node(output_name="index")
def build_index_from_original(items: List[dict]) -> str:
    # Simulates downstream consumer that still needs original items
    return f"Index of {len(items)} items"


def test_daft_preserve_original_column_for_downstream():
    process_single = Pipeline(nodes=[process_item], name="process_single")

    # Mapped node transforms 'items' -> 'all_processed'
    process_many = process_single.as_node(
        input_mapping={"items": "item"},
        output_mapping={"processed": "all_processed"},
        map_over="items",
        name="process_many",
    )

    # Both mapped transform and downstream builder should coexist
    pipeline = Pipeline(
        nodes=[create_items, process_many, build_index_from_original],
        engine=DaftEngine(),
        name="test_preserve_original",
    )

    result = pipeline.run(inputs={"count": 3})
    assert "all_processed" in result
    assert "index" in result
    assert result["index"] == "Index of 3 items"

