"""Reproduce the nested map progress bar issue."""
import time

from hypernodes import Pipeline, node
from hypernodes.telemetry import ProgressCallback


@node(output_name="processed")
def process_item(item: str) -> str:
    """Process single item."""
    time.sleep(0.1)
    return f"proc_{item}"


# Single-item pipeline
process_single = Pipeline(nodes=[process_item], name="process_single")

# Mapped version
process_many = process_single.as_node(
    input_mapping={"items": "item"},
    output_mapping={"processed": "all_processed"},
    map_over="items",
    name="encode_passages_mapped",  # Same name as user's
)


@node(output_name="items")
def load_items() -> list:
    return ["a", "b", "c", "d", "e"]


@node(output_name="count")
def count(all_processed: list) -> int:
    return len(all_processed)


# Main pipeline
pipeline = Pipeline(
    nodes=[load_items, process_many, count],
    callbacks=[ProgressCallback()],
    name="hebrew_retrieval",  # Same name as user's
)

print("Running - watch for duplicate progress bars...")
print("ISSUE: Should NOT see 'encode_passages_mapped 0% 0/1'")
print()

results = pipeline.run(inputs={})
print(f"\nProcessed: {results['count']}")
