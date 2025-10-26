"""Test progress bar with .as_node(map_over=...) pattern."""
import time

from hypernodes import Pipeline, node
from hypernodes.telemetry import ProgressCallback


# Single-item processing nodes
@node(output_name="processed")
def process_item(item: str) -> str:
    """Process a single item."""
    time.sleep(0.2)
    return f"processed_{item}"


@node(output_name="enhanced")
def enhance_item(processed: str) -> str:
    """Enhance a processed item."""
    time.sleep(0.2)
    return f"enhanced_{processed}"


# Create single-item pipeline
process_single_item = Pipeline(
    nodes=[process_item, enhance_item],
    name="process_single_item",
)

# Wrap it as a mapped node
process_items_mapped = process_single_item.as_node(
    input_mapping={"items": "item"},  # items -> item
    output_mapping={"enhanced": "all_enhanced"},  # enhanced -> all_enhanced
    map_over="items",
    name="process_items_mapped",
)

# Build parent pipeline
@node(output_name="items")
def load_items() -> list:
    """Load items to process."""
    return ["a", "b", "c", "d", "e"]


@node(output_name="count")
def count_results(all_enhanced: list) -> int:
    """Count results."""
    return len(all_enhanced)


pipeline = Pipeline(
    nodes=[load_items, process_items_mapped, count_results],
    callbacks=[ProgressCallback()],
    name="main_pipeline",
)

print("Running pipeline with .as_node(map_over=...)...")
print("You should see:")
print("  - main_pipeline → load_items")
print("  - main_pipeline → process_items_mapped")
print("  - main_pipeline → count_results")
print()

results = pipeline.run(inputs={})

print(f"\nProcessed {results['count']} items")
print(f"Results: {results['all_enhanced'][:2]}...")
