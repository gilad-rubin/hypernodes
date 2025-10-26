"""Test nested map progress with more items to see progress bars."""
import time

from hypernodes import Pipeline, node
from hypernodes.telemetry import ProgressCallback


@node(output_name="step1")
def step_one(item: str) -> str:
    """Process step 1."""
    time.sleep(0.2)
    return f"s1_{item}"


@node(output_name="step2")
def step_two(step1: str) -> str:
    """Process step 2."""
    time.sleep(0.2)
    return f"s2_{step1}"


# Single-item pipeline with TWO nodes
process_single = Pipeline(
    nodes=[step_one, step_two],
    name="process_single"
)

# Mapped version
process_many = process_single.as_node(
    input_mapping={"items": "item"},
    output_mapping={"step2": "all_results"},
    map_over="items",
    name="process_passages",
)


@node(output_name="items")
def load_items() -> list:
    return ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]


@node(output_name="count")
def count(all_results: list) -> int:
    return len(all_results)


# Main pipeline
pipeline = Pipeline(
    nodes=[load_items, process_many, count],
    callbacks=[ProgressCallback()],
    name="main_pipeline",
)

print("=" * 70)
print("Testing nested map progress bars")
print("=" * 70)
print()
print("Expected to see:")
print("  1. main_pipeline bar showing progress through nodes")
print("  2. process_single map bars showing progress through 10 items")
print("  3. Per-node bars (step_one, step_two) showing item progress")
print()
print("Should NOT see: 'process_passages 0% 0/1'")
print("=" * 70)
print()

results = pipeline.run(inputs={})

print(f"\n✓ Processed {results['count']} items")
