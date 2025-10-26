"""Test script for hierarchy display and per-node map progress bars."""
import time
from hypernodes import node, Pipeline
from hypernodes.telemetry import ProgressCallback
import logfire

# Configure logfire without console output
logfire.configure(send_to_logfire=False, console=False)

print("=" * 60)
print("TEST 1: Nested Pipeline Hierarchy Display")
print("=" * 60)
print("Testing that nested pipelines show clear hierarchy\n")

# Define inner pipeline nodes
@node(output_name="cleaned")
def clean_data(data: str) -> str:
    """Clean the raw data."""
    time.sleep(0.2)
    return data.strip()


@node(output_name="normalized")
def normalize(cleaned: str) -> str:
    """Normalize the data."""
    time.sleep(0.3)
    return cleaned.lower()


# Create inner pipeline
inner_pipeline = Pipeline(nodes=[clean_data, normalize], name="preprocessing")


# Define outer pipeline nodes
@node(output_name="data")
def load_data(source: str) -> str:
    """Load data from source."""
    time.sleep(0.1)
    return f"  DATA FROM {source}  "


@node(output_name="final_result")
def aggregate(normalized: str) -> dict:
    """Aggregate the results."""
    time.sleep(0.2)
    return {"processed": normalized, "length": len(normalized)}


# Create outer pipeline
outer_pipeline = Pipeline(
    nodes=[load_data, inner_pipeline, aggregate],
    callbacks=[ProgressCallback()],
    id="main_pipeline",
)

result = outer_pipeline.run(inputs={"source": "database"})
print(f"\n✓ Nested pipeline test completed!")
print(f"Result: {result}\n")

print("=" * 60)
print("TEST 2: Map Operation with Per-Node Progress Bars")
print("=" * 60)
print("Testing that map shows one progress bar per node\n")

# Define simple pipeline for mapping
@node(output_name="squared")
def square(x: int) -> int:
    """Square the input."""
    time.sleep(0.1)
    return x * 2


@node(output_name="is_even")
def check_even(squared: int) -> bool:
    """Check if squared is even."""
    time.sleep(0.05)
    return squared % 2 == 0


# Create map pipeline
map_pipeline = Pipeline(
    nodes=[square, check_even],
    callbacks=[ProgressCallback()],
    id="map_test_pipeline",
)

# Run map operation
numbers = list(range(10))
map_result = map_pipeline.map(
    inputs={"x": numbers},
    map_over=["x"],
    map_mode="zip"
)

print(f"\n✓ Map operation test completed!")
print(f"Processed {len(numbers)} items")
print(f"Results: squared={map_result['squared'][:3]}..., is_even={map_result['is_even'][:3]}...\n")

print("=" * 60)
print("All tests completed!")
print("=" * 60)
print("\nExpected behavior:")
print("1. Nested pipelines should show clear top-to-bottom hierarchy")
print("2. Map operations should show one progress bar per node (not per item)")
print("3. Each map bar should go from 0 to N (number of items)")
