"""Test script with slower operations to see node updates."""
import time

from hypernodes import Pipeline, node
from hypernodes.telemetry import ProgressCallback


@node(output_name="step1")
def process_step1(text: str) -> str:
    """Simulate slow step 1."""
    time.sleep(0.5)
    return f"step1_{text}"


@node(output_name="step2")
def process_step2(step1: str) -> str:
    """Simulate slow step 2."""
    time.sleep(0.5)
    return f"step2_{step1}"


@node(output_name="step3")
def process_step3(step2: str) -> str:
    """Simulate slow step 3."""
    time.sleep(0.5)
    return f"step3_{step2}"


@node(output_name="final")
def finalize(step3: str) -> str:
    """Simulate final step."""
    time.sleep(0.5)
    return f"final_{step3}"


# Create pipeline with progress
progress = ProgressCallback(enable=True)
pipeline = Pipeline(
    nodes=[process_step1, process_step2, process_step3, finalize],
    callbacks=[progress],
    name="multi-step pipeline"
)

# Run with map to test node name display
print("Starting multi-step pipeline with 5 items...")
results = pipeline.map(
    inputs={"text": ["a", "b", "c", "d", "e"]},
    map_over="text"
)

print("\nCompleted!")
print(f"Processed {len(results['final'])} items")
