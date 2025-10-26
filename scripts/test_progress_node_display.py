"""Test script to verify progress bar shows current node name."""
import time

from hypernodes import Pipeline, node
from hypernodes.telemetry import ProgressCallback


@node(output_name="encoded")
def encode_text(text: str) -> list:
    """Simulate encoding."""
    time.sleep(0.1)
    return [1, 2, 3]


@node(output_name="embedded")
def embed_vectors(encoded: list) -> list:
    """Simulate embedding."""
    time.sleep(0.1)
    return [0.1, 0.2, 0.3]


@node(output_name="scores")
def calculate_scores(embedded: list) -> float:
    """Simulate scoring."""
    time.sleep(0.1)
    return sum(embedded)


# Create pipeline with progress
progress = ProgressCallback(enable=True)
pipeline = Pipeline(
    nodes=[encode_text, embed_vectors, calculate_scores],
    callbacks=[progress],
    name="retrieval pipeline"
)

# Run with map to test node name display
print("Running retrieval pipeline with 3 examples...")
results = pipeline.map(
    inputs={"text": ["doc1", "doc2", "doc3"]},
    map_over="text"
)

print("\nResults:", results)
