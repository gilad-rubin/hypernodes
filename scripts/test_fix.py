import time
from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine

@node(output_name="encoded")
def encode_text(text: str) -> list:
    """Simulate text encoding."""
    time.sleep(0.1)
    return [1, 2, 3]

@node(output_name="embedded")
def embed_vectors(encoded: list) -> list:
    """Simulate vector embedding."""
    time.sleep(0.1)
    return [0.1, 0.2, 0.3]

@node(output_name="normalized")
def normalize(embedded: list) -> list:
    """Simulate normalization."""
    time.sleep(0.1)
    total = sum(embedded)
    return [x / total for x in embedded]

@node(output_name="scores")
def calculate_scores(normalized: list) -> float:
    """Simulate scoring."""
    time.sleep(0.1)
    return sum(normalized)

# Create pipeline
retrieval_pipeline = Pipeline(
    nodes=[encode_text, embed_vectors, normalize, calculate_scores],
    name="retrieval pipeline",
    engine=DaftEngine(),
)

print("Testing map with DaftEngine...")
results = retrieval_pipeline.map(
    inputs={"text": ["doc1", "doc2", "doc3", "doc4", "doc5"]}, 
    map_over="text"
)

print(f"\nProcessed {len(results)} documents")
print(f"Results: {results}")
print("\n✅ Test passed!")

