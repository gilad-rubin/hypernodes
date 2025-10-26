"""Test progress with retrieval-like pipeline structure."""
import time

from hypernodes import Pipeline, node
from hypernodes.telemetry import ProgressCallback


# Simulate single-item operations
@node(output_name="encoded_passage")
def encode_passage(passage: str, encoder: str) -> str:
    """Encode single passage."""
    time.sleep(0.2)
    return f"encoded[{passage}]"


@node(output_name="encoded_query")
def encode_query(query: str, encoder: str) -> str:
    """Encode single query."""
    time.sleep(0.2)
    return f"encoded[{query}]"


@node(output_name="hits")
def retrieve(encoded_query: str, index: str, top_k: int) -> list:
    """Retrieve hits."""
    time.sleep(0.2)
    return ["hit1", "hit2", "hit3"]


# Create single-item pipelines
encode_single_passage = Pipeline(
    nodes=[encode_passage],
    name="encode_single_passage",
)

encode_single_query = Pipeline(
    nodes=[encode_query],
    name="encode_single_query",
)

retrieve_single_query = Pipeline(
    nodes=[retrieve],
    name="retrieve_single_query",
)

# Create mapped versions
encode_passages_mapped = encode_single_passage.as_node(
    input_mapping={"passages": "passage"},
    output_mapping={"encoded_passage": "encoded_passages"},
    map_over="passages",
    name="encode_passages_mapped",
)

encode_queries_mapped = encode_single_query.as_node(
    input_mapping={"queries": "query"},
    output_mapping={"encoded_query": "encoded_queries"},
    map_over="queries",
    name="encode_queries_mapped",
)

retrieve_queries_mapped = retrieve_single_query.as_node(
    input_mapping={"encoded_queries": "encoded_query"},
    output_mapping={"hits": "all_hits"},
    map_over="encoded_queries",
    name="retrieve_queries_mapped",
)


# Other nodes
@node(output_name="passages")
def load_passages() -> list:
    """Load passages."""
    return ["passage1", "passage2", "passage3"]


@node(output_name="queries")
def load_queries() -> list:
    """Load queries."""
    return ["query1", "query2"]


@node(output_name="encoder")
def create_encoder() -> str:
    """Create encoder."""
    return "encoder_model"


@node(output_name="index")
def build_index(encoded_passages: list) -> str:
    """Build index."""
    return "vector_index"


@node(output_name="result_count")
def count_results(all_hits: list) -> int:
    """Count results."""
    return len(all_hits)


# Build main pipeline
pipeline = Pipeline(
    nodes=[
        # Load data
        load_passages,
        load_queries,
        # Setup
        create_encoder,
        # Encode passages (mapped)
        encode_passages_mapped,
        # Build index
        build_index,
        # Encode queries (mapped)
        encode_queries_mapped,
        # Retrieve (mapped)
        retrieve_queries_mapped,
        # Count
        count_results,
    ],
    callbacks=[ProgressCallback()],
    name="retrieval_pipeline",
)

print("=" * 60)
print("Running retrieval-like pipeline...")
print("=" * 60)
print()
print("Expected progress updates:")
print("  retrieval_pipeline → load_passages")
print("  retrieval_pipeline → load_queries")
print("  retrieval_pipeline → create_encoder")
print("  retrieval_pipeline → encode_passages_mapped")
print("  retrieval_pipeline → build_index")
print("  retrieval_pipeline → encode_queries_mapped")
print("  retrieval_pipeline → retrieve_queries_mapped")
print("  retrieval_pipeline → count_results")
print()
print("=" * 60)

results = pipeline.run(inputs={"top_k": 5})

print("\n" + "=" * 60)
print(f"Processed {results['result_count']} queries")
print(f"Encoded {len(results['encoded_passages'])} passages")
print(f"Encoded {len(results['encoded_queries'])} queries")
print("=" * 60)
