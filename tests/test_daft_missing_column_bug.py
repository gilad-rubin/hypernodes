"""
Test for DaftEngine bug: columns dropped during nested map operations.

This test reproduces a bug where columns created between map operations
are not preserved in the groupby aggregation, causing "column does not exist" errors.
"""

from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine


@node(output_name="items")
def create_items() -> list[str]:
    """Create a list of items."""
    return ["a", "b", "c"]


@node(output_name="more_items")
def create_more_items() -> list[str]:
    """Create another list of items."""
    return ["x", "y"]


@node(output_name="index")
def build_index(items: list[str]) -> dict:
    """Build an index from items."""
    return {item: i for i, item in enumerate(items)}


@node(output_name="processed_item")
def process_item(item: str) -> str:
    """Process a single item."""
    return item.upper()


@node(output_name="lookup_result")
def lookup_in_index(processed_item: str, index: dict) -> int:
    """Lookup processed item in index.

    This will fail if index column is dropped during map operation.
    """
    return index.get(processed_item.lower(), -1)


def test_daft_preserves_columns_between_maps():
    """
    Test that DaftEngine preserves columns created between map operations.

    Pipeline structure:
    1. Create items list
    2. Create more_items list
    3. Build index from items (stateful object)
    4. Map over more_items -> process each -> lookup in index

    The bug: When generating code for the map operation, the groupby doesn't
    preserve the 'index' column that was created before the map started.
    """
    # Single-item pipeline
    single_pipeline = Pipeline(
        nodes=[process_item, lookup_in_index],
        name="process_single",
    )

    # Mapped version
    mapped_node = single_pipeline.as_node(
        input_mapping={"more_items": "item"},
        output_mapping={"lookup_result": "results"},
        map_over="more_items",
        name="process_mapped",
    )

    # Full pipeline
    full_pipeline = Pipeline(
        nodes=[
            create_items,
            create_more_items,
            build_index,  # Creates 'index' column
            mapped_node,   # Maps over more_items - should preserve 'index'!
        ],
        name="full_pipeline",
    )

    # Generate Daft code
    code = full_pipeline.show_daft_code(inputs={}, output_name="results")

    print("Generated Daft code:")
    print("=" * 60)
    print(code)
    print("=" * 60)

    # Try to run the pipeline with DaftEngine
    full_pipeline = full_pipeline.with_engine(DaftEngine(code_generation_mode=False))

    # This should work but will fail with "index does not exist in schema"
    result = full_pipeline.run(inputs={}, output_name="results")

    # Verify results
    assert "results" in result
    assert result["results"] == [23, 24, -1]  # x=23, y=24, z not in original items


def test_daft_preserves_columns_between_multiple_maps():
    """
    Test with multiple map operations to ensure columns propagate correctly.

    This is the pattern from the Hebrew retrieval pipeline:
    1. Map over passages -> encode -> build vector_index
    2. Build bm25_index from passages
    3. Map over queries -> encode
    4. Map over encoded_queries -> retrieve from BOTH indexes

    The bug: After the second map (queries), the bm25_index is dropped.
    """

    @node(output_name="passages")
    def load_passages() -> list[str]:
        return ["passage1", "passage2"]

    @node(output_name="queries")
    def load_queries() -> list[str]:
        return ["query1", "query2"]

    @node(output_name="encoded_passage")
    def encode_passage(passage: str) -> str:
        return f"encoded_{passage}"

    @node(output_name="vector_index")
    def build_vector_index(encoded_passages: list[str]) -> dict:
        return {p: i for i, p in enumerate(encoded_passages)}

    @node(output_name="bm25_index")
    def build_bm25_index(passages: list[str]) -> dict:
        """Build BM25 index - created AFTER first map."""
        return {p: i * 10 for i, p in enumerate(passages)}

    @node(output_name="encoded_query")
    def encode_query(query: str) -> str:
        return f"encoded_{query}"

    @node(output_name="vector_score")
    def search_vector(encoded_query: str, vector_index: dict) -> float:
        return float(len(encoded_query))

    @node(output_name="bm25_score")
    def search_bm25(query: str, bm25_index: dict) -> float:
        """This will fail if bm25_index is dropped!"""
        return float(len(query) * 10)

    @node(output_name="combined_score")
    def combine_scores(vector_score: float, bm25_score: float) -> float:
        return vector_score + bm25_score

    # Build single-item pipelines
    encode_passage_pipeline = Pipeline(nodes=[encode_passage], name="encode_passage")
    encode_query_pipeline = Pipeline(nodes=[encode_query], name="encode_query")
    search_pipeline = Pipeline(
        nodes=[search_vector, search_bm25, combine_scores],
        name="search",
    )

    # Create mapped nodes
    encode_passages_mapped = encode_passage_pipeline.as_node(
        input_mapping={"passages": "passage"},
        output_mapping={"encoded_passage": "encoded_passages"},
        map_over="passages",
    )

    encode_queries_mapped = encode_query_pipeline.as_node(
        input_mapping={"queries": "query"},
        output_mapping={"encoded_query": "encoded_queries"},
        map_over="queries",
    )

    search_mapped = search_pipeline.as_node(
        input_mapping={"encoded_queries": "encoded_query"},
        output_mapping={"combined_score": "scores"},
        map_over="encoded_queries",
    )

    # Full pipeline
    pipeline = Pipeline(
        nodes=[
            load_passages,
            load_queries,
            encode_passages_mapped,  # Map 1: passages
            build_vector_index,       # Uses encoded_passages
            build_bm25_index,         # Uses original passages - THIS COLUMN GETS LOST
            encode_queries_mapped,   # Map 2: queries - groupby drops bm25_index!
            search_mapped,           # Map 3: tries to use bm25_index -> ERROR
        ],
        name="retrieval_pipeline",
    )

    # Generate code to inspect
    code = pipeline.show_daft_code(inputs={}, output_name="scores")

    print("\nGenerated Daft code for multi-map pipeline:")
    print("=" * 60)
    print(code)
    print("=" * 60)

    # Try to run - this will fail with "bm25_index does not exist in schema"
    pipeline = pipeline.with_engine(DaftEngine(code_generation_mode=False))
    result = pipeline.run(inputs={}, output_name="scores")

    assert "scores" in result


if __name__ == "__main__":
    # Run the simpler test first
    print("Running test_daft_preserves_columns_between_maps...")
    try:
        test_daft_preserves_columns_between_maps()
        print("✓ Test passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")

    print("\n" + "=" * 60 + "\n")

    # Run the more complex test
    print("Running test_daft_preserves_columns_between_multiple_maps...")
    try:
        test_daft_preserves_columns_between_multiple_maps()
        print("✓ Test passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
