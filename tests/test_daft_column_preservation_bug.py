"""
Minimal test to reproduce DaftEngine column preservation bug.

When DaftEngine generates code for nested map operations, it fails to preserve
columns created between map operations. This causes "column does not exist in schema" errors.
"""

import re
from hypernodes import Pipeline, node


def test_daft_code_generation_preserves_intermediate_columns():
    """
    Verify that generated Daft code preserves columns created between map operations.

    Pipeline structure:
    1. Create list A
    2. Create list B
    3. Map over A -> encode
    4. Build index from encoded A  <--- This column should be preserved
    5. Map over B -> use index     <--- But it gets dropped here!

    Expected: The groupby after map #2 should preserve 'index_a' column
    Actual: The groupby only preserves mapped columns, dropping 'index_a'
    """

    @node(output_name="items_a")
    def create_items_a() -> list[str]:
        return ["a1", "a2"]

    @node(output_name="items_b")
    def create_items_b() -> list[str]:
        return ["b1", "b2"]

    @node(output_name="encoded_item")
    def encode_item(item: str) -> str:
        return f"encoded_{item}"

    @node(output_name="index_a")
    def build_index(encoded_items_a: list[str]) -> dict:
        """Build index - this column gets dropped!"""
        return {item: i for i, item in enumerate(encoded_items_a)}

    @node(output_name="result")
    def use_index(encoded_item: str, index_a: dict) -> int:
        """Use index - will fail if index_a is dropped."""
        return index_a.get(encoded_item, -1)

    # Single-item pipelines
    encode_pipeline_a = Pipeline(nodes=[encode_item], name="encode_a")
    encode_pipeline_b = Pipeline(nodes=[encode_item], name="encode_b")
    use_index_pipeline = Pipeline(nodes=[use_index], name="use_index")

    # Map operations
    encode_a_mapped = encode_pipeline_a.as_node(
        input_mapping={"items_a": "item"},
        output_mapping={"encoded_item": "encoded_items_a"},
        map_over="items_a",
    )

    encode_b_mapped = encode_pipeline_b.as_node(
        input_mapping={"items_b": "item"},
        output_mapping={"encoded_item": "encoded_items_b"},
        map_over="items_b",
    )

    use_index_mapped = use_index_pipeline.as_node(
        input_mapping={"encoded_items_b": "encoded_item"},
        output_mapping={"result": "results"},
        map_over="encoded_items_b",
    )

    # Full pipeline
    pipeline = Pipeline(
        nodes=[
            create_items_a,
            create_items_b,
            encode_a_mapped,  # Map #1
            build_index,      # Creates index_a column
            encode_b_mapped,  # Map #2 - groupby should preserve index_a!
            use_index_mapped, # Map #3 - tries to use index_a
        ],
        name="test_pipeline",
    )

    # Generate Daft code
    code = pipeline.show_daft_code(inputs={}, output_name="results")

    print("\nGenerated Daft code:")
    print("=" * 80)
    print(code)
    print("=" * 80)

    # Verify the fix works by checking the generated code
    # After map #2 (items_b), the groupby should preserve index_a

    # Find the section after map #2 starts
    map2_marker = '# Map over: items_b'
    map2_idx = code.find(map2_marker)
    assert map2_idx != -1, "Could not find map #2 marker"

    # Extract the next 1500 characters after map #2 marker
    map2_section = code[map2_idx:map2_idx + 1500]

    print(f"\nCode section for map #2 (first 800 chars):")
    print(map2_section[:800])

    # Check if index_a is preserved in the groupby aggregation
    has_index_a = 'daft.col("index_a").any_value()' in map2_section

    print(f"\nPreserves index_a in groupby? {has_index_a}")

    # This assertion should now PASS with the fix
    assert has_index_a, (
        "BUG DETECTED: Generated Daft code does not preserve 'index_a' column "
        "in the groupby aggregation after map #2. This will cause a "
        "'column does not exist in schema' error when trying to use index_a later."
    )


def test_daft_code_generation_preserves_multiple_intermediate_columns():
    """
    Test with multiple columns created between maps (closer to real Hebrew retrieval pipeline).

    Pipeline structure:
    1. Load passages
    2. Load queries
    3. Map over passages -> encode
    4. Build vector_index from encoded passages
    5. Build bm25_index from original passages
    6. Map over queries -> encode
    7. Map over encoded queries -> retrieve using BOTH indexes

    Expected: Both vector_index and bm25_index should be preserved
    Actual: They get dropped after map #2 (queries)
    """

    @node(output_name="passages")
    def load_passages() -> list[str]:
        return ["passage1", "passage2"]

    @node(output_name="queries")
    def load_queries() -> list[str]:
        return ["query1", "query2"]

    @node(output_name="encoded_passage")
    def encode_passage(passage: str) -> str:
        return f"enc_{passage}"

    @node(output_name="encoded_query")
    def encode_query(query: str) -> str:
        return f"enc_{query}"

    @node(output_name="vector_index")
    def build_vector_index(encoded_passages: list[str]) -> dict:
        return {p: i for i, p in enumerate(encoded_passages)}

    @node(output_name="bm25_index")
    def build_bm25_index(passages: list[str]) -> dict:
        return {p: i * 10 for i, p in enumerate(passages)}

    @node(output_name="vector_score")
    def search_vector(encoded_query: str, vector_index: dict) -> float:
        return 1.0

    @node(output_name="bm25_score")
    def search_bm25(encoded_query: str, bm25_index: dict) -> float:
        return 2.0

    @node(output_name="combined")
    def combine(vector_score: float, bm25_score: float) -> float:
        return vector_score + bm25_score

    # Pipelines
    encode_passage_pipe = Pipeline(nodes=[encode_passage], name="encode_passage")
    encode_query_pipe = Pipeline(nodes=[encode_query], name="encode_query")
    search_pipe = Pipeline(
        nodes=[search_vector, search_bm25, combine],
        name="search"
    )

    # Mapped nodes
    encode_passages_mapped = encode_passage_pipe.as_node(
        input_mapping={"passages": "passage"},
        output_mapping={"encoded_passage": "encoded_passages"},
        map_over="passages",
    )

    encode_queries_mapped = encode_query_pipe.as_node(
        input_mapping={"queries": "query"},
        output_mapping={"encoded_query": "encoded_queries"},
        map_over="queries",
    )

    search_mapped = search_pipe.as_node(
        input_mapping={"encoded_queries": "encoded_query"},
        output_mapping={"combined": "results"},
        map_over="encoded_queries",
    )

    # Full pipeline
    pipeline = Pipeline(
        nodes=[
            load_passages,
            load_queries,
            encode_passages_mapped,   # Map #1
            build_vector_index,        # Creates vector_index
            build_bm25_index,          # Creates bm25_index
            encode_queries_mapped,    # Map #2 - should preserve both indexes!
            search_mapped,            # Map #3 - uses both indexes
        ],
        name="retrieval_pipeline",
    )

    # Generate code
    code = pipeline.show_daft_code(inputs={}, output_name="results")

    print("\nGenerated Daft code:")
    print("=" * 80)
    print(code)
    print("=" * 80)

    # Find the section after map #2 starts (simpler approach)
    map2_marker = '# Map over: queries'
    map2_idx = code.find(map2_marker)
    assert map2_idx != -1, "Could not find map #2 marker"

    # Extract the next 2000 characters after map #2 marker (should include the groupby)
    map2_section = code[map2_idx:map2_idx + 2000]

    print(f"\nCode section for map #2 (first 1000 chars):")
    print(map2_section[:1000])

    # Check if both indexes are preserved in the groupby aggregation
    # Look for the .any_value().alias pattern which indicates preservation
    has_vector_index = 'daft.col("vector_index").any_value()' in map2_section
    has_bm25_index = 'daft.col("bm25_index").any_value()' in map2_section

    print(f"\nPreserves vector_index in groupby? {has_vector_index}")
    print(f"Preserves bm25_index in groupby? {has_bm25_index}")

    # Both should now PASS with the fix
    assert has_vector_index, (
        "BUG DETECTED: vector_index not preserved in groupby after map #2"
    )
    assert has_bm25_index, (
        "BUG DETECTED: bm25_index not preserved in groupby after map #2"
    )


if __name__ == "__main__":
    print("=" * 80)
    print("TEST 1: Single intermediate column")
    print("=" * 80)
    try:
        test_daft_code_generation_preserves_intermediate_columns()
        print("✓ Test 1 PASSED")
    except AssertionError as e:
        print(f"✗ Test 1 FAILED: {e}")

    print("\n" + "=" * 80)
    print("TEST 2: Multiple intermediate columns (Hebrew retrieval pattern)")
    print("=" * 80)
    try:
        test_daft_code_generation_preserves_multiple_intermediate_columns()
        print("✓ Test 2 PASSED")
    except AssertionError as e:
        print(f"✗ Test 2 FAILED: {e}")
