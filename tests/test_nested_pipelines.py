"""Tests for nested pipelines with config inheritance."""

from hypernodes import Pipeline, node


def test_simple_nested_pipeline():
    """Test pipeline used as a node in another pipeline.
    
    With automatic output pruning, only outputs that are used by downstream
    nodes are returned. Here, 'doubled' is an internal detail of the inner
    pipeline that is NOT used by the outer pipeline, so it's pruned.
    Only 'incremented' (used by square) and 'result' are in the final output.
    """
    
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="incremented")
    def add_one(doubled: int) -> int:
        return doubled + 1
    
    inner_pipeline = Pipeline(nodes=[double, add_one])
    
    @node(output_name="result")
    def square(incremented: int) -> int:
        return incremented ** 2
    
    outer_pipeline = Pipeline(nodes=[inner_pipeline.as_node(), square])
    
    result = outer_pipeline.run(inputs={"x": 5})
    # 'doubled' is NOT in result because it's only used internally by inner_pipeline
    # Only 'incremented' (used by square) and 'result' are returned
    assert result == {"incremented": 11, "result": 121}
    assert "doubled" not in result  # Verify optimization worked


def test_two_level_nesting():
    """Test deeper nesting (3 levels total).
    
    With automatic output pruning applied at each nesting level:
    - inner_inner produces 'doubled', but only 'doubled' is used by add_one in inner
    - inner produces 'doubled' and 'incremented', but only 'incremented' is used by square
    - So only 'incremented' and 'result' appear in final output
    """
    
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    inner_inner = Pipeline(nodes=[double])
    
    @node(output_name="incremented")
    def add_one(doubled: int) -> int:
        return doubled + 1
    
    inner = Pipeline(nodes=[inner_inner.as_node(), add_one])
    
    @node(output_name="result")
    def square(incremented: int) -> int:
        return incremented ** 2
    
    outer = Pipeline(nodes=[inner.as_node(), square])
    
    result = outer.run(inputs={"x": 5})
    # 'doubled' is pruned at the outer level because square doesn't need it
    assert result == {"incremented": 11, "result": 121}
    assert "doubled" not in result  # Verify optimization worked


def test_nested_pipeline_with_input_mapping():
    """Test nested pipeline with input parameter renaming."""
    
    @node(output_name="cleaned")
    def clean_text(passage: str) -> str:
        return passage.strip().lower()
    
    inner = Pipeline(nodes=[clean_text])
    
    # Outer pipeline uses "document" instead of "passage"
    adapted = inner.as_node(input_mapping={"document": "passage"})
    
    outer = Pipeline(nodes=[adapted])
    
    result = outer.run(inputs={"document": "  Hello World  "})
    assert result == {"cleaned": "hello world"}


def test_nested_pipeline_with_output_mapping():
    """Test nested pipeline with output renaming."""
    
    @node(output_name="result")
    def process(data: str) -> str:
        return data.upper()
    
    inner = Pipeline(nodes=[process])
    
    # Rename output from "result" to "processed_data"
    adapted = inner.as_node(output_mapping={"result": "processed_data"})
    
    outer = Pipeline(nodes=[adapted])
    
    result = outer.run(inputs={"data": "hello"})
    assert result == {"processed_data": "HELLO"}
    assert "result" not in result


def test_nested_pipeline_with_combined_mapping():
    """Test nested pipeline with both input and output mapping."""
    
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    inner = Pipeline(nodes=[double])
    
    # Rename both input and output
    adapted = inner.as_node(
        input_mapping={"value": "x"},
        output_mapping={"doubled": "result"}
    )
    
    outer = Pipeline(nodes=[adapted])
    
    result = outer.run(inputs={"value": 5})
    assert result == {"result": 10}
    assert "doubled" not in result
    assert "x" not in result


def test_nested_pipeline_with_map_over():
    """Test nested pipeline with internal mapping."""
    
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    # Inner pipeline processes ONE item
    single_process = Pipeline(nodes=[double])
    
    # Adapt to process a LIST
    batch_process = single_process.as_node(
        map_over="items",
        input_mapping={"items": "x"},
        output_mapping={"doubled": "results"}
    )
    
    outer = Pipeline(nodes=[batch_process])
    
    result = outer.run(inputs={"items": [1, 2, 3]})
    assert result == {"results": [2, 4, 6]}


def test_nested_pipeline_with_map_over_and_shared_inputs():
    """Test nested pipeline with map_over AND shared inputs (broadcast).
    
    Regression test for issue where shared inputs were dropped when using map_over.
    """
    @node(output_name="result")
    def process_item(item: str, prefix: str) -> str:
        return f"{prefix}: {item}"

    inner_pipeline = Pipeline(nodes=[process_item])

    # Adapt to process a list of items, but 'prefix' is shared
    batch_process = inner_pipeline.as_node(
        map_over="items",
        input_mapping={"items": "item"}, # map outer 'items' to inner 'item'
        # 'prefix' is not mapped, so it should be passed through as-is (shared)
        output_mapping={"result": "results"}
    )
    
    outer = Pipeline(nodes=[batch_process])
    
    result = outer.run(inputs={
        "items": ["apple", "banana"],
        "prefix": "FRUIT"
    })
    
    assert result["results"] == ["FRUIT: apple", "FRUIT: banana"]


def test_nested_map_in_outer_pipeline():
    """Test map operation on outer pipeline with nested pipeline."""
    
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="result")
    def add_one(doubled: int) -> int:
        return doubled + 1
    
    inner_pipeline = Pipeline(nodes=[double, add_one])
    
    outer_pipeline = Pipeline(nodes=[inner_pipeline.as_node()])
    
    results = outer_pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
    assert results == [
        {"doubled": 2, "result": 3},
        {"doubled": 4, "result": 5},
        {"doubled": 6, "result": 7},
    ]


def test_namespace_collision_avoidance():
    """Test that output_mapping prevents naming collisions."""
    
    @node(output_name="result")
    def process_a(input: int) -> int:
        return input * 2
    
    @node(output_name="result")
    def process_b(input: int) -> int:
        return input * 3
    
    pipeline_a = Pipeline(nodes=[process_a]).as_node(
        output_mapping={"result": "result_a"}
    )
    
    pipeline_b = Pipeline(nodes=[process_b]).as_node(
        output_mapping={"result": "result_b"}
    )
    
    @node(output_name="combined")
    def combine(result_a: int, result_b: int) -> int:
        return result_a + result_b
    
    outer = Pipeline(nodes=[pipeline_a, pipeline_b, combine])
    
    result = outer.run(inputs={"input": 5})
    assert result == {"result_a": 10, "result_b": 15, "combined": 25}
