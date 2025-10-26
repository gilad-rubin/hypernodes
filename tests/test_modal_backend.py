"""Tests for ModalBackend execution.

Starts with minimal examples and gradually increases complexity.
Tests the PipelineExecutionEngine abstraction and Modal integration.
"""

import pytest
from hypernodes import Pipeline, node
from hypernodes.backend import ModalBackend


# ==================== Fixtures ====================
@pytest.fixture
def modal_image():
    """Create a minimal Modal image for testing."""
    try:
        import modal
    except ImportError:
        pytest.skip("Modal not installed")
    
    # Minimal image with just cloudpickle
    return modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "cloudpickle>=3.0.0"
    )


@pytest.fixture
def modal_backend(modal_image):
    """Create a basic ModalBackend for testing."""
    return ModalBackend(image=modal_image, timeout=60)


# ==================== Test 1: Simplest Possible - Single Node ====================
def test_modal_single_node_simple(modal_backend):
    """Test the absolute simplest case: single node, single input."""
    @node(output_name="result")
    def add_one(x: int) -> int:
        return x + 1
    
    pipeline = Pipeline(nodes=[add_one]).with_backend(modal_backend)
    
    result = pipeline.run(inputs={"x": 5})
    
    assert result == {"result": 6}
    print("✓ Test 1 passed: Single node execution on Modal")


# ==================== Test 2: Multiple Nodes with Dependencies ====================
def test_modal_multiple_nodes(modal_backend):
    """Test multiple dependent nodes."""
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="result")
    def add_ten(doubled: int) -> int:
        return doubled + 10
    
    pipeline = Pipeline(nodes=[double, add_ten]).with_backend(modal_backend)
    
    result = pipeline.run(inputs={"x": 5})
    
    assert result == {"doubled": 10, "result": 20}
    print("✓ Test 2 passed: Multiple dependent nodes")


# ==================== Test 3: Simple Map Operation ====================
def test_modal_simple_map(modal_backend):
    """Test basic map operation with small list."""
    @node(output_name="squared")
    def square(x: int) -> int:
        return x ** 2
    
    pipeline = Pipeline(nodes=[square]).with_backend(modal_backend)
    
    # Map over small list
    results = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
    
    assert results == {"squared": [1, 4, 9]}
    print("✓ Test 3 passed: Simple map operation")


# ==================== Test 4: Nested Pipeline with as_node ====================
def test_modal_nested_pipeline(modal_backend):
    """Test nested pipeline using .as_node()."""
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    # Inner pipeline
    inner = Pipeline(nodes=[double], name="inner")
    
    # Wrap as node
    inner_node = inner.as_node()
    
    @node(output_name="result")
    def add_five(doubled: int) -> int:
        return doubled + 5
    
    # Outer pipeline
    outer = Pipeline(nodes=[inner_node, add_five]).with_backend(modal_backend)
    
    result = outer.run(inputs={"x": 10})
    
    assert result == {"doubled": 20, "result": 25}
    print("✓ Test 4 passed: Nested pipeline")


# ==================== Test 5: as_node with map_over ====================
def test_modal_as_node_with_map(modal_backend):
    """Test .as_node() with map_over (key feature for Hebrew pipeline)."""
    @node(output_name="processed")
    def process_item(item: int) -> int:
        return item * 3
    
    # Inner pipeline that processes single items
    inner = Pipeline(nodes=[process_item], name="process_single")
    
    # Wrap with map_over to process lists
    mapped_node = inner.as_node(
        input_mapping={"items": "item"},
        output_mapping={"processed": "results"},
        map_over="items",
        name="process_all"
    )
    
    @node(output_name="sum")
    def sum_results(results: list) -> int:
        return sum(results)
    
    # Outer pipeline
    pipeline = Pipeline(nodes=[mapped_node, sum_results]).with_backend(modal_backend)
    
    result = pipeline.run(inputs={"items": [1, 2, 3, 4]})
    
    assert result == {"results": [3, 6, 9, 12], "sum": 30}
    print("✓ Test 5 passed: as_node with map_over")


# ==================== Test 6: Execution Engine Config ====================
def test_modal_execution_engine_config(modal_image):
    """Test that execution engine config is passed through correctly."""
    # Create backend with specific execution modes
    backend = ModalBackend(
        image=modal_image,
        node_execution="sequential",
        map_execution="sequential",
        max_workers=4,
        timeout=60
    )
    
    @node(output_name="y")
    def inc(x: int) -> int:
        return x + 1
    
    pipeline = Pipeline(nodes=[inc]).with_backend(backend)
    
    result = pipeline.run(inputs={"x": 41})
    
    assert result == {"y": 42}
    print("✓ Test 6 passed: Execution engine configuration")


# ==================== Test 7: Pydantic Models (Hebrew Pipeline Pattern) ====================
def test_modal_with_pydantic(modal_backend):
    """Test with Pydantic models like the Hebrew pipeline uses."""
    from pydantic import BaseModel
    
    class Item(BaseModel):
        id: str
        value: int
        
        model_config = {"frozen": True}
    
    class ProcessedItem(BaseModel):
        id: str
        value: int
        score: float
        
        model_config = {"frozen": True}
    
    @node(output_name="items")
    def create_items() -> list[Item]:
        return [
            Item(id="a", value=10),
            Item(id="b", value=20),
        ]
    
    @node(output_name="processed")
    def process_item(item: Item) -> ProcessedItem:
        return ProcessedItem(id=item.id, value=item.value, score=item.value * 1.5)
    
    # Single item pipeline
    process_single = Pipeline(nodes=[process_item], name="process_single")
    
    # Map over items
    process_mapped = process_single.as_node(
        input_mapping={"items": "item"},
        output_mapping={"processed": "all_processed"},
        map_over="items",
        name="process_all"
    )
    
    pipeline = Pipeline(
        nodes=[create_items, process_mapped]
    ).with_backend(modal_backend)
    
    result = pipeline.run(inputs={})
    
    assert len(result["all_processed"]) == 2
    assert result["all_processed"][0].id == "a"
    assert result["all_processed"][0].score == 15.0
    assert result["all_processed"][1].id == "b"
    assert result["all_processed"][1].score == 30.0
    print("✓ Test 7 passed: Pydantic models")


# ==================== Test 8: Mini Hebrew-Style Pipeline ====================
def test_modal_mini_hebrew_pipeline(modal_backend):
    """Minimal version of the Hebrew retrieval pattern."""
    from pydantic import BaseModel
    from typing import Any
    
    class Document(BaseModel):
        id: str
        text: str
        model_config = {"frozen": True}
    
    class EncodedDoc(BaseModel):
        id: str
        embedding: Any  # Would be numpy array
        model_config = {"frozen": True, "arbitrary_types_allowed": True}
    
    # Load documents
    @node(output_name="documents")
    def load_docs() -> list[Document]:
        return [
            Document(id="doc1", text="hello"),
            Document(id="doc2", text="world"),
        ]
    
    # Encode single document
    @node(output_name="encoded")
    def encode_doc(document: Document) -> EncodedDoc:
        # Mock encoding: just store text length as "embedding"
        return EncodedDoc(id=document.id, embedding=len(document.text))
    
    # Single-doc pipeline
    encode_single = Pipeline(nodes=[encode_doc], name="encode_single")
    
    # Map over all docs
    encode_all = encode_single.as_node(
        input_mapping={"documents": "document"},
        output_mapping={"encoded": "all_encoded"},
        map_over="documents",
        name="encode_all"
    )
    
    # Sum embeddings (mock aggregation)
    @node(output_name="total")
    def aggregate(all_encoded: list[EncodedDoc]) -> int:
        return sum(doc.embedding for doc in all_encoded)
    
    pipeline = Pipeline(
        nodes=[load_docs, encode_all, aggregate]
    ).with_backend(modal_backend)
    
    result = pipeline.run(inputs={})
    
    assert result["total"] == 10  # len("hello") + len("world") = 5 + 5 = 10
    print("✓ Test 8 passed: Mini Hebrew-style pipeline")


# ==================== Test 9: Error Handling ====================
def test_modal_error_handling(modal_backend):
    """Test that errors in remote execution are properly propagated."""
    @node(output_name="result")
    def failing_node(x: int) -> int:
        if x == 42:
            raise ValueError("Cannot process 42!")
        return x + 1
    
    pipeline = Pipeline(nodes=[failing_node]).with_backend(modal_backend)
    
    # Should work
    result = pipeline.run(inputs={"x": 5})
    assert result == {"result": 6}
    
    # Should fail and propagate error
    with pytest.raises(Exception):  # Will be wrapped in Modal's exception
        pipeline.run(inputs={"x": 42})
    
    print("✓ Test 9 passed: Error handling")


# ==================== Test 10: Larger Map with Execution Config ====================
def test_modal_larger_map_with_config(modal_image):
    """Test larger map operation with threaded execution."""
    backend = ModalBackend(
        image=modal_image,
        map_execution="threaded",  # Use threaded for faster execution
        max_workers=4,
        timeout=120
    )
    
    @node(output_name="processed")
    def heavy_process(x: int) -> int:
        # Simulate some work
        result = x
        for _ in range(100):
            result = (result * 2) % 1000
        return result
    
    pipeline = Pipeline(nodes=[heavy_process]).with_backend(backend)
    
    # Map over 20 items
    items = list(range(20))
    results = pipeline.map(inputs={"x": items}, map_over="x")
    
    assert len(results["processed"]) == 20
    assert all(isinstance(r, int) for r in results["processed"])
    print("✓ Test 10 passed: Larger map with threaded execution")


# ==================== Integration Test ====================
@pytest.mark.integration
def test_modal_full_integration(modal_image):
    """Full integration test combining all features."""
    from pydantic import BaseModel
    
    class DataPoint(BaseModel):
        id: int
        value: float
        model_config = {"frozen": True}
    
    # Setup
    @node(output_name="multiplier")
    def get_multiplier() -> float:
        return 2.5
    
    @node(output_name="data")
    def generate_data() -> list[DataPoint]:
        return [DataPoint(id=i, value=float(i * 10)) for i in range(1, 6)]
    
    # Process single item
    @node(output_name="scaled")
    def scale_value(item: DataPoint, multiplier: float) -> DataPoint:
        return DataPoint(id=item.id, value=item.value * multiplier)
    
    # Single-item pipeline
    scale_single = Pipeline(nodes=[scale_value], name="scale_single")
    
    # Map over all
    scale_all = scale_single.as_node(
        input_mapping={"data": "item"},
        output_mapping={"scaled": "all_scaled"},
        map_over="data",
        name="scale_all"
    )
    
    # Aggregate
    @node(output_name="average")
    def compute_average(all_scaled: list[DataPoint]) -> float:
        return sum(d.value for d in all_scaled) / len(all_scaled)
    
    # Full pipeline with execution config
    backend = ModalBackend(
        image=modal_image,
        node_execution="sequential",
        map_execution="threaded",
        max_workers=4,
        timeout=120
    )
    
    pipeline = Pipeline(
        nodes=[get_multiplier, generate_data, scale_all, compute_average],
        name="integration_test"
    ).with_backend(backend)
    
    result = pipeline.run(inputs={})
    
    # Expected: (10*2.5 + 20*2.5 + 30*2.5 + 40*2.5 + 50*2.5) / 5 = 75.0
    assert abs(result["average"] - 75.0) < 0.001
    print("✓ Integration test passed: Full pipeline with all features")


if __name__ == "__main__":
    """Run tests with detailed output."""
    pytest.main([__file__, "-v", "-s"])
