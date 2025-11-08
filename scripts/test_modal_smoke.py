#!/usr/bin/env python3
"""
Quick smoke test for Modal backend.

Run this to verify Modal is working before running full tests.
"""

from hypernodes import Pipeline, node
from hypernodes.backend import ModalBackend
import modal


def test_1_simplest():
    """Absolute simplest test: single node, single input."""
    print("\n" + "="*60)
    print("Test 1: Simplest possible - single node")
    print("="*60)
    
    @node(output_name="result")
    def add_one(x: int) -> int:
        return x + 1
    
    image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("cloudpickle")
    backend = ModalBackend(image=image, timeout=60)
    
    pipeline = Pipeline(nodes=[add_one]).with_engine(backend)
    result = pipeline.run(inputs={"x": 5})
    
    print(f"Input: x=5")
    print(f"Result: {result}")
    assert result == {"result": 6}, f"Expected {{'result': 6}}, got {result}"
    print("✓ PASSED")


def test_2_map_operation():
    """Test basic map over a list."""
    print("\n" + "="*60)
    print("Test 2: Map operation")
    print("="*60)
    
    @node(output_name="squared")
    def square(x: int) -> int:
        return x ** 2
    
    image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("cloudpickle")
    backend = ModalBackend(image=image, timeout=60)
    
    pipeline = Pipeline(nodes=[square]).with_engine(backend)
    results = pipeline.map(inputs={"x": [1, 2, 3, 4]}, map_over="x")
    
    print(f"Input: x=[1, 2, 3, 4]")
    print(f"Result: {results}")
    assert results == {"squared": [1, 4, 9, 16]}, f"Expected squared values, got {results}"
    print("✓ PASSED")


def test_3_as_node_with_map():
    """Test .as_node() with map_over (Hebrew pipeline pattern)."""
    print("\n" + "="*60)
    print("Test 3: as_node with map_over")
    print("="*60)
    
    @node(output_name="processed")
    def process_item(item: int) -> int:
        return item * 10
    
    # Inner pipeline for single items
    inner = Pipeline(nodes=[process_item], name="process_single")
    
    # Wrap with map_over
    mapped_node = inner.as_node(
        input_mapping={"items": "item"},
        output_mapping={"processed": "results"},
        map_over="items",
        name="process_all"
    )
    
    @node(output_name="sum")
    def sum_results(results: list) -> int:
        return sum(results)
    
    image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("cloudpickle")
    backend = ModalBackend(image=image, timeout=60)
    
    pipeline = Pipeline(nodes=[mapped_node, sum_results]).with_engine(backend)
    result = pipeline.run(inputs={"items": [1, 2, 3, 4, 5]})
    
    print(f"Input: items=[1, 2, 3, 4, 5]")
    print(f"Result: {result}")
    assert result == {"results": [10, 20, 30, 40, 50], "sum": 150}
    print("✓ PASSED")


def test_4_pydantic_models():
    """Test with Pydantic models."""
    print("\n" + "="*60)
    print("Test 4: Pydantic models (Hebrew pipeline pattern)")
    print("="*60)
    
    from pydantic import BaseModel
    
    class Document(BaseModel):
        id: str
        text: str
        model_config = {"frozen": True}
    
    class Processed(BaseModel):
        id: str
        length: int
        model_config = {"frozen": True}
    
    @node(output_name="documents")
    def create_docs() -> list[Document]:
        return [
            Document(id="doc1", text="hello"),
            Document(id="doc2", text="world"),
            Document(id="doc3", text="test"),
        ]
    
    @node(output_name="processed")
    def process_doc(document: Document) -> Processed:
        return Processed(id=document.id, length=len(document.text))
    
    # Single-doc pipeline
    process_single = Pipeline(nodes=[process_doc], name="process_single")
    
    # Map over all docs
    process_all = process_single.as_node(
        input_mapping={"documents": "document"},
        output_mapping={"processed": "all_processed"},
        map_over="documents",
        name="process_all"
    )
    
    @node(output_name="total_length")
    def aggregate(all_processed: list[Processed]) -> int:
        return sum(p.length for p in all_processed)
    
    image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "cloudpickle",
        "pydantic>=2.0.0"
    )
    backend = ModalBackend(image=image, timeout=60)
    
    pipeline = Pipeline(
        nodes=[create_docs, process_all, aggregate]
    ).with_engine(backend)
    
    result = pipeline.run(inputs={})
    
    print(f"Created documents: doc1='hello', doc2='world', doc3='test'")
    print(f"Result: {result}")
    assert result["total_length"] == 14  # 5 + 5 + 4
    print("✓ PASSED")


def test_5_execution_config():
    """Test execution engine configuration."""
    print("\n" + "="*60)
    print("Test 5: Execution engine configuration")
    print("="*60)
    
    @node(output_name="value")
    def process(x: int) -> int:
        return x * 2
    
    image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("cloudpickle")
    
    # Test with different execution configs
    backend = ModalBackend(
        image=image,
        node_execution="sequential",
        map_execution="sequential",
        max_workers=4,
        timeout=60
    )
    
    pipeline = Pipeline(nodes=[process]).with_engine(backend)
    
    # Single run
    result = pipeline.run(inputs={"x": 21})
    print(f"Single run: x=21 -> {result}")
    assert result == {"value": 42}
    
    # Map operation (uses map_execution config)
    results = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
    print(f"Map run: x=[1, 2, 3] -> {results}")
    assert results == {"value": [2, 4, 6]}
    
    print("✓ PASSED")


if __name__ == "__main__":
    """Run smoke tests in order."""
    import sys
    
    tests = [
        test_1_simplest,
        test_2_map_operation,
        test_3_as_node_with_map,
        test_4_pydantic_models,
        test_5_execution_config,
    ]
    
    print("\n" + "="*60)
    print("MODAL BACKEND SMOKE TESTS")
    print("="*60)
    print(f"Running {len(tests)} tests...")
    
    failed = []
    for i, test in enumerate(tests, 1):
        try:
            test()
        except Exception as e:
            print(f"\n✗ FAILED: {test.__name__}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test.__name__)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Passed: {len(tests) - len(failed)}/{len(tests)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All tests passed! ✓")
        sys.exit(0)
