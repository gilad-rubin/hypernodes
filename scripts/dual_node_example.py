"""Example demonstrating DualNode usage (stateless and stateful)."""

from typing import List

from hypernodes import DualNode, Pipeline

try:
    from daft import Series
except ImportError:
    # Fallback for type hints
    Series = List


# ============================================================================
# Example 1: Stateless DualNode
# ============================================================================


class Encoder:
    """Simple encoder with batch capability."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]

    def encode_batch(self, texts: Series) -> Series:
        """Batch encoding using Daft Series."""
        return Series([[0.1, 0.2, 0.3] for _ in texts.to_pylist()])


def encode_singular(text: str, encoder: Encoder) -> List[float]:
    """Singular encoding function."""
    return encoder.encode(text)


def encode_batch(texts: Series, encoder: Encoder) -> Series:
    """Batch encoding function."""
    return encoder.encode_batch(texts)


# Create stateless DualNode
encode_node = DualNode(
    output_name="encoded_text",
    singular=encode_singular,
    batch=encode_batch,
)


# ============================================================================
# Example 2: Stateful DualNode
# ============================================================================


class TextProcessor:
    """Stateful text processor with expensive initialization."""

    def __init__(self, model_name: str, prefix: str = ">>"):
        print(f"[INIT] Loading model: {model_name}")
        self.model_name = model_name
        self.prefix = prefix
        # Simulate expensive model loading
        self.model = {"name": model_name, "loaded": True}

    def process_singular(self, text: str) -> str:
        """Process single text."""
        return f"{self.prefix} {text.upper()}"

    def process_batch(self, texts: Series) -> Series:
        """Process batch of texts."""
        processed = [f"{self.prefix} {t.upper()}" for t in texts.to_pylist()]
        return Series(processed)


# Create stateful instance (lazy - __init__ not called yet)
processor = TextProcessor(model_name="text-model-v1", prefix=">>>")

# Create stateful DualNode
process_node = DualNode(
    output_name="processed_text",
    singular=processor.process_singular,
    batch=processor.process_batch,
)


# ============================================================================
# Pipeline Creation and Testing
# ============================================================================


def test_stateless_dual_node():
    """Test stateless DualNode with .run() and .map()."""
    print("\n" + "=" * 60)
    print("Testing Stateless DualNode")
    print("=" * 60)

    pipeline = Pipeline(nodes=[encode_node])

    # Visualize
    print("\nPipeline visualization:")
    pipeline.visualize()

    # Test .run() - should use singular
    print("\n[TEST] Running with .run() (singular):")
    encoder = Encoder(model_name="test-model")
    result = pipeline.run(inputs={"text": "Hello, world!", "encoder": encoder})
    print(f"Result: {result}")
    assert result["encoded_text"] == [0.1, 0.2, 0.3]

    # Test .map() - should use batch
    print("\n[TEST] Running with .map() (batch):")
    results = pipeline.map(
        inputs={
            "text": ["Hello, world!", "Hey! World", "Goodbye!"],
            "encoder": encoder,
        },
        map_over="text",
    )
    print(f"Results: {results}")
    assert len(results) == 3
    assert all(r["encoded_text"] == [0.1, 0.2, 0.3] for r in results)

    print("\n✅ Stateless DualNode tests passed!")


def test_stateful_dual_node():
    """Test stateful DualNode with .run() and .map()."""
    print("\n" + "=" * 60)
    print("Testing Stateful DualNode")
    print("=" * 60)

    pipeline = Pipeline(nodes=[process_node])

    # Visualize
    print("\nPipeline visualization:")
    pipeline.visualize()

    # Test .run() - should use singular
    print("\n[TEST] Running with .run() (singular):")
    result = pipeline.run(inputs={"text": "hello"})
    print(f"Result: {result}")
    assert result["processed_text"] == ">>> HELLO"

    # Test .map() - should use batch
    print("\n[TEST] Running with .map() (batch):")
    results = pipeline.map(
        inputs={"text": ["hello", "world", "daft"]},
        map_over="text",
    )
    print(f"Results: {results}")
    assert len(results) == 3
    assert results[0]["processed_text"] == ">>> HELLO"
    assert results[1]["processed_text"] == ">>> WORLD"
    assert results[2]["processed_text"] == ">>> DAFT"

    print("\n✅ Stateful DualNode tests passed!")


def test_combined_pipeline():
    """Test pipeline with both stateless and stateful DualNodes."""
    print("\n" + "=" * 60)
    print("Testing Combined Pipeline")
    print("=" * 60)

    # Create pipeline with both nodes
    pipeline = Pipeline(nodes=[process_node, encode_node])

    # Visualize
    print("\nPipeline visualization:")
    pipeline.visualize()

    # Test .run()
    print("\n[TEST] Running combined pipeline with .run():")
    encoder = Encoder(model_name="test-model")
    result = pipeline.run(
        inputs={"text": "hello", "encoder": encoder}
    )
    print(f"Result: {result}")
    assert result["processed_text"] == ">>> HELLO"
    # Note: encode_node expects "text" but gets "processed_text" - need to fix mapping

    print("\n✅ Combined pipeline test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("DualNode Example Script")
    print("=" * 60)

    try:
        test_stateless_dual_node()
    except Exception as e:
        print(f"\n❌ Stateless test failed (expected): {e}")

    try:
        test_stateful_dual_node()
    except Exception as e:
        print(f"\n❌ Stateful test failed (expected): {e}")

    # Combined test will need dependency mapping
    # try:
    #     test_combined_pipeline()
    # except Exception as e:
    #     print(f"\n❌ Combined test failed (expected): {e}")

    print("\n" + "=" * 60)
    print("Expected to fail - DualNode not implemented yet!")
    print("=" * 60)

