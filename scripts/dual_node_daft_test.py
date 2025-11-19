"""Test DualNode with DaftEngine to verify batch execution."""

from typing import List

from hypernodes import DualNode, Pipeline

try:
    from daft import Series

    from hypernodes.engines import DaftEngine

    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False
    print("⚠️  Daft not available - skipping DaftEngine tests")
    exit(0)


# ============================================================================
# Instrumented Encoder to Track Calls
# ============================================================================


class InstrumentedEncoder:
    """Encoder that tracks whether singular or batch was called."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.singular_calls = 0
        self.batch_calls = 0

    def encode(self, text: str) -> List[float]:
        self.singular_calls += 1
        print(f"  [SINGULAR] encode() called (total: {self.singular_calls})")
        return [0.1, 0.2, 0.3]

    def encode_batch(self, texts: Series) -> Series:
        self.batch_calls += 1
        batch_size = len(texts)
        print(
            f"  [BATCH] encode_batch() called with {batch_size} items (total batches: {self.batch_calls})"
        )
        # Return Series.from_pylist for proper construction
        return Series.from_pylist([[0.1, 0.2, 0.3] for _ in texts.to_pylist()])


def encode_singular(text: str, encoder: InstrumentedEncoder) -> List[float]:
    """Singular encoding function."""
    return encoder.encode(text)


def encode_batch(texts: Series, encoder: InstrumentedEncoder) -> Series:
    """Batch encoding function."""
    result = encoder.encode_batch(texts)
    return result


# Create DualNode
encode_node = DualNode(
    output_name="encoded_text",
    singular=encode_singular,
    batch=encode_batch,
)


# ============================================================================
# Test with SeqEngine (should use singular)
# ============================================================================


def test_sequential_engine():
    """Test that SeqEngine uses singular function."""
    print("\n" + "=" * 60)
    print("Testing SeqEngine (should use SINGULAR)")
    print("=" * 60)

    from hypernodes.engines import SeqEngine

    pipeline = Pipeline(nodes=[encode_node], engine=SeqEngine())
    encoder = InstrumentedEncoder(model_name="test-model")

    print("\n[TEST] Running .map() with SeqEngine (3 items):")
    results = pipeline.map(
        inputs={
            "text": ["Hello", "World", "Test"],
            "encoder": encoder,
        },
        map_over="text",
    )

    print(f"\nResults: {len(results)} items")
    print(f"Singular calls: {encoder.singular_calls}")
    print(f"Batch calls: {encoder.batch_calls}")

    # SeqEngine should call singular 3 times, batch 0 times
    assert encoder.singular_calls == 3, (
        f"Expected 3 singular calls, got {encoder.singular_calls}"
    )
    assert encoder.batch_calls == 0, (
        f"Expected 0 batch calls, got {encoder.batch_calls}"
    )

    print("\n✅ SeqEngine correctly uses singular function!")


# ============================================================================
# Test with DaftEngine (should use batch)
# ============================================================================


def test_daft_engine():
    """Test that DaftEngine uses batch function."""
    print("\n" + "=" * 60)
    print("Testing DaftEngine (should use BATCH)")
    print("=" * 60)

    pipeline = Pipeline(nodes=[encode_node], engine=DaftEngine())
    encoder = InstrumentedEncoder(model_name="test-model")

    print("\n[TEST] Running .map() with DaftEngine (3 items):")
    results = pipeline.map(
        inputs={
            "text": ["Hello", "World", "Test"],
            "encoder": encoder,
        },
        map_over="text",
    )

    print(f"\nResults: {len(results)} items")
    print(f"Singular calls: {encoder.singular_calls}")
    print(f"Batch calls: {encoder.batch_calls}")

    # DaftEngine should call batch function, NOT singular
    assert encoder.batch_calls > 0, f"Expected batch calls, got {encoder.batch_calls}"
    print(f"  ✓ Batch function was called {encoder.batch_calls} time(s)")

    # Verify results
    assert len(results) == 3
    assert all(r["encoded_text"] == [0.1, 0.2, 0.3] for r in results)

    print("\n✅ DaftEngine correctly uses batch function!")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DualNode DaftEngine Integration Test")
    print("=" * 60)

    test_sequential_engine()
    test_daft_engine()

    print("\n" + "=" * 60)
    print("🎉 All tests passed!")
    print("=" * 60)
