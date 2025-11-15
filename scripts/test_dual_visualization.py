"""Test DualNode visualization improvements."""

from typing import List

from hypernodes import DualNode, Pipeline

try:
    from daft import Series
except ImportError:
    Series = List


class Encoder:
    def encode(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]

    def encode_batch(self, texts: Series) -> Series:
        return Series.from_pylist([[0.1, 0.2, 0.3] for _ in texts.to_pylist()])


def encode_singular(text: str, encoder: Encoder) -> List[float]:
    """Singular encoding function."""
    return encoder.encode(text)


def encode_batch(texts: Series, encoder: Encoder) -> Series:
    """Batch encoding function."""
    return encoder.encode_batch(texts)


# Create DualNode
node = DualNode(output_name="encoded_text", singular=encode_singular, batch=encode_batch)

print("=" * 60)
print("DualNode Visualization Test")
print("=" * 60)

print(f"\n✅ Node name: '{node.name}'")
print(f"   (stripped '_singular' suffix: encode_singular → encode)")

print(f"\n✅ Has func property: {hasattr(node, 'func')}")
print(f"   (for visualization type hint extraction)")

print(f"\n✅ Func name: {node.func.__name__}")
print(f"   (same as singular)")

# Test visualization
pipeline = Pipeline(nodes=[node])

print("\n" + "=" * 60)
print("Generating Visualization...")
print("=" * 60)
print("\nExpected visualization:")
print("  - Node name: 'encode ⚡' (with lightning bolt)")
print("  - Type hints: text : str, encoder : Encoder")
print("  - Return type: list[float] or List[float]")
print("\nRendering...")

pipeline.visualize()

print("\n✅ Visualization complete!")
print("\nLook for:")
print("  1. Node labeled 'encode ⚡' (not 'encode_singular')")
print("  2. Input parameters with type hints visible")
print("  3. Return type shown in output box")
