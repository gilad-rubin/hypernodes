"""Minimal script to reproduce DaftEngine serialization bug with stateful classes.

This version closely mimics the original error by having:
1. A custom data class (like Document/Passage)
2. A stateful processor class with methods that use the data class in type annotations
3. A node that uses the stateful processor
4. DaftEngine execution that creates a @daft.cls wrapper

The key issue: when the stateful processor's methods reference custom classes
in type annotations, those classes need to be properly serialized for Daft's
UDF worker process.
"""

from typing import List
from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine


# Custom data class that will be used in type annotations (mimics the original models)
class TextData:
    """A simple data class to demonstrate the serialization issue."""

    def __init__(self, content: str):
        self.content = content

    def __repr__(self):
        return f"TextData('{self.content}')"


# Stateful processor with method that has typed annotations using TextData
class TextEncoder:
    """Encoder that processes TextData objects - method explicitly typed."""

    def __init__(self, prefix: str = "ENCODED"):
        self.prefix = prefix

    def encode(self, data: TextData) -> TextData:
        """
        Encode TextData - THIS METHOD SIGNATURE IS THE KEY!
        The type annotation references TextData from this module.
        """
        return TextData(f"{self.prefix}[{data.content}]")


# Node that uses the stateful encoder
@node(output_name="encoded_text")
def encode_text(text: str, encoder: TextEncoder) -> str:
    """Encode text using the stateful encoder."""
    # Create TextData instance and pass to encoder
    data = TextData(text)
    encoded = encoder.encode(data)  # This calls the method with TextData annotations
    return encoded.content


def main():
    """Run the test that triggers the serialization issue."""
    print("="*60)
    print("Testing DaftEngine with stateful class containing typed methods")
    print("="*60)

    # Create the stateful encoder (this will be in the payload)
    print("\n1. Creating TextEncoder instance...")
    encoder = TextEncoder(prefix="PROCESSED")

    # Build pipeline with DaftEngine
    print("2. Building pipeline with DaftEngine...")
    pipeline = Pipeline(
        nodes=[encode_text],
        engine=DaftEngine(
            collect=True,
            debug=True,  # Enable debug output
        ),
    )

    print("3. Running pipeline...")
    print("   (DaftEngine will create a @daft.cls wrapper)")
    print("   (The encoder instance will be in the wrapper's payload)")
    print("   (When Daft pickles this, it needs to serialize TextEncoder)")
    print("   (TextEncoder.encode has TextData in its type annotations)")
    print("   (If TextData's module is not properly fixed, unpickling fails)")
    print()

    inputs = {
        "text": "Hello World",
        "encoder": encoder,  # Stateful parameter
    }

    try:
        result = pipeline.run(inputs=inputs)
        print(f"\n{'='*60}")
        print(f"SUCCESS! Result: {result}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {type(e).__name__}")
        print(f"Message: {str(e)[:200]}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
