"""Minimal script to reproduce DaftEngine serialization bug with imported classes.

This version imports classes from a separate module (custom_models.py),
which should trigger the module resolution issue that causes:
    ModuleNotFoundError: No module named 'custom_models'
in Daft's UDF worker process.
"""

from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine

# Import from separate module - THIS IS THE KEY!
# When DaftEngine tries to serialize these classes, the module name matters
from custom_models import TextData, TextEncoder


# Node that uses the stateful encoder
@node(output_name="encoded_text")
def encode_text(text: str, encoder: TextEncoder) -> str:
    """Encode text using the stateful encoder."""
    data = TextData(text)
    encoded = encoder.encode(data)  # Calls method with TextData annotations
    return encoded.content


def main():
    """Run the test that triggers the serialization issue."""
    print("="*60)
    print("Testing DaftEngine with classes from separate module")
    print("="*60)

    # Create the stateful encoder
    print("\n1. Creating TextEncoder from custom_models module...")
    print(f"   TextEncoder module: {TextEncoder.__module__}")
    print(f"   TextData module: {TextData.__module__}")
    encoder = TextEncoder(prefix="PROCESSED")

    # Build pipeline with DaftEngine
    print("\n2. Building pipeline with DaftEngine...")
    pipeline = Pipeline(
        nodes=[encode_text],
        engine=DaftEngine(
            collect=True,
            debug=True,
        ),
    )

    print("\n3. Running pipeline...")
    print("   DaftEngine will try to fix module names to '__main__'")
    print("   But Daft's UDF worker may still try to import 'custom_models'")
    print()

    inputs = {
        "text": "Hello World",
        "encoder": encoder,
    }

    try:
        result = pipeline.run(inputs=inputs)
        print(f"\n{'='*60}")
        print(f"SUCCESS! Result: {result}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {type(e).__name__}")
        print(f"First 500 chars of error:")
        print(str(e)[:500])
        print(f"{'='*60}\n")

        # Check if it's the module error we're looking for
        if "custom_models" in str(e) and "No module named" in str(e):
            print("✓ REPRODUCED THE BUG!")
            print("  The error shows Daft's UDF worker couldn't find 'custom_models'")

        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
