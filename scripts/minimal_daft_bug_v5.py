"""Force Daft to spawn UDF worker process to trigger the serialization bug.

This version:
1. Imports classes from a separate module
2. Forces use_process=True via __daft_udf_config__
3. Uses map() to process multiple items
"""

from typing import List
from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine

# Import from separate module
from custom_models import TextData, TextEncoder


# Node with forced use_process=True to spawn worker processes
@node(output_name="encoded_text")
def encode_text(text: str, encoder: TextEncoder) -> str:
    """Encode text using the stateful encoder."""
    data = TextData(text)
    encoded = encoder.encode(data)
    return encoded.content


# Add Daft UDF config to force process spawning
encode_text.func.__daft_udf_config__ = {
    "use_process": True,  # Force spawning a separate process
    "max_concurrency": 2,
}


# Node to create multiple texts for map operation
@node(output_name="texts")
def create_texts() -> List[str]:
    """Create multiple texts to process."""
    return [
        "Text 1",
        "Text 2",
        "Text 3",
        "Text 4",
        "Text 5",
    ]


def main():
    """Run test with process spawning."""
    print("="*60)
    print("Testing DaftEngine with use_process=True")
    print("="*60)

    print("\n1. Creating TextEncoder from custom_models module...")
    print(f"   TextEncoder module: {TextEncoder.__module__}")
    print(f"   TextData module: {TextData.__module__}")
    encoder = TextEncoder(prefix="PROC")

    print("\n2. Building pipeline with map operation...")

    # Create inner pipeline for single text
    encode_single = Pipeline(
        nodes=[encode_text],
        name="encode_single",
    )

    # Wrap as node with map_over
    encode_all = encode_single.as_node(
        input_mapping={
            "text": "text",
            "encoder": "encoder",
        },
        output_mapping={
            "encoded_text": "result",
        },
        map_over="text",
    )

    # Outer pipeline
    pipeline = Pipeline(
        nodes=[create_texts, encode_all],
        engine=DaftEngine(collect=True, debug=True),
        name="process_all",
    )

    print("\n3. Running pipeline with map operation...")
    print("   This should force Daft to spawn UDF worker processes")
    print("   Worker processes need to deserialize TextEncoder instances")
    print()

    inputs = {"encoder": encoder}

    try:
        result = pipeline.run(inputs=inputs)
        print(f"\n{'='*60}")
        print(f"SUCCESS! Results: {result.get('result', [])}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {type(e).__name__}")
        print(f"Error message (first 500 chars):")
        print(str(e)[:500])
        print(f"{'='*60}\n")

        if "custom_models" in str(e):
            print("\n✓ REPRODUCED THE BUG!")
            print("  Error mentions 'custom_models' module")

        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
