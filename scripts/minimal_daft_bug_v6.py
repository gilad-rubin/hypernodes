"""Simplest possible test with use_process=True to force UDF worker spawning.

This should trigger the serialization issue when Daft spawns a worker process.
"""

from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine

# Import from separate module - the key to triggering the bug
from custom_models import TextData, TextEncoder


# Node that uses stateful encoder with use_process=True
@node(output_name="result")
def process_with_encoder(text: str, encoder: TextEncoder) -> str:
    """Process text with encoder in a separate process."""
    data = TextData(text)
    result = encoder.encode(data)
    return result.content


# Force Daft to use a separate process for this UDF
process_with_encoder.func.__daft_udf_config__ = {
    "use_process": True,
    "max_concurrency": 1,
}


def main():
    """Run test that forces process spawning."""
    print("="*60)
    print("Testing with use_process=True (forces UDF worker)")
    print("="*60)

    print("\n1. Creating encoder from custom_models module")
    print(f"   TextEncoder.__module__ = {TextEncoder.__module__}")
    encoder = TextEncoder(prefix="TEST")

    print("\n2. Building pipeline")
    print("   process_with_encoder has __daft_udf_config__ = use_process=True")

    pipeline = Pipeline(
        nodes=[process_with_encoder],
        engine=DaftEngine(collect=True, debug=True),
    )

    print("\n3. Running - Daft should spawn UDF worker process")
    print("   Worker needs to unpickle encoder (which references custom_models.TextData)")
    print()

    try:
        result = pipeline.run(inputs={"text": "Hello", "encoder": encoder})
        print(f"\n{'='*60}")
        print(f"SUCCESS: {result}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {type(e).__name__}")
        error_str = str(e)
        print(f"Error (first 300 chars): {error_str[:300]}")

        if "custom_models" in error_str or "ModuleNotFoundError" in error_str:
            print("\n✓ REPRODUCED! Module import error in UDF worker")

        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
