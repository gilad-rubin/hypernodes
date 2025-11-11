"""
Final verification that the serialization fixes work correctly.
Tests from both parent directory and script directory context.
"""

from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine


# Test classes from "unimportable" module
class TestDocument:
    """Simple test document."""
    def __init__(self, text: str):
        self.text = text


class TestEncoder:
    """Simple encoder with typed methods."""
    def __init__(self, model: str):
        self.model = model

    def encode(self, doc: TestDocument) -> TestDocument:
        return TestDocument(f"[{self.model}] {doc.text}")


# Simulate unimportable module
TestDocument.__module__ = "fake_unimportable_module"
TestEncoder.__module__ = "fake_unimportable_module"


@node(output_name="result")
def process(text: str, encoder: TestEncoder) -> str:
    doc = TestDocument(text)
    encoded = encoder.encode(doc)
    return encoded.text


# Force use_process to trigger serialization
process.func.__daft_udf_config__ = {"use_process": True, "max_concurrency": 1}


def main():
    print("=" * 70)
    print("FINAL VERIFICATION TEST")
    print("=" * 70)

    encoder = TestEncoder("verification-model")

    pipeline = Pipeline(
        nodes=[process],
        engine=DaftEngine(collect=True, debug=False),
    )

    result = pipeline.run(inputs={
        "text": "Testing serialization",
        "encoder": encoder,
    })

    expected = "[verification-model] Testing serialization"
    actual = result["result"]

    if actual == expected:
        print(f"\n✓ SUCCESS: Serialization fixes working correctly!")
        print(f"  Result: {actual}")
        return True
    else:
        print(f"\n✗ FAILED: Unexpected result")
        print(f"  Expected: {expected}")
        print(f"  Actual: {actual}")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
