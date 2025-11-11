"""
MINIMAL REPRODUCTION OF DAFT SERIALIZATION BUG

This script demonstrates the issue where custom classes fail to serialize/deserialize
properly when used with DaftEngine in distributed environments (like Modal).

## The Problem

When you:
1. Define custom classes in a script module (not in __main__)
2. Use these classes in type annotations of stateful objects
3. Run with DaftEngine which creates @daft.cls UDF wrappers
4. Execute in a distributed environment where the module isn't available

Then: Daft's UDF worker fails with "ModuleNotFoundError: No module named 'your_module'"

## Why It Happens

DaftEngine tries to fix this by changing __module__ to "__main__" to force by-value
serialization. However, there appear to be edge cases where module references still
leak through, especially with:
- Complex type annotations
- Nested class references
- Methods with typed parameters

## This Script

Tests the scenario by manually creating classes that appear to be from an
unimportable module, which triggers DaftEngine's fix logic. In practice, the
bug manifests on Modal where script modules aren't in the worker's Python path.
"""

from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine


# ============================================================================
# CUSTOM CLASSES (simulating being from a separate module)
# ============================================================================


class Document:
    """A document data class."""

    def __init__(self, text: str):
        self.text = text

    def __repr__(self):
        return f"Document('{self.text[:30]}...')"


class DocumentEncoder:
    """Encoder with method that uses Document in type annotations."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, doc: Document) -> Document:
        """
        Encode a document.

        This method signature is key: it uses Document in type annotations.
        When this class is pickled for Daft's UDF worker, these annotations
        need to be serializable.
        """
        encoded_text = f"[{self.model_name}]: {doc.text}"
        return Document(encoded_text)


# Simulate being from an unimportable module (like 'test_modal' on Modal)
Document.__module__ = "nonexistent_script_module"
DocumentEncoder.__module__ = "nonexistent_script_module"


# ============================================================================
# HYPERNODES PIPELINE
# ============================================================================


@node(output_name="encoded_doc")
def encode_document(text: str, encoder: DocumentEncoder) -> str:
    """Encode a document using the stateful encoder."""
    doc = Document(text)
    encoded = encoder.encode(doc)  # Calls method with Document type hints
    return encoded.text


# Force Daft to spawn a UDF worker process (where the bug manifests)
encode_document.func.__daft_udf_config__ = {
    "use_process": True,
    "max_concurrency": 1,
}


# ============================================================================
# TEST
# ============================================================================


def main():
    """Run the test."""
    print("="*70)
    print("Testing DaftEngine Serialization Bug")
    print("="*70)

    print(f"\nDocumentEncoder.__module__ = '{DocumentEncoder.__module__}'")
    print(f"Document.__module__ = '{Document.__module__}'")

    # Verify module is not importable
    import importlib.util
    spec = importlib.util.find_spec("nonexistent_script_module")
    print(f"\nimportlib.util.find_spec('nonexistent_script_module') = {spec}")
    print("(None = not importable, triggers DaftEngine's fix logic)\n")

    # Create stateful encoder
    encoder = DocumentEncoder(model_name="test-model")

    # Build pipeline with DaftEngine
    pipeline = Pipeline(
        nodes=[encode_document],
        engine=DaftEngine(collect=True, debug=True),
    )

    print("Running pipeline with use_process=True...")
    print("(DaftEngine will create @daft.cls wrapper with encoder in payload)")
    print("(Worker process will try to unpickle the encoder instance)\n")

    try:
        result = pipeline.run(inputs={
            "text": "Hello World",
            "encoder": encoder,
        })

        print("\n" + "="*70)
        print(f"✓ SUCCESS: {result}")
        print("="*70)
        print("\nThe serialization fix is working correctly!")
        return True

    except Exception as e:
        print("\n" + "="*70)
        print(f"✗ FAILED: {type(e).__name__}")
        print("="*70)
        print(f"\nError message:\n{str(e)[:500]}")

        if "nonexistent_script_module" in str(e) or "ModuleNotFoundError" in str(e):
            print("\n" + "!"*70)
            print("BUG REPRODUCED!")
            print("!"*70)
            print("\nThe worker tried to import 'nonexistent_script_module'")
            print("This is the same issue that happens on Modal with 'test_modal'")

        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
