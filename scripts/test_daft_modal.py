"""
Test DaftEngine serialization bug on Modal.

This script runs on Modal to reproduce the exact environment where the bug occurs.
"""

import modal

# Create Modal app
app = modal.App("test-daft-serialization")

# Modal image with dependencies and mounted source
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "daft",
        "cloudpickle",
        "pydantic",
        "networkx",
        "tqdm",
        "rich",
        "graphviz",
    )
    .add_local_dir(
        "/Users/giladrubin/python_workspace/hypernodes/src/hypernodes",
        remote_path="/root/hypernodes",
    )
)


# ============================================================================
# CUSTOM CLASSES (defined in script module)
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

        Key issue: This method has Document in type annotations.
        When pickled to Modal's worker, these references need to work.
        """
        encoded_text = f"[{self.model_name}]: {doc.text}"
        return Document(encoded_text)


# ============================================================================
# MODAL FUNCTION
# ============================================================================


@app.function(image=image)
def run_test():
    """Run the test on Modal."""
    import sys
    sys.path.insert(0, "/root")

    from hypernodes import Pipeline, node
    from hypernodes.engines import DaftEngine

    print("="*70)
    print("Running on Modal")
    print("="*70)
    print(f"\nDocument.__module__ = '{Document.__module__}'")
    print(f"DocumentEncoder.__module__ = '{DocumentEncoder.__module__}'")

    # Node using stateful encoder
    @node(output_name="encoded_doc")
    def encode_document(text: str, encoder: DocumentEncoder) -> str:
        """Encode a document using the stateful encoder."""
        doc = Document(text)
        encoded = encoder.encode(doc)
        return encoded.text

    # Create encoder instance
    print("\nCreating DocumentEncoder instance...")
    encoder = DocumentEncoder(model_name="modal-test")

    # Build pipeline
    print("Building pipeline with DaftEngine...")
    pipeline = Pipeline(
        nodes=[encode_document],
        engine=DaftEngine(collect=True, debug=True),
    )

    print("\nRunning pipeline on Modal...")
    print("(This will create Daft UDF worker processes)")
    print("(Worker needs to unpickle DocumentEncoder with Document references)\n")

    try:
        result = pipeline.run(inputs={
            "text": "Hello from Modal",
            "encoder": encoder,
        })

        print("\n" + "="*70)
        print(f"✓ SUCCESS: {result}")
        print("="*70)
        return {"status": "success", "result": result}

    except Exception as e:
        print("\n" + "="*70)
        print(f"✗ FAILED: {type(e).__name__}")
        print("="*70)
        print(f"\nError: {str(e)[:500]}")

        if "test_daft_modal" in str(e) or "ModuleNotFoundError" in str(e):
            print("\n" + "!"*70)
            print("BUG REPRODUCED ON MODAL!")
            print("!"*70)
            print("\nWorker couldn't import the script module")

        import traceback
        traceback.print_exc()
        raise


@app.local_entrypoint()
def main():
    """Local entrypoint for Modal."""
    print("Starting Modal test...")
    result = run_test.remote()
    print(f"\nFinal result: {result}")


if __name__ == "__main__":
    # For local testing
    run_test.local()
