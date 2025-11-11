"""
Simplified test focusing purely on serialization of stateful objects with typed methods.

This removes map operations and focuses on the core issue: whether stateful objects
with methods that have custom type annotations can be serialized to Modal workers.
"""

import modal

# Create Modal app
app = modal.App("test-daft-serialization-simple")

# Modal image with dependencies
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
# CUSTOM CLASSES
# ============================================================================


class Document:
    """Document class that will be used in type annotations."""
    def __init__(self, text: str, doc_id: int):
        self.text = text
        self.doc_id = doc_id

    def __repr__(self):
        return f"Document(id={self.doc_id}, text='{self.text[:20]}...')"


class Encoder:
    """Encoder with methods using Document in type annotations."""
    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, doc: Document) -> Document:
        """Critical: This method signature uses Document from this module."""
        encoded_text = f"[{self.model_name}] {doc.text}"
        return Document(encoded_text, doc.doc_id)


class Fusion:
    """Fusion class with typed methods."""
    def __init__(self, strategy: str):
        self.strategy = strategy

    def fuse(self, doc1: Document, doc2: Document) -> Document:
        """Fuse two documents."""
        fused_text = f"{doc1.text} | {doc2.text}"
        return Document(fused_text, doc1.doc_id)


# ============================================================================
# MODAL FUNCTION
# ============================================================================


@app.function(image=image)
def run_simple_test():
    """Run simple test with multiple stateful objects."""
    import sys
    sys.path.insert(0, "/root")

    from hypernodes import Pipeline, node
    from hypernodes.engines import DaftEngine

    print("="*70)
    print("Simple serialization test on Modal")
    print("="*70)
    print(f"\nDocument.__module__ = '{Document.__module__}'")
    print(f"Encoder.__module__ = '{Encoder.__module__}'")
    print(f"Fusion.__module__ = '{Fusion.__module__}'")

    # Nodes using stateful objects with typed methods
    @node(output_name="doc1")
    def create_doc1() -> Document:
        return Document("First document", 1)

    @node(output_name="doc2")
    def create_doc2() -> Document:
        return Document("Second document", 2)

    @node(output_name="encoded_doc1")
    def encode_doc1(doc1: Document, encoder: Encoder) -> Document:
        """Uses Encoder.encode which has Document-typed parameters."""
        return encoder.encode(doc1)

    @node(output_name="encoded_doc2")
    def encode_doc2(doc2: Document, encoder: Encoder) -> Document:
        """Uses Encoder.encode which has Document-typed parameters."""
        return encoder.encode(doc2)

    @node(output_name="fused")
    def fuse_docs(encoded_doc1: Document, encoded_doc2: Document, fusion: Fusion) -> Document:
        """Uses Fusion.fuse which has Document-typed parameters."""
        return fusion.fuse(encoded_doc1, encoded_doc2)

    @node(output_name="result")
    def extract_text(fused: Document) -> str:
        return fused.text

    # Force use_process on critical nodes
    encode_doc1.func.__daft_udf_config__ = {"use_process": True, "max_concurrency": 1}
    fuse_docs.func.__daft_udf_config__ = {"use_process": True, "max_concurrency": 1}

    # Create stateful objects
    print("\nCreating stateful objects...")
    encoder = Encoder("modal-encoder-v1")
    fusion = Fusion("concatenate")

    print("Stateful objects created:")
    print(f"  encoder: {type(encoder).__module__}.{type(encoder).__name__}")
    print(f"  fusion: {type(fusion).__module__}.{type(fusion).__name__}")

    # Build pipeline
    print("\nBuilding pipeline...")
    pipeline = Pipeline(
        nodes=[create_doc1, create_doc2, encode_doc1, encode_doc2, fuse_docs, extract_text],
        engine=DaftEngine(collect=True, debug=True),
    )

    print("\nRunning pipeline on Modal with use_process=True...")
    print("Key test: Can stateful objects with Document-typed methods")
    print("           be serialized to Modal workers?\n")

    try:
        result = pipeline.run(inputs={
            "encoder": encoder,
            "fusion": fusion,
        })

        print("\n" + "="*70)
        print(f"✓ SUCCESS!")
        print(f"Result: {result['result'][:100]}")
        print("="*70)
        print("\nStateful object serialization works correctly!")
        return {"status": "success", "result": result}

    except Exception as e:
        print("\n" + "="*70)
        print(f"✗ FAILED: {type(e).__name__}")
        print("="*70)
        error_str = str(e)
        print(f"\nError message (first 600 chars):")
        print(error_str[:600])

        if "test_daft_modal_simple" in error_str or "ModuleNotFoundError" in error_str:
            print("\n" + "!"*70)
            print("BUG REPRODUCED!")
            print("!"*70)
            print("The worker couldn't import the module with custom classes")

        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        raise


@app.local_entrypoint()
def main():
    """Local entrypoint."""
    print("Starting simple serialization test on Modal...")
    result = run_simple_test.remote()
    print(f"\nFinal result: {result['status']}")


if __name__ == "__main__":
    run_simple_test.local()
