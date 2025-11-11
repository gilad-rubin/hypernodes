"""Test Level 8 only - nested stateful objects."""

import modal
from pathlib import Path

app = modal.App("test-level-8")

hypernodes_dir = Path("/Users/giladrubin/python_workspace/hypernodes/src/hypernodes")
models_file = Path("/Users/giladrubin/python_workspace/hypernodes/scripts/models_test.py")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("daft", "cloudpickle", "pydantic", "networkx", "tqdm", "rich", "graphviz")
    .add_local_dir(str(hypernodes_dir), remote_path="/root/hypernodes")
    .add_local_file(str(models_file), remote_path="/root/models_test.py")
)


@app.function(image=image)
def test_nested_stateful():
    """Test stateful object containing other stateful objects."""
    import sys
    sys.path.insert(0, "/root")

    from hypernodes import Pipeline, node
    from hypernodes.engines import DaftEngine
    from models_test import Document, EncoderWithHints, Reranker2

    print("\n" + "="*70)
    print("LEVEL 8: Stateful object CONTAINING other stateful objects")
    print("="*70)

    @node(output_name="doc")
    def create_doc() -> Document:
        return Document(text="test", doc_id=1)

    @node(output_name="reranked")
    def rerank_doc(doc: Document, reranker: Reranker2) -> Document:
        return reranker.rerank(doc)

    @node(output_name="result")
    def extract(reranked: Document) -> str:
        return reranked.text

    # Create nested stateful objects
    print("\nCreating nested stateful objects...")
    encoder = EncoderWithHints(model_name="inner-encoder")
    print(f"  encoder module: {encoder.__class__.__module__}")
    
    doc_lookup = {str(i): Document(f"doc{i}", i) for i in range(3)}
    print(f"  doc_lookup has {len(doc_lookup)} items")
    
    reranker = Reranker2(encoder=encoder, doc_lookup=doc_lookup)
    print(f"  reranker module: {reranker.__class__.__module__}")
    print(f"  reranker._encoder module: {reranker._encoder.__class__.__module__}")

    pipeline = Pipeline(
        nodes=[create_doc, rerank_doc, extract],
        engine=DaftEngine(collect=True, debug=True),
    )

    print("\nRunning pipeline...")
    result = pipeline.run(inputs={"reranker": reranker})
    print(f"\n✓ Level 8 passed: {result}")
    return {"level": 8, "status": "pass"}


@app.local_entrypoint()
def main():
    """Run test."""
    print("Testing nested stateful objects...")
    try:
        result = test_nested_stateful.remote()
        print(f"\n✓ TEST PASSED: {result}")
    except Exception as e:
        print(f"\n✗ TEST FAILED: {type(e).__name__}")
        print(f"Error: {str(e)[:800]}")
        
        if "models_test" in str(e) or "ModuleNotFoundError" in str(e):
            print("\n" + "!"*70)
            print("BUG REPRODUCED!")
            print("!"*70)
        
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_nested_stateful.local()
