"""
Progressive complexity test to find what causes the serialization bug.

Start simple and uncomment features progressively until it breaks.
"""

import modal
from pathlib import Path

# Create Modal app
app = modal.App("test-progressive")

# Get paths
hypernodes_dir = Path("/Users/giladrubin/python_workspace/hypernodes/src/hypernodes")
models_file = Path("/Users/giladrubin/python_workspace/hypernodes/scripts/models_test.py")

# Modal image
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
        "numpy",
        "pandas",
    )
    .add_local_dir(str(hypernodes_dir), remote_path="/root/hypernodes")
    .add_local_file(str(models_file), remote_path="/root/models_test.py")
)


@app.function(image=image)
def test_level_1_simple():
    """Level 1: Simple classes from separate module."""
    import sys
    sys.path.insert(0, "/root")

    from hypernodes import Pipeline, node
    from hypernodes.engines import DaftEngine
    from models_test import Document

    print("\n" + "="*70)
    print("LEVEL 1: Simple class from separate module")
    print("="*70)

    @node(output_name="doc")
    def create_doc() -> Document:
        return Document(text="test", doc_id=1)

    @node(output_name="result")
    def extract(doc: Document) -> str:
        return doc.text

    pipeline = Pipeline(
        nodes=[create_doc, extract],
        engine=DaftEngine(collect=True, debug=True),
    )

    result = pipeline.run(inputs={})
    print(f"✓ Level 1 passed: {result}")
    return {"level": 1, "status": "pass"}


@app.function(image=image)
def test_level_2_stateful():
    """Level 2: Stateful class with typed methods."""
    import sys
    sys.path.insert(0, "/root")

    from hypernodes import Pipeline, node
    from hypernodes.engines import DaftEngine
    from models_test import Document, Encoder

    print("\n" + "="*70)
    print("LEVEL 2: Stateful class with typed methods")
    print("="*70)

    @node(output_name="doc")
    def create_doc() -> Document:
        return Document(text="test", doc_id=1)

    @node(output_name="encoded")
    def encode_doc(doc: Document, encoder: Encoder) -> Document:
        return encoder.encode(doc)

    @node(output_name="result")
    def extract(encoded: Document) -> str:
        return encoded.text

    encoder = Encoder(model_name="test-model")

    pipeline = Pipeline(
        nodes=[create_doc, encode_doc, extract],
        engine=DaftEngine(collect=True, debug=True),
    )

    result = pipeline.run(inputs={"encoder": encoder})
    print(f"✓ Level 2 passed: {result}")
    return {"level": 2, "status": "pass"}


@app.function(image=image)
def test_level_3_use_process():
    """Level 3: Add use_process=True."""
    import sys
    sys.path.insert(0, "/root")

    from hypernodes import Pipeline, node
    from hypernodes.engines import DaftEngine
    from models_test import Document, Encoder

    print("\n" + "="*70)
    print("LEVEL 3: With use_process=True")
    print("="*70)

    @node(output_name="doc")
    def create_doc() -> Document:
        return Document(text="test", doc_id=1)

    @node(output_name="encoded")
    def encode_doc(doc: Document, encoder: Encoder) -> Document:
        return encoder.encode(doc)

    @node(output_name="result")
    def extract(encoded: Document) -> str:
        return encoded.text

    # Force use_process
    encode_doc.func.__daft_udf_config__ = {"use_process": True, "max_concurrency": 1}

    encoder = Encoder(model_name="test-model")

    pipeline = Pipeline(
        nodes=[create_doc, encode_doc, extract],
        engine=DaftEngine(collect=True, debug=True),
    )

    result = pipeline.run(inputs={"encoder": encoder})
    print(f"✓ Level 3 passed: {result}")
    return {"level": 3, "status": "pass"}


@app.function(image=image)
def test_level_4_daft_hints():
    """Level 4: Add __daft_hint__ attributes."""
    import sys
    sys.path.insert(0, "/root")

    from hypernodes import Pipeline, node
    from hypernodes.engines import DaftEngine
    from models_test import Document, EncoderWithHints

    print("\n" + "="*70)
    print("LEVEL 4: With __daft_hint__ attributes")
    print("="*70)

    @node(output_name="doc")
    def create_doc() -> Document:
        return Document(text="test", doc_id=1)

    @node(output_name="encoded")
    def encode_doc(doc: Document, encoder: EncoderWithHints) -> Document:
        return encoder.encode(doc)

    @node(output_name="result")
    def extract(encoded: Document) -> str:
        return encoded.text

    encoder = EncoderWithHints(model_name="test-model")

    pipeline = Pipeline(
        nodes=[create_doc, encode_doc, extract],
        engine=DaftEngine(collect=True, debug=True),
    )

    result = pipeline.run(inputs={"encoder": encoder})
    print(f"✓ Level 4 passed: {result}")
    return {"level": 4, "status": "pass"}


@app.function(image=image)
def test_level_5_multiple_stateful():
    """Level 5: Multiple stateful objects."""
    import sys
    sys.path.insert(0, "/root")

    from hypernodes import Pipeline, node
    from hypernodes.engines import DaftEngine
    from models_test import Document, EncoderWithHints, Fusion

    print("\n" + "="*70)
    print("LEVEL 5: Multiple stateful objects")
    print("="*70)

    @node(output_name="doc1")
    def create_doc1() -> Document:
        return Document(text="doc1", doc_id=1)

    @node(output_name="doc2")
    def create_doc2() -> Document:
        return Document(text="doc2", doc_id=2)

    @node(output_name="encoded1")
    def encode_doc1(doc1: Document, encoder: EncoderWithHints) -> Document:
        return encoder.encode(doc1)

    @node(output_name="encoded2")
    def encode_doc2(doc2: Document, encoder: EncoderWithHints) -> Document:
        return encoder.encode(doc2)

    @node(output_name="fused")
    def fuse_docs(encoded1: Document, encoded2: Document, fusion: Fusion) -> Document:
        return fusion.fuse(encoded1, encoded2)

    @node(output_name="result")
    def extract(fused: Document) -> str:
        return fused.text

    encoder = EncoderWithHints(model_name="test-model")
    fusion = Fusion(strategy="concat")

    pipeline = Pipeline(
        nodes=[create_doc1, create_doc2, encode_doc1, encode_doc2, fuse_docs, extract],
        engine=DaftEngine(collect=True, debug=True),
    )

    result = pipeline.run(inputs={"encoder": encoder, "fusion": fusion})
    print(f"✓ Level 5 passed: {result}")
    return {"level": 5, "status": "pass"}


@app.function(image=image)
def test_level_6_nested_types():
    """Level 6: Complex nested type annotations."""
    import sys
    sys.path.insert(0, "/root")

    from typing import List
    from hypernodes import Pipeline, node
    from hypernodes.engines import DaftEngine
    from models_test import Document, Hit, Ranker

    print("\n" + "="*70)
    print("LEVEL 6: Complex nested types (List[CustomClass])")
    print("="*70)

    @node(output_name="docs")
    def create_docs() -> List[Document]:
        return [Document(text=f"doc{i}", doc_id=i) for i in range(3)]

    @node(output_name="hits")
    def docs_to_hits(docs: List[Document]) -> List[Hit]:
        return [Hit(doc_id=d.doc_id, score=1.0) for d in docs]

    @node(output_name="ranked")
    def rank_hits(hits: List[Hit], ranker: Ranker) -> List[Hit]:
        return ranker.rank(hits)

    @node(output_name="result")
    def count_hits(ranked: List[Hit]) -> int:
        return len(ranked)

    ranker = Ranker(strategy="score")

    pipeline = Pipeline(
        nodes=[create_docs, docs_to_hits, rank_hits, count_hits],
        engine=DaftEngine(collect=True, debug=True),
    )

    result = pipeline.run(inputs={"ranker": ranker})
    print(f"✓ Level 6 passed: {result}")
    return {"level": 6, "status": "pass"}


@app.function(image=image)
def test_level_7_dict_annotations():
    """Level 7: Dict with custom class values in annotations."""
    import sys
    sys.path.insert(0, "/root")

    from typing import List
    from hypernodes import Pipeline, node
    from hypernodes.engines import DaftEngine
    from models_test import Document, Hit, Reranker

    print("\n" + "="*70)
    print("LEVEL 7: Dict[str, CustomClass] in annotations")
    print("="*70)

    @node(output_name="docs")
    def create_docs() -> List[Document]:
        return [Document(text=f"doc{i}", doc_id=i) for i in range(3)]

    @node(output_name="doc_lookup")
    def build_lookup(docs: List[Document]) -> dict:
        return {str(d.doc_id): d for d in docs}

    @node(output_name="hits")
    def create_hits() -> List[Hit]:
        return [Hit(doc_id=i, score=float(i)) for i in range(3)]

    @node(output_name="reranked")
    def rerank_hits(hits: List[Hit], doc_lookup: dict, reranker: Reranker) -> List[Hit]:
        return reranker.rerank(hits, doc_lookup)

    @node(output_name="result")
    def count_hits(reranked: List[Hit]) -> int:
        return len(reranked)

    reranker = Reranker(model="test")

    pipeline = Pipeline(
        nodes=[create_docs, build_lookup, create_hits, rerank_hits, count_hits],
        engine=DaftEngine(collect=True, debug=True),
    )

    result = pipeline.run(inputs={"reranker": reranker})
    print(f"✓ Level 7 passed: {result}")
    return {"level": 7, "status": "pass"}


@app.local_entrypoint()
def main():
    """Run all tests progressively."""
    tests = [
        ("Level 1: Simple class", test_level_1_simple),
        ("Level 2: Stateful class", test_level_2_stateful),
        ("Level 3: use_process=True", test_level_3_use_process),
        ("Level 4: __daft_hint__", test_level_4_daft_hints),
        ("Level 5: Multiple stateful", test_level_5_multiple_stateful),
        ("Level 6: Nested types", test_level_6_nested_types),
        ("Level 7: Dict annotations", test_level_7_dict_annotations),
    ]

    print("\n" + "="*70)
    print("PROGRESSIVE COMPLEXITY TESTING")
    print("="*70)

    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            result = test_func.remote()
            print(f"  ✓ {test_name}: PASSED")
        except Exception as e:
            print(f"  ✗ {test_name}: FAILED")
            print(f"\nError: {type(e).__name__}")
            print(f"Message: {str(e)[:500]}")

            if "models_test" in str(e) or "ModuleNotFoundError" in str(e):
                print("\n" + "!"*70)
                print(f"BUG REPRODUCED AT {test_name}!")
                print("!"*70)

            import traceback
            traceback.print_exc()
            return

    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)


if __name__ == "__main__":
    test_level_1_simple.local()


@app.function(image=image)
def test_level_8_nested_stateful():
    """Level 8: Stateful object containing other stateful objects."""
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
    encoder = EncoderWithHints(model_name="inner-encoder")
    doc_lookup = {str(i): Document(f"doc{i}", i) for i in range(3)}
    reranker = Reranker2(encoder=encoder, doc_lookup=doc_lookup)

    print(f"Reranker contains encoder: {type(reranker._encoder)}")
    print(f"Encoder module: {reranker._encoder.__class__.__module__}")

    pipeline = Pipeline(
        nodes=[create_doc, rerank_doc, extract],
        engine=DaftEngine(collect=True, debug=True),
    )

    result = pipeline.run(inputs={"reranker": reranker})
    print(f"✓ Level 8 passed: {result}")
    return {"level": 8, "status": "pass"}
