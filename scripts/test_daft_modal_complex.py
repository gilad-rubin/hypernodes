"""
More complex test matching the original failure scenario with multiple stateful objects.

This mimics the structure of your hebrew retrieval pipeline with:
- Multiple custom classes (Encoder, RRF, NDCGEvaluator, etc.)
- Nested pipelines
- Map operations
- Multiple stateful parameters
"""

import modal

# Create Modal app
app = modal.App("test-daft-complex")

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
# CUSTOM CLASSES (mimicking the original models)
# ============================================================================


class Passage:
    """Document passage."""
    def __init__(self, text: str, id: int):
        self.text = text
        self.id = id


class Query:
    """Query."""
    def __init__(self, text: str):
        self.text = text


class Hits:
    """Search hits."""
    def __init__(self, ids: list):
        self.ids = ids


class Encoder:
    """Encoder with typed methods."""
    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode_passage(self, passage: Passage) -> Passage:
        """Encode passage - uses Passage in signature."""
        return Passage(f"ENC[{passage.text}]", passage.id)

    def encode_query(self, query: Query) -> Query:
        """Encode query - uses Query in signature."""
        return Query(f"ENC[{query.text}]")


class RRFFusion:
    """RRF fusion ranker."""
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, hits1: Hits, hits2: Hits) -> Hits:
        """Fuse hits - uses Hits in signature."""
        combined = list(set(hits1.ids + hits2.ids))
        return Hits(combined[:10])


class NDCGEvaluator:
    """NDCG evaluator."""
    def __init__(self, k: int = 10):
        self.k = k

    def evaluate(self, predictions: list, ground_truth: list) -> float:
        """Evaluate NDCG score."""
        return 0.85  # Mock score


# ============================================================================
# MODAL FUNCTION
# ============================================================================


@app.function(image=image)
def run_complex_test():
    """Run complex test with multiple stateful objects."""
    import sys
    sys.path.insert(0, "/root")

    from hypernodes import Pipeline, node
    from hypernodes.engines import DaftEngine

    print("="*70)
    print("Running complex test on Modal")
    print("="*70)
    print(f"\nEncoder.__module__ = '{Encoder.__module__}'")
    print(f"Passage.__module__ = '{Passage.__module__}'")
    print(f"Query.__module__ = '{Query.__module__}'")
    print(f"Hits.__module__ = '{Hits.__module__}'")

    # Nodes with stateful parameters (like your retrieval pipeline)
    @node(output_name="passages")
    def load_passages() -> list:
        return [Passage("passage 1", 1), Passage("passage 2", 2)]

    @node(output_name="queries")
    def load_queries() -> list:
        return [Query("query 1"), Query("query 2")]

    @node(output_name="encoded_passage")
    def encode_passage(passage: Passage, encoder: Encoder) -> Passage:
        """Uses encoder with Passage-typed method."""
        return encoder.encode_passage(passage)

    @node(output_name="encoded_query")
    def encode_query(query: Query, encoder: Encoder) -> Query:
        """Uses encoder with Query-typed method."""
        return encoder.encode_query(query)

    @node(output_name="hits1")
    def search1(query: Query) -> Hits:
        return Hits([1, 2, 3])

    @node(output_name="hits2")
    def search2(query: Query) -> Hits:
        return Hits([2, 3, 4])

    @node(output_name="fused_hits")
    def fuse(hits1: Hits, hits2: Hits, rrf: RRFFusion) -> Hits:
        """Uses RRF with Hits-typed method."""
        return rrf.fuse(hits1, hits2)

    @node(output_name="score")
    def evaluate(predictions: list, ground_truth: list, evaluator: NDCGEvaluator) -> float:
        """Uses evaluator."""
        return evaluator.evaluate(predictions, ground_truth)

    # Force use_process for some nodes
    encode_passage.func.__daft_udf_config__ = {"use_process": True, "max_concurrency": 2}
    fuse.func.__daft_udf_config__ = {"use_process": True}

    # Create stateful objects
    print("\nCreating stateful objects...")
    encoder = Encoder("test-model")
    rrf = RRFFusion(k=60)
    evaluator = NDCGEvaluator(k=10)

    # Build pipeline
    print("Building pipeline with nested structure...")

    # Inner pipeline for encoding a single passage
    encode_passage_pipeline = Pipeline(
        nodes=[encode_passage],
        name="encode_single_passage",
    )

    encode_all_passages = encode_passage_pipeline.as_node(
        input_mapping={"passage": "passage", "encoder": "encoder"},
        output_mapping={"encoded_passage": "encoded_passages"},
        map_over="passage",
    )

    # Main pipeline
    pipeline = Pipeline(
        nodes=[
            load_passages,
            load_queries,
            encode_all_passages,
        ],
        engine=DaftEngine(collect=True, debug=True),
        name="main_pipeline",
    )

    print("\nRunning pipeline on Modal with use_process=True...")
    print("(Multiple stateful objects with typed methods)")
    print("(Nested pipelines with map operations)\n")

    try:
        result = pipeline.run(inputs={
            "encoder": encoder,
        })

        print("\n" + "="*70)
        print(f"✓ SUCCESS")
        print(f"Encoded passages: {len(result.get('encoded_passages', []))}")
        print("="*70)
        return {"status": "success", "result": "passed"}

    except Exception as e:
        print("\n" + "="*70)
        print(f"✗ FAILED: {type(e).__name__}")
        print("="*70)
        print(f"\nError: {str(e)[:800]}")

        if "test_daft_modal_complex" in str(e) or "ModuleNotFoundError" in str(e):
            print("\n" + "!"*70)
            print("BUG REPRODUCED ON MODAL!")
            print("!"*70)

        import traceback
        traceback.print_exc()
        raise


@app.local_entrypoint()
def main():
    """Local entrypoint."""
    print("Starting complex Modal test...")
    result = run_complex_test.remote()
    print(f"\nFinal result: {result}")


if __name__ == "__main__":
    run_complex_test.local()
