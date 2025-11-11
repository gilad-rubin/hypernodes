"""Test ONLY serialization without map operations."""

import modal
from pathlib import Path

app = modal.App("test-serialization-only")

hypernodes_dir = Path("/Users/giladrubin/python_workspace/hypernodes/src/hypernodes")
script_file = Path(__file__)

modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("daft", "cloudpickle", "pydantic", "networkx", "tqdm", "rich", "graphviz")
    .add_local_dir(str(hypernodes_dir), remote_path="/root/hypernodes")
    .add_local_file(str(script_file), remote_path="/root/test_serialization_only.py")
)


# Classes defined in THIS script module (like your original)
class ColBERTEncoder:
    __daft_hint__ = "@daft.cls"
    __daft_use_process__ = True
    __daft_max_concurrency__ = 2
    __daft_gpus__ = 0

    def __init__(self, model_name: str, trust_remote_code: bool = True):
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code

    def encode(self, text: str, is_query: bool = False):
        return [0.1, 0.2, 0.3]


class RRFFusion:
    __daft_hint__ = "@daft.cls"

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, results_list):
        return results_list[0] if results_list else []


class NDCGEvaluator:
    __daft_hint__ = "@daft.cls"

    def __init__(self, k: int):
        self.k = k

    def compute(self, predictions, ground_truths):
        return 0.85


@app.function(image=modal_image)
def test_serialization():
    """Test serialization WITHOUT map operations."""
    import sys
    sys.path.insert(0, "/root")

    from hypernodes import Pipeline, node
    from hypernodes.engines import DaftEngine

    print("="*70)
    print("TESTING SERIALIZATION ONLY (no map operations)")
    print("="*70)

    @node(output_name="doc")
    def create_doc() -> dict:
        return {"text": "test doc"}

    @node(output_name="encoded")
    def encode_doc(doc: dict, encoder: ColBERTEncoder) -> dict:
        embedding = encoder.encode(doc["text"])
        return {"text": doc["text"], "embedding": embedding}

    @node(output_name="fused")
    def fuse(encoded: dict, rrf: RRFFusion) -> dict:
        return rrf.fuse([encoded])

    @node(output_name="score")
    def evaluate(fused: dict, evaluator: NDCGEvaluator) -> float:
        return evaluator.compute([fused], [])

    # Create stateful objects
    print("\nCreating stateful objects from script module...")
    encoder = ColBERTEncoder(model_name="test-model")
    rrf = RRFFusion(k=60)
    evaluator = NDCGEvaluator(k=20)

    print(f"  ColBERTEncoder.__module__ = {ColBERTEncoder.__module__}")
    print(f"  RRFFusion.__module__ = {RRFFusion.__module__}")
    print(f"  NDCGEvaluator.__module__ = {NDCGEvaluator.__module__}")

    pipeline = Pipeline(
        nodes=[create_doc, encode_doc, fuse, evaluate],
        engine=DaftEngine(collect=True, debug=True),
    )

    print("\nRunning pipeline with DaftEngine...")
    result = pipeline.run(inputs={
        "encoder": encoder,
        "rrf": rrf,
        "evaluator": evaluator,
    })

    print(f"\n✓ SUCCESS: {result}")
    return result


@app.local_entrypoint()
def main():
    print("Testing serialization of stateful objects from script module...")
    try:
        result = test_serialization.remote()
        print(f"\n{'='*70}")
        print("✓ SERIALIZATION TEST PASSED")
        print(f"{'='*70}")
        print(f"\nResult: {result}")
    except Exception as e:
        print(f"\n{'='*70}")
        print("✗ SERIALIZATION TEST FAILED")
        print(f"{'='*70}")
        print(f"\nError: {type(e).__name__}")
        print(f"Message: {str(e)[:500]}")
        
        if "ModuleNotFoundError" in str(e) and "test_serialization_only" in str(e):
            print("\n" + "!"*70)
            print("SERIALIZATION BUG REPRODUCED!")
            print("!"*70)
        else:
            print("\nThis is a different error, not the serialization bug")
        
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_serialization.local()
