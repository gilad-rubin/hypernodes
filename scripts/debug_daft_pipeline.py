#!/usr/bin/env python3
"""
Debug utility for Daft pipelines.

This script helps you debug Daft backend issues by:
1. Running with debug mode enabled
2. Showing detailed error messages
3. Allowing step-by-step execution
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from typing import Any, List

import numpy as np
from pydantic import BaseModel

from hypernodes import Pipeline, node
from hypernodes.daft_backend import DaftBackend


# ==================== Data Models ====================
class Passage(BaseModel):
    """A single document passage."""

    uuid: str
    text: str
    model_config = {"frozen": True}


class EncodedPassage(BaseModel):
    """A passage with its embedding."""

    uuid: str
    text: str
    embedding: Any
    model_config = {"frozen": True, "arbitrary_types_allowed": True}


class MockEncoder:
    """Mock encoder for testing."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, text: str, is_query: bool = False) -> np.ndarray:
        base_value = len(text)
        return np.array([base_value, base_value / 2], dtype=np.float32)


# ==================== Nodes ====================
@node(output_name="passages")
def load_passages(num_passages: int) -> List[Passage]:
    """Load passages."""
    print(f"  → Loading {num_passages} passages...")
    return [
        Passage(uuid=f"p{i}", text=f"Passage text {i}") for i in range(num_passages)
    ]


@node(output_name="encoder")
def create_encoder(model_name: str) -> MockEncoder:
    """Create encoder."""
    print(f"  → Creating encoder: {model_name}")
    return MockEncoder(model_name)


@node(output_name="encoded_passage")
def encode_passage(passage: Passage, encoder: MockEncoder) -> EncodedPassage:
    """Encode a single passage."""
    print(f"  → Encoding passage: {passage.uuid} (type: {type(passage).__name__})")
    embedding = encoder.encode(passage.text)
    result = EncodedPassage(uuid=passage.uuid, text=passage.text, embedding=embedding)
    print(f"  ← Encoded: {result.uuid} (type: {type(result).__name__})")
    return result


@node(output_name="results")
def check_results(encoded_passages: List[Any]) -> dict:
    """Check results."""
    print(f"  → Checking {len(encoded_passages)} encoded passages...")
    first = encoded_passages[0]
    print(f"    First passage type: {type(first)}")
    if isinstance(first, dict):
        print(f"    First passage keys: {first.keys()}")
        uuid = first["uuid"]
    else:
        uuid = first.uuid
    print(f"    First passage UUID: {uuid}")
    return {"count": len(encoded_passages), "first_uuid": uuid}


def main():
    """Run debug pipeline."""
    print("\n" + "=" * 70)
    print("DAFT BACKEND DEBUG MODE")
    print("=" * 70)
    print()

    # Build pipeline
    encode_single = Pipeline(nodes=[encode_passage], name="encode_single")

    encode_many = encode_single.as_node(
        input_mapping={"passages": "passage"},
        output_mapping={"encoded_passage": "encoded_passages"},
        map_over="passages",
        name="encode_many",
    )

    pipeline = Pipeline(
        nodes=[load_passages, create_encoder, encode_many, check_results],
        backend=DaftBackend(show_plan=True, debug=True),  # DEBUG MODE ENABLED
        name="debug_pipeline",
    )

    # Run with minimal data
    inputs = {"num_passages": 3, "model_name": "test-model"}

    print("Running pipeline with debug mode enabled...")
    print()

    try:
        result = pipeline.run(output_name="results", inputs=inputs)
        print("\n" + "=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print(f"\nResult: {result}")
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ ERROR!")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
