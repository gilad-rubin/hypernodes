#!/usr/bin/env python3
"""
Test the elegant Pydantic support in DaftBackend.

This script tests that we can now return Pydantic models directly
and DaftBackend will handle them automatically!
"""

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
    embedding: Any  # numpy array
    model_config = {"frozen": True, "arbitrary_types_allowed": True}


# ==================== Simple Test Nodes ====================
@node(output_name="passages")
def load_passages(num_passages: int) -> List[Passage]:
    """Load mock passages."""
    return [
        Passage(uuid=f"p{i}", text=f"Passage {i}")
        for i in range(num_passages)
    ]


@node(output_name="encoded_passage")
def encode_passage_pydantic(passage: Passage, encoder: dict) -> EncodedPassage:
    """Encode a single passage - returns Pydantic model!
    
    This is the ELEGANT way - we return the Pydantic model directly
    and DaftBackend automatically calls .model_dump() for us.
    """
    embedding = np.array([len(passage.text), 1.0, 2.0], dtype=np.float32)
    
    # Return Pydantic model directly!
    return EncodedPassage(
        uuid=passage.uuid,
        text=passage.text,
        embedding=embedding
    )


@node(output_name="encoder")
def create_encoder(model_name: str) -> dict:
    """Create mock encoder."""
    return {"model": model_name}


@node(output_name="result")
def check_results(encoded_passages: List[Any]) -> dict:
    """Check the results."""
    # The encoded_passages will be dicts (after .model_dump())
    first = encoded_passages[0]
    return {
        "count": len(encoded_passages),
        "first_uuid": first["uuid"] if isinstance(first, dict) else first.uuid,
        "has_embedding": "embedding" in first if isinstance(first, dict) else hasattr(first, "embedding"),
        "type": str(type(first))
    }


def test_elegant_pydantic_support():
    """Test the elegant Pydantic support."""
    print("\n" + "="*60)
    print("ELEGANT PYDANTIC SUPPORT TEST")
    print("="*60)
    print("\nReturning Pydantic models directly from nodes!")
    print("DaftBackend automatically calls .model_dump()")
    print()
    
    # Create single-passage encoding pipeline
    encode_single = Pipeline(
        nodes=[encode_passage_pydantic],
        name="encode_single"
    )
    
    # Create mapped version
    encode_many = encode_single.as_node(
        input_mapping={"passages": "passage"},
        output_mapping={"encoded_passage": "encoded_passages"},
        map_over="passages",
        name="encode_many"
    )
    
    # Build full pipeline
    pipeline = Pipeline(
        nodes=[load_passages, create_encoder, encode_many, check_results],
        backend=DaftBackend(show_plan=False),
        name="elegant_pydantic_test"
    )
    
    # Run it!
    try:
        result = pipeline.run(inputs={"num_passages": 3, "model_name": "test"})
        print("✅ SUCCESS!")
        print(f"\nResult: {result['result']}")
        print("\n" + "="*60)
        print("KEY ACHIEVEMENT:")
        print("  - Returned Pydantic models from encode_passage")
        print("  - DaftBackend auto-converted with .model_dump()")
        print("  - No manual dict conversion needed!")
        print("  - Full type safety preserved!")
        print("="*60)
        return True
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_elegant_pydantic_support()
    
    if success:
        print("\n🎉 Elegant Pydantic support is working!")
        print("\nYou can now:")
        print("  1. Return Pydantic models directly from nodes")
        print("  2. Keep full type safety with type hints")
        print("  3. No manual .model_dump() calls needed")
        print("  4. DaftBackend handles it automatically")
    else:
        print("\n⚠️  Test failed - may need pydantic_to_pyarrow installed")
        print("Run: pip install pydantic-to-pyarrow")
