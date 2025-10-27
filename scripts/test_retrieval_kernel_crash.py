#!/usr/bin/env python3
"""
Minimal test to reproduce the kernel crash with Pydantic models and DaftBackend.

This script tests the key issue: Daft serializing/deserializing Pydantic models with numpy arrays.
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
def encode_passage(passage: Any) -> dict:
    """Encode a single passage.
    
    KEY FIX: Accept Any and return dict (not Pydantic model).
    Daft will convert Pydantic models to dicts/structs internally.
    """
    # Normalize input (might be dict, Pydantic, or struct)
    if isinstance(passage, Passage):
        passage_obj = passage
    elif isinstance(passage, dict):
        passage_obj = Passage(**passage)
    else:
        # Handle struct/tuple formats
        passage_obj = Passage(uuid=getattr(passage, "uuid"), text=getattr(passage, "text"))
    
    # Create embedding
    embedding = np.array([len(passage_obj.text), 1.0, 2.0], dtype=np.float32)
    
    # Return as dict (Daft handles this better than Pydantic)
    return {
        "uuid": passage_obj.uuid,
        "text": passage_obj.text,
        "embedding": embedding
    }


@node(output_name="result")
def check_encoding(encoded_passages: Any) -> dict:
    """Check that encoding worked."""
    # Handle potential nesting from Daft
    if encoded_passages and isinstance(encoded_passages[0], list):
        encoded_passages = [item for sublist in encoded_passages for item in sublist]
    
    # Check first item
    first = encoded_passages[0]
    if isinstance(first, dict):
        return {"status": "success", "count": len(encoded_passages), "first_uuid": first["uuid"]}
    else:
        return {"status": "unknown_type", "count": len(encoded_passages), "type": str(type(first))}


def test_without_map_over():
    """Test 1: Without map_over (should work)"""
    print("\n" + "="*60)
    print("TEST 1: Without map_over (baseline)")
    print("="*60)
    
    @node(output_name="result")
    def encode_all(passages: List[Passage]) -> List[dict]:
        """Encode all passages at once."""
        return [
            {
                "uuid": p.uuid,
                "text": p.text,
                "embedding": np.array([len(p.text), 1.0, 2.0], dtype=np.float32)
            }
            for p in passages
        ]
    
    pipeline = Pipeline(
        nodes=[load_passages, encode_all],
        backend=DaftBackend(show_plan=False),
        name="no_map_over"
    )
    
    result = pipeline.run(inputs={"num_passages": 3})
    print(f"✅ Success! Encoded {len(result['result'])} passages")
    return True


def test_with_map_over_dict_return():
    """Test 2: With map_over, returning dict (should work)"""
    print("\n" + "="*60)
    print("TEST 2: With map_over, returning dict")
    print("="*60)
    
    encode_single = Pipeline(
        nodes=[encode_passage],
        name="encode_single"
    )
    
    encode_many = encode_single.as_node(
        input_mapping={"passages": "passage"},
        output_mapping={"encoded_passage": "encoded_passages"},
        map_over="passages",
        name="encode_many"
    )
    
    pipeline = Pipeline(
        nodes=[load_passages, encode_many, check_encoding],
        backend=DaftBackend(show_plan=False),
        name="with_map_over"
    )
    
    result = pipeline.run(inputs={"num_passages": 3})
    print(f"✅ Success! Result: {result['result']}")
    return True


def test_with_map_over_pydantic_return():
    """Test 3: With map_over, returning Pydantic model (might crash)"""
    print("\n" + "="*60)
    print("TEST 3: With map_over, returning Pydantic model")
    print("="*60)
    
    @node(output_name="encoded_passage")
    def encode_passage_pydantic(passage: Any) -> EncodedPassage:
        """Encode and return Pydantic model (problematic)."""
        if isinstance(passage, Passage):
            passage_obj = passage
        elif isinstance(passage, dict):
            passage_obj = Passage(**passage)
        else:
            passage_obj = Passage(uuid=getattr(passage, "uuid"), text=getattr(passage, "text"))
        
        embedding = np.array([len(passage_obj.text), 1.0, 2.0], dtype=np.float32)
        
        # Return Pydantic model - this might cause issues
        return EncodedPassage(
            uuid=passage_obj.uuid,
            text=passage_obj.text,
            embedding=embedding
        )
    
    encode_single = Pipeline(
        nodes=[encode_passage_pydantic],
        name="encode_single_pydantic"
    )
    
    encode_many = encode_single.as_node(
        input_mapping={"passages": "passage"},
        output_mapping={"encoded_passage": "encoded_passages"},
        map_over="passages",
        name="encode_many_pydantic"
    )
    
    pipeline = Pipeline(
        nodes=[load_passages, encode_many, check_encoding],
        backend=DaftBackend(show_plan=False),
        name="with_pydantic"
    )
    
    try:
        result = pipeline.run(inputs={"num_passages": 3})
        print(f"✅ Success! Result: {result['result']}")
        return True
    except Exception as e:
        print(f"❌ Failed with error: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("Testing Pydantic + numpy + DaftBackend Kernel Crash")
    print("="*60)
    
    # Run tests
    test1 = test_without_map_over()
    test2 = test_with_map_over_dict_return()
    test3 = test_with_map_over_pydantic_return()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Test 1 (no map_over): {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Test 2 (map_over + dict): {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Test 3 (map_over + Pydantic): {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if not test3:
        print("\n⚠️  KEY INSIGHT:")
        print("Returning Pydantic models with numpy arrays from map_over causes issues.")
        print("SOLUTION: Return dicts instead of Pydantic models in mapped functions.")
