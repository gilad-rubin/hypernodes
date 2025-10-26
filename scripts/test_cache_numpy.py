#!/usr/bin/env python3
"""Test caching with numpy arrays."""

from typing import Any
import numpy as np
from hypernodes import Pipeline, node, DiskCache
from pydantic import BaseModel


class Passage(BaseModel):
    uuid: str
    text: str
    model_config = {"frozen": True}


class EncodedPassage(BaseModel):
    uuid: str
    text: str
    embedding: Any  # Numpy array
    model_config = {"frozen": True, "arbitrary_types_allowed": True}


class DummyEncoder:
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def encode(self, text: str) -> np.ndarray:
        """Fake encoding with numpy."""
        # Create a deterministic embedding based on text
        values = [float(ord(c)) for c in text[:5]]
        return np.array(values, dtype=np.float32)


@node(output_name="encoder")
def create_encoder(model_name: str) -> DummyEncoder:
    """Create encoder."""
    print(f"Creating encoder with model: {model_name}")
    return DummyEncoder(model_name)


@node(output_name="encoded_passage")
def encode_passage(passage: Passage, encoder: DummyEncoder) -> EncodedPassage:
    """Encode a single passage."""
    print(f"  Encoding passage: {passage.uuid}")
    embedding = encoder.encode(passage.text)
    return EncodedPassage(uuid=passage.uuid, text=passage.text, embedding=embedding)


# Single-item encoding pipeline
encode_single = Pipeline(
    nodes=[encode_passage],
    name="encode_single",
)

# Mapped encoding node
encode_mapped = encode_single.as_node(
    input_mapping={"passages": "passage"},
    output_mapping={"encoded_passage": "encoded_passages"},
    map_over="passages",
    name="encode_mapped",
)

# Full pipeline
full_pipeline = Pipeline(
    nodes=[
        create_encoder,
        encode_mapped,
    ],
    cache=DiskCache(path=".cache/test_numpy_cache"),
    name="full_pipeline",
)

# Test data
passages = [
    Passage(uuid="p1", text="hello"),
    Passage(uuid="p2", text="world"),
]

print("=" * 60)
print("FIRST RUN")
print("=" * 60)
results1 = full_pipeline.run(inputs={
    "model_name": "test-model",
    "passages": passages,
})
print(f"Encoded {len(results1['encoded_passages'])} passages")
print(f"First embedding: {results1['encoded_passages'][0].embedding}")

print("\n" + "=" * 60)
print("SECOND RUN (should use cache)")
print("=" * 60)
results2 = full_pipeline.run(inputs={
    "model_name": "test-model",
    "passages": passages,
})
print(f"Encoded {len(results2['encoded_passages'])} passages")
print(f"First embedding: {results2['encoded_passages'][0].embedding}")

print("\n" + "=" * 60)
print("THIRD RUN WITH DIFFERENT TEXT (should NOT use cache)")
print("=" * 60)
passages_new = [
    Passage(uuid="p1", text="hello"),  # Same
    Passage(uuid="p2", text="earth"),  # Different!
]
results3 = full_pipeline.run(inputs={
    "model_name": "test-model",
    "passages": passages_new,
})
print(f"Encoded {len(results3['encoded_passages'])} passages")

print("\n" + "=" * 60)
print("CACHE CHECK")
print("=" * 60)
import os
if os.path.exists(".cache/test_numpy_cache"):
    print("✓ Cache directory exists")
    if os.path.exists(".cache/test_numpy_cache/meta.json"):
        import json
        with open(".cache/test_numpy_cache/meta.json") as f:
            meta = json.load(f)
        print(f"✓ Cache has {len(meta)} entries")
        print(f"  Expected: 1 encoder + 2 passages (run 1) + 0 (run 2, cached) + 1 new passage (run 3) = 4 total")
    else:
        print("✗ meta.json doesn't exist")
else:
    print("✗ Cache directory doesn't exist")
