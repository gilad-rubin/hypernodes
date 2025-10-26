#!/usr/bin/env python3
"""Test caching with encoder-like objects."""

from hypernodes import Pipeline, node, DiskCache
from pydantic import BaseModel


class Passage(BaseModel):
    uuid: str
    text: str
    model_config = {"frozen": True}


class EncodedPassage(BaseModel):
    uuid: str
    text: str
    embedding: list[float]  # Simplified - using list instead of numpy
    model_config = {"frozen": True}


class DummyEncoder:
    """Simulates an encoder with internal state."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Simulate internal model state
        self._internal_counter = 0
    
    def encode(self, text: str) -> list[float]:
        """Fake encoding."""
        self._internal_counter += 1
        return [float(ord(c)) for c in text[:5]]  # Simple fake encoding


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
    cache=DiskCache(path=".cache/test_encoder_cache"),
    name="full_pipeline",
)

# Test data
passages = [
    Passage(uuid="p1", text="hello world"),
    Passage(uuid="p2", text="test passage"),
]

print("=" * 60)
print("FIRST RUN")
print("=" * 60)
results1 = full_pipeline.run(inputs={
    "model_name": "test-model",
    "passages": passages,
})
print(f"Encoded {len(results1['encoded_passages'])} passages")

print("\n" + "=" * 60)
print("SECOND RUN (should use cache)")
print("=" * 60)
results2 = full_pipeline.run(inputs={
    "model_name": "test-model",
    "passages": passages,
})
print(f"Encoded {len(results2['encoded_passages'])} passages")

print("\n" + "=" * 60)
print("CACHE CHECK")
print("=" * 60)
import os
if os.path.exists(".cache/test_encoder_cache"):
    print("✓ Cache directory exists")
    if os.path.exists(".cache/test_encoder_cache/meta.json"):
        import json
        with open(".cache/test_encoder_cache/meta.json") as f:
            meta = json.load(f)
        print(f"✓ Cache has {len(meta)} entries")
    else:
        print("✗ meta.json doesn't exist")
else:
    print("✗ Cache directory doesn't exist")

# Test hash consistency
print("\n" + "=" * 60)
print("ENCODER HASH TEST")
print("=" * 60)
from hypernodes.cache import hash_value

e1 = DummyEncoder("test-model")
e2 = DummyEncoder("test-model")

h1 = hash_value(e1)
h2 = hash_value(e2)

print(f"Encoder 1 hash: {h1[:32]}...")
print(f"Encoder 2 hash: {h2[:32]}...")
print(f"Hashes match: {h1 == h2}")
print(f"\nEncoder 1 attrs: {e1.__dict__}")
print(f"Encoder 2 attrs: {e2.__dict__}")
