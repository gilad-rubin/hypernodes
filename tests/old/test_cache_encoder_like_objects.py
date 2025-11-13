"""
Cache tests adapted from scripts/test_cache_encoder.py.

Uses dataclasses for deterministic encoder behavior and verifies that:
- First run executes nodes
- Second run hits cache for both the encoder creation and mapped encoding
"""

import tempfile
from dataclasses import dataclass
from typing import List

from hypernodes import DiskCache, Pipeline, PipelineCallback, node


class CacheEvents(PipelineCallback):
    def __init__(self):
        self.cached_nodes: List[str] = []
        self.map_item_cached = []

    def on_node_cached(self, node_id, signature, ctx):
        self.cached_nodes.append(node_id)

    def on_map_item_cached(self, item_index: int, signature: str, ctx):
        self.map_item_cached.append(item_index)


@dataclass(frozen=True)
class Passage:
    uuid: str
    text: str


@dataclass(frozen=True)
class EncodedPassage:
    uuid: str
    text: str
    embedding: List[float]


@dataclass(frozen=True)
class DeterministicEncoder:
    model_name: str

    def encode(self, text: str) -> List[float]:
        # Deterministic: first 5 codepoints
        return [float(ord(c)) for c in text[:5]]


def test_encoder_and_mapped_encoding_cache_hits_on_second_run():
    events = CacheEvents()

    @node(output_name="encoder")
    def create_encoder(model_name: str) -> DeterministicEncoder:
        return DeterministicEncoder(model_name)

    @node(output_name="encoded_passage")
    def encode_passage(passage: Passage, encoder: DeterministicEncoder) -> EncodedPassage:
        return EncodedPassage(
            uuid=passage.uuid,
            text=passage.text,
            embedding=encoder.encode(passage.text),
        )

    # Single-item encoding pipeline
    encode_single = Pipeline(nodes=[encode_passage], name="encode_single")

    # Mapped encoding node
    encode_mapped = encode_single.as_node(
        input_mapping={"passages": "passage"},
        output_mapping={"encoded_passage": "encoded_passages"},
        map_over="passages",
        name="encode_mapped",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        full_pipeline = Pipeline(
            nodes=[create_encoder, encode_mapped],
            cache=DiskCache(path=tmpdir),
            callbacks=[events],
            name="full_pipeline",
        )

        passages = [
            Passage(uuid="p1", text="hello world"),
            Passage(uuid="p2", text="test passage"),
        ]

        # First run: executes and populates cache
        out1 = full_pipeline.run(inputs={"model_name": "test-model", "passages": passages})
        assert "encoded_passages" in out1 and len(out1["encoded_passages"]) == 2
        assert events.cached_nodes == []
        events.cached_nodes.clear(); events.map_item_cached.clear()

        # Second run: should hit cache for encoder and for each mapped item
        out2 = full_pipeline.run(inputs={"model_name": "test-model", "passages": passages})
        assert out2 == out1

        # Ensure at least the encoder node and the mapped PipelineNode are cached
        cached_set = set(events.cached_nodes)
        assert "create_encoder" in cached_set
        assert "encode_mapped" in cached_set
        # Per-item cached signals are optional depending on caching layer
