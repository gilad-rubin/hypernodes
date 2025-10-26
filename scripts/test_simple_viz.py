"""Quick test to verify visualization fixes."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

from hypernodes import Pipeline, node

Vector = npt.NDArray[np.float32]

class Encoder:
    dim: int
    def encode(self, text: str, is_query: bool = False) -> Vector:
        return np.random.random(4).astype(np.float32)

@dataclass(frozen=True)
class Passage:
    pid: str
    text: str

@dataclass(frozen=True)
class EncodedPassage:
    pid: str
    text: str
    embedding: Vector

# ---- Core text encoding (reusable) ------------------------------------------
@node(output_name="cleaned_text")
def clean_text(text: str) -> str:
    return text.strip().lower()

@node(output_name="embedding")
def encode_text(encoder: Encoder, cleaned_text: str, is_query: bool = False) -> Vector:
    return encoder.encode(cleaned_text, is_query=is_query)

# Reusable text encoding pipeline
text_encode = Pipeline(nodes=[clean_text, encode_text], name="text_encode")

# ---- Passage encoding: extract -> encode -> pack ----------------------------
@node(output_name="text")
def extract_passage_text(passage: Passage) -> str:
    return passage.text

@node(output_name="encoded_passage")
def pack_passage(passage: Passage, embedding: Vector) -> EncodedPassage:
    return EncodedPassage(pid=passage.pid, text=passage.text, embedding=embedding)

# Single passage encoding pipeline
single_encode = Pipeline(nodes=[extract_passage_text, text_encode, pack_passage], name="single_encode")

print("Testing visualization...")
print(f"min_arg_group_size default should be None")
print(f"Visualizing single_encode pipeline...")
single_encode.visualize(filename="outputs/simple_test.svg")
print("✓ Done! Check outputs/simple_test.svg")
print("\nIf you still see groups, you need to restart your Jupyter kernel!")
