"""Test models for progressive complexity testing."""

from typing import List


# Level 1: Simple class
class Document:
    """Simple document class."""
    def __init__(self, text: str, doc_id: int):
        self.text = text
        self.doc_id = doc_id


# Level 2: Stateful class with typed methods
class Encoder:
    """Encoder with typed methods."""
    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, doc: Document) -> Document:
        """Encode document - uses Document in signature."""
        return Document(f"[{self.model_name}] {doc.text}", doc.doc_id)


# Level 4: With __daft_hint__
class EncoderWithHints:
    """Encoder with Daft hints."""
    __daft_hint__ = "@daft.cls"
    __daft_use_process__ = True
    __daft_max_concurrency__ = 2

    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, doc: Document) -> Document:
        """Encode document - uses Document in signature."""
        return Document(f"[{self.model_name}] {doc.text}", doc.doc_id)


# Level 5: Another stateful class
class Fusion:
    """Fusion class."""
    __daft_hint__ = "@daft.cls"

    def __init__(self, strategy: str):
        self.strategy = strategy

    def fuse(self, doc1: Document, doc2: Document) -> Document:
        """Fuse documents."""
        return Document(f"{doc1.text} | {doc2.text}", doc1.doc_id)


# Level 6: More complex types
class Hit:
    """Search hit."""
    def __init__(self, doc_id: int, score: float):
        self.doc_id = doc_id
        self.score = score


class Ranker:
    """Ranker with List[Hit] in signature."""
    __daft_hint__ = "@daft.cls"

    def __init__(self, strategy: str):
        self.strategy = strategy

    def rank(self, hits: List[Hit]) -> List[Hit]:
        """Rank hits - uses List[Hit] in signature."""
        return sorted(hits, key=lambda h: h.score, reverse=True)


# Level 7: Dict annotations
class Reranker:
    """Reranker with dict[str, Document] in signature."""
    __daft_hint__ = "@daft.cls"
    __daft_use_process__ = True

    def __init__(self, model: str):
        self.model = model

    def rerank(self, hits: List[Hit], doc_lookup: dict) -> List[Hit]:
        """Rerank hits - uses dict parameter."""
        # Just return top 2
        return hits[:2]


# Level 8: Stateful object containing other custom class instances
class Reranker2:
    """Reranker that contains encoder as attribute."""
    __daft_hint__ = "@daft.cls"
    __daft_use_process__ = True

    def __init__(self, encoder: EncoderWithHints, doc_lookup: dict):
        self._encoder = encoder  # Contains another stateful object!
        self._doc_lookup = doc_lookup

    def rerank(self, doc: Document) -> Document:
        """Rerank using the internal encoder."""
        return self._encoder.encode(doc)
