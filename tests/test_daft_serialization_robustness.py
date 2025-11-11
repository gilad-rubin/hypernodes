"""
Tests for DaftEngine serialization robustness with custom classes.

This test ensures that custom classes from script modules can be properly
serialized and deserialized when using DaftEngine with use_process=True,
regardless of working directory context.

Regression test for: ModuleNotFoundError when running pipelines with
stateful objects containing custom classes.
"""

import sys
import pytest
from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine

# Skip if daft not available
pytest.importorskip("daft")


# =============================================================================
# Test Classes - Simulating script-module classes
# =============================================================================


class Document:
    """Document class for testing."""

    def __init__(self, text: str, doc_id: int = 0):
        self.text = text
        self.doc_id = doc_id

    def __repr__(self):
        return f"Document(text='{self.text[:20]}...', id={self.doc_id})"


class SimpleEncoder:
    """Simple encoder with typed methods."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, doc: Document) -> Document:
        """Method with custom type annotations."""
        encoded_text = f"[{self.model_name}] {doc.text}"
        return Document(encoded_text, doc.doc_id)


class NestedEncoder:
    """Encoder that contains another stateful object."""

    def __init__(self, inner_encoder: SimpleEncoder):
        self.inner_encoder = inner_encoder
        self.prefix = "NESTED"

    def encode(self, doc: Document) -> Document:
        """Uses nested encoder."""
        intermediate = self.inner_encoder.encode(doc)
        return Document(f"{self.prefix}: {intermediate.text}", doc.doc_id)


# =============================================================================
# Tests
# =============================================================================


def test_daft_simple_stateful_with_custom_types():
    """Test stateful object with custom type annotations."""

    @node(output_name="result")
    def process(text: str, encoder: SimpleEncoder) -> str:
        doc = Document(text)
        encoded = encoder.encode(doc)
        return encoded.text

    encoder = SimpleEncoder("test-model")

    pipeline = Pipeline(
        nodes=[process],
        engine=DaftEngine(collect=True, debug=False),
    )

    result = pipeline.run(inputs={"text": "hello", "encoder": encoder})

    assert result["result"] == "[test-model] hello"


def test_daft_nested_stateful_objects():
    """Test stateful object containing other stateful objects."""

    @node(output_name="result")
    def process(text: str, encoder: NestedEncoder) -> str:
        doc = Document(text)
        encoded = encoder.encode(doc)
        return encoded.text

    inner = SimpleEncoder("inner-model")
    encoder = NestedEncoder(inner)

    pipeline = Pipeline(
        nodes=[process],
        engine=DaftEngine(collect=True, debug=False),
    )

    result = pipeline.run(inputs={"text": "test", "encoder": encoder})

    assert result["result"] == "NESTED: [inner-model] test"


def test_daft_with_use_process():
    """Test serialization with use_process=True (multiprocessing)."""

    @node(output_name="result")
    def process(text: str, encoder: SimpleEncoder) -> str:
        doc = Document(text)
        encoded = encoder.encode(doc)
        return encoded.text

    # Force use_process to trigger serialization
    process.func.__daft_udf_config__ = {"use_process": True, "max_concurrency": 1}

    encoder = SimpleEncoder("process-model")

    pipeline = Pipeline(
        nodes=[process],
        engine=DaftEngine(collect=True, debug=False),
    )

    result = pipeline.run(inputs={"text": "multiprocess", "encoder": encoder})

    assert result["result"] == "[process-model] multiprocess"


def test_daft_multiple_stateful_objects():
    """Test multiple different stateful objects in same pipeline."""

    @node(output_name="doc")
    def create_doc(text: str) -> Document:
        return Document(text)

    @node(output_name="encoded1")
    def encode_first(doc: Document, encoder1: SimpleEncoder) -> Document:
        return encoder1.encode(doc)

    @node(output_name="encoded2")
    def encode_second(encoded1: Document, encoder2: SimpleEncoder) -> Document:
        return encoder2.encode(encoded1)

    @node(output_name="result")
    def extract(encoded2: Document) -> str:
        return encoded2.text

    encoder1 = SimpleEncoder("first")
    encoder2 = SimpleEncoder("second")

    pipeline = Pipeline(
        nodes=[create_doc, encode_first, encode_second, extract],
        engine=DaftEngine(collect=True, debug=False),
    )

    result = pipeline.run(
        inputs={"text": "test", "encoder1": encoder1, "encoder2": encoder2}
    )

    assert result["result"] == "[second] [first] test"


def test_daft_with_dict_containing_custom_objects():
    """Test stateful value that's a dict containing custom objects."""

    @node(output_name="result")
    def process(key: str, doc_lookup: dict) -> str:
        doc = doc_lookup[key]
        return doc.text

    doc_lookup = {
        "a": Document("doc a", 1),
        "b": Document("doc b", 2),
        "c": Document("doc c", 3),
    }

    pipeline = Pipeline(
        nodes=[process],
        engine=DaftEngine(collect=True, debug=False),
    )

    result = pipeline.run(inputs={"key": "b", "doc_lookup": doc_lookup})

    assert result["result"] == "doc b"


def test_daft_with_list_containing_custom_objects():
    """Test stateful value that's a list containing custom objects."""

    @node(output_name="result")
    def process(index: int, docs: list) -> str:
        return docs[index].text

    docs = [Document(f"doc {i}", i) for i in range(5)]

    pipeline = Pipeline(
        nodes=[process],
        engine=DaftEngine(collect=True, debug=False),
    )

    result = pipeline.run(inputs={"index": 2, "docs": docs})

    assert result["result"] == "doc 2"


def test_daft_simulated_unimportable_module():
    """
    Test with simulated unimportable module.

    This simulates the exact failure case where classes are from a script
    module that won't be available in worker processes.
    """

    # Create test classes with fake module
    class FakeDocument:
        def __init__(self, text: str):
            self.text = text

    class FakeEncoder:
        def __init__(self, model: str):
            self.model = model

        def encode(self, doc: FakeDocument) -> FakeDocument:
            return FakeDocument(f"[{self.model}] {doc.text}")

    # Simulate being from unimportable module
    FakeDocument.__module__ = "fake_unimportable_module"
    FakeEncoder.__module__ = "fake_unimportable_module"

    @node(output_name="result")
    def process(text: str, encoder: FakeEncoder) -> str:
        doc = FakeDocument(text)
        encoded = encoder.encode(doc)
        return encoded.text

    # Force use_process to trigger serialization
    process.func.__daft_udf_config__ = {"use_process": True, "max_concurrency": 1}

    encoder = FakeEncoder("fake-model")

    pipeline = Pipeline(
        nodes=[process],
        engine=DaftEngine(collect=True, debug=False),
    )

    # This should NOT raise ModuleNotFoundError
    result = pipeline.run(inputs={"text": "test", "encoder": encoder})

    assert result["result"] == "[fake-model] test"


def test_daft_nested_pipeline_with_stateful():
    """Test nested pipeline with stateful objects."""

    @node(output_name="encoded")
    def encode_doc(text: str, encoder: SimpleEncoder) -> str:
        doc = Document(text)
        encoded = encoder.encode(doc)
        return encoded.text

    inner = Pipeline(nodes=[encode_doc], name="inner")

    @node(output_name="texts")
    def create_texts() -> list:
        return ["text1", "text2", "text3"]

    encode_all = inner.as_node(
        input_mapping={"texts": "text", "encoder": "encoder"},
        output_mapping={"encoded": "encoded_texts"},
        map_over="texts",
    )

    @node(output_name="result")
    def join_texts(encoded_texts: list) -> str:
        return ", ".join(encoded_texts)

    outer = Pipeline(
        nodes=[create_texts, encode_all, join_texts],
        engine=DaftEngine(collect=True, debug=False),
    )

    encoder = SimpleEncoder("nested")

    result = outer.run(inputs={"encoder": encoder})

    expected = "[nested] text1, [nested] text2, [nested] text3"
    assert result["result"] == expected


def test_daft_with_class_methods():
    """Test stateful objects with various method signatures."""

    class ComplexEncoder:
        def __init__(self, prefix: str):
            self.prefix = prefix
            self.count = 0

        def encode(self, doc: Document) -> Document:
            self.count += 1
            return Document(f"{self.prefix}-{self.count}: {doc.text}", doc.doc_id)

        def get_count(self) -> int:
            return self.count

    @node(output_name="result")
    def process(text: str, encoder: ComplexEncoder) -> str:
        doc = Document(text)
        encoded = encoder.encode(doc)
        return f"{encoded.text} (count: {encoder.get_count()})"

    encoder = ComplexEncoder("test")

    pipeline = Pipeline(
        nodes=[process],
        engine=DaftEngine(collect=True, debug=False),
    )

    result = pipeline.run(inputs={"text": "hello", "encoder": encoder})

    assert "test-1: hello (count: 1)" in result["result"]
