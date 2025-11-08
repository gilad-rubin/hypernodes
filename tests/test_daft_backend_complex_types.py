"""Tests for DaftBackend with complex types (Pydantic models, lists of objects, etc.)

These tests verify that DaftBackend can handle complex return types that Daft
cannot automatically infer, by falling back to Python object storage.
"""

import pytest
from typing import List

try:
    import daft
    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False

from hypernodes import node, Pipeline
from pydantic import BaseModel

if DAFT_AVAILABLE:
    from hypernodes.engines import DaftEngine

pytestmark = pytest.mark.skipif(not DAFT_AVAILABLE, reason="Daft not installed")


class Document(BaseModel):
    """A simple document model."""
    id: str
    text: str
    
    model_config = {"frozen": True}


class EncodedDocument(BaseModel):
    """A document with embedding."""
    id: str
    text: str
    embedding: List[float]
    
    model_config = {"frozen": True}


def test_daft_backend_list_of_pydantic_models():
    """Test DaftBackend with List[Pydantic] return type."""
    @node(output_name="documents")
    def create_documents(count: int) -> List[Document]:
        return [
            Document(id=f"doc_{i}", text=f"Document {i}")
            for i in range(count)
        ]
    
    pipeline = Pipeline(nodes=[create_documents], engine=DaftEngine())
    result = pipeline.run(inputs={"count": 3})
    
    assert "documents" in result
    docs = result["documents"]
    assert len(docs) == 3
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].id == "doc_0"
    assert docs[0].text == "Document 0"


def test_daft_backend_pydantic_to_pydantic():
    """Test DaftBackend with Pydantic input and output."""
    @node(output_name="documents")
    def create_documents(count: int) -> List[Document]:
        return [
            Document(id=f"doc_{i}", text=f"Document {i}")
            for i in range(count)
        ]
    
    @node(output_name="encoded_documents")
    def encode_documents(documents: List[Document]) -> List[EncodedDocument]:
        return [
            EncodedDocument(
                id=doc.id,
                text=doc.text,
                embedding=[1.0, 2.0, 3.0]
            )
            for doc in documents
        ]
    
    pipeline = Pipeline(
        nodes=[create_documents, encode_documents],
        engine=DaftEngine()
    )
    result = pipeline.run(inputs={"count": 2})
    
    assert "encoded_documents" in result
    encoded = result["encoded_documents"]
    assert len(encoded) == 2
    assert all(isinstance(d, EncodedDocument) for d in encoded)
    assert encoded[0].embedding == [1.0, 2.0, 3.0]


def test_daft_backend_map_with_pydantic():
    """Test DaftBackend map operation with Pydantic models."""
    @node(output_name="document")
    def create_document(text: str, idx: int) -> Document:
        return Document(id=f"doc_{idx}", text=text)
    
    @node(output_name="encoded")
    def encode_document(document: Document) -> EncodedDocument:
        return EncodedDocument(
            id=document.id,
            text=document.text,
            embedding=[float(len(document.text))]
        )
    
    pipeline = Pipeline(
        nodes=[create_document, encode_document],
        engine=DaftEngine()
    )
    
    results = pipeline.map(
        inputs={
            "text": ["Hello", "World", "Test"],
            "idx": [0, 1, 2]
        },
        map_over=["text", "idx"]
    )
    
    assert "encoded" in results
    encoded = results["encoded"]
    assert len(encoded) == 3
    assert all(isinstance(d, EncodedDocument) for d in encoded)
    assert encoded[0].text == "Hello"
    assert encoded[0].embedding == [5.0]  # len("Hello")


def test_daft_backend_dict_return_type():
    """Test DaftBackend with dict return type."""
    from typing import Dict, Any
    
    @node(output_name="config")
    def create_config(name: str, value: int) -> Dict[str, Any]:
        return {"name": name, "value": value, "computed": value * 2}
    
    pipeline = Pipeline(nodes=[create_config], engine=DaftEngine())
    result = pipeline.run(inputs={"name": "test", "value": 5})
    
    assert "config" in result
    config = result["config"]
    assert config["name"] == "test"
    assert config["value"] == 5
    assert config["computed"] == 10


def test_daft_backend_nested_list():
    """Test DaftBackend with nested list return type."""
    @node(output_name="matrix")
    def create_matrix(rows: int, cols: int) -> List[List[int]]:
        return [[i * cols + j for j in range(cols)] for i in range(rows)]
    
    pipeline = Pipeline(nodes=[create_matrix], engine=DaftEngine())
    result = pipeline.run(inputs={"rows": 2, "cols": 3})
    
    assert "matrix" in result
    matrix = result["matrix"]
    assert matrix == [[0, 1, 2], [3, 4, 5]]


def test_daft_backend_any_type():
    """Test DaftBackend with Any return type."""
    @node(output_name="result")
    def flexible_function(value: int) -> any:
        if value < 0:
            return None
        elif value == 0:
            return []
        elif value == 1:
            return {"single": True}
        else:
            return [i for i in range(value)]
    
    pipeline = Pipeline(nodes=[flexible_function], engine=DaftEngine())
    
    # Test different return types
    result = pipeline.run(inputs={"value": -1})
    assert result["result"] is None
    
    result = pipeline.run(inputs={"value": 0})
    assert result["result"] == []
    
    result = pipeline.run(inputs={"value": 1})
    assert result["result"] == {"single": True}
    
    result = pipeline.run(inputs={"value": 3})
    assert result["result"] == [0, 1, 2]


def test_daft_backend_mixed_simple_and_complex():
    """Test DaftBackend with mix of simple and complex types."""
    @node(output_name="documents")
    def create_documents(count: int) -> List[Document]:
        return [
            Document(id=f"doc_{i}", text=f"Document {i}")
            for i in range(count)
        ]
    
    @node(output_name="doc_count")
    def count_documents(documents: List[Document]) -> int:
        return len(documents)
    
    @node(output_name="first_text")
    def get_first_text(documents: List[Document]) -> str:
        return documents[0].text if documents else ""
    
    pipeline = Pipeline(
        nodes=[create_documents, count_documents, get_first_text],
        engine=DaftEngine()
    )
    result = pipeline.run(inputs={"count": 3})
    
    # Complex type
    assert len(result["documents"]) == 3
    # Simple types (should work with normal type inference)
    assert result["doc_count"] == 3
    assert result["first_text"] == "Document 0"
