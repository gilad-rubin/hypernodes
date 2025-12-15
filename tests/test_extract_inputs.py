"""Tests for field extraction from Pydantic models and dataclasses."""

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from hypernodes import Pipeline, node


class Document(BaseModel):
    """Pydantic model for testing extraction."""

    file_path: str
    document_type: str
    author: str
    created_at: str
    content: str


@dataclass
class DataclassDocument:
    """Dataclass for testing extraction."""

    file_path: str
    document_type: str
    author: str


class TestExtractDecorator:
    """Tests for the extract parameter on @node decorator."""

    def test_basic_extraction_pydantic(self):
        """Test extracting fields from a Pydantic model."""

        @node(output_name="result", extract={"doc": ["file_path", "document_type"]})
        def process(file_path: str, document_type: str) -> dict:
            return {"path": file_path, "type": document_type}

        doc = Document(
            file_path="/path/to/file.pdf",
            document_type="pdf",
            author="John",
            created_at="2024-01-01",
            content="Hello world",
        )

        result = process(doc=doc)
        assert result == {"path": "/path/to/file.pdf", "type": "pdf"}

    def test_basic_extraction_dataclass(self):
        """Test extracting fields from a dataclass."""

        @node(output_name="result", extract={"doc": ["file_path", "document_type"]})
        def process(file_path: str, document_type: str) -> dict:
            return {"path": file_path, "type": document_type}

        doc = DataclassDocument(
            file_path="/path/to/file.txt",
            document_type="text",
            author="Jane",
        )

        result = process(doc=doc)
        assert result == {"path": "/path/to/file.txt", "type": "text"}

    def test_extraction_with_passthrough_param(self):
        """Test extraction with additional non-extracted parameters."""

        @node(output_name="result", extract={"doc": ["file_path"]})
        def process(file_path: str, multiplier: int) -> str:
            return file_path * multiplier

        doc = Document(
            file_path="x",
            document_type="pdf",
            author="John",
            created_at="2024-01-01",
            content="Hello",
        )

        result = process(doc=doc, multiplier=3)
        assert result == "xxx"

    def test_root_args_with_extraction(self):
        """Test that root_args reflects the extraction mapping."""

        @node(output_name="result", extract={"doc": ["file_path", "document_type"]})
        def process(file_path: str, document_type: str, extra: int) -> dict:
            return {}

        # root_args should be: source param "doc" + non-extracted param "extra"
        assert "doc" in process.root_args
        assert "extra" in process.root_args
        assert "file_path" not in process.root_args
        assert "document_type" not in process.root_args

    def test_node_still_callable_directly(self):
        """Test that node can still be called with direct params (no extraction)."""

        @node(output_name="result", extract={"doc": ["file_path", "document_type"]})
        def process(file_path: str, document_type: str) -> dict:
            return {"path": file_path, "type": document_type}

        # Direct call with explicit params should still work
        result = process(file_path="/direct/path", document_type="direct")
        assert result == {"path": "/direct/path", "type": "direct"}


class TestWithExtraction:
    """Tests for the with_extraction() method."""

    def test_with_extraction_creates_new_node(self):
        """Test that with_extraction returns a new adapted node."""

        @node(output_name="result")
        def process(file_path: str, document_type: str) -> dict:
            return {"path": file_path, "type": document_type}

        # Original node has no extraction
        assert process._extract == {}

        # Create adapted version
        adapted = process.with_extraction(doc=["file_path", "document_type"])

        # Original is unchanged
        assert process._extract == {}

        # Adapted has extraction
        assert adapted._extract == {"doc": ["file_path", "document_type"]}

    def test_with_extraction_works(self):
        """Test that adapted node works with source objects."""

        @node(output_name="result")
        def process(file_path: str, document_type: str) -> dict:
            return {"path": file_path, "type": document_type}

        adapted = process.with_extraction(doc=["file_path", "document_type"])

        doc = Document(
            file_path="/adapted/path",
            document_type="md",
            author="Adapted",
            created_at="2024-01-01",
            content="Test",
        )

        result = adapted(doc=doc)
        assert result == {"path": "/adapted/path", "type": "md"}

    def test_with_extraction_original_still_works(self):
        """Test that original node still works with direct params."""

        @node(output_name="result")
        def process(file_path: str, document_type: str) -> dict:
            return {"path": file_path, "type": document_type}

        adapted = process.with_extraction(doc=["file_path", "document_type"])

        # Original still works with direct params
        result = process(file_path="/original", document_type="orig")
        assert result == {"path": "/original", "type": "orig"}

    def test_with_extraction_updates_root_args(self):
        """Test that adapted node has correct root_args."""

        @node(output_name="result")
        def process(file_path: str, document_type: str) -> dict:
            return {}

        adapted = process.with_extraction(doc=["file_path", "document_type"])

        assert "doc" in adapted.root_args
        assert "file_path" not in adapted.root_args


class TestExtractionInPipeline:
    """Tests for extraction working within pipelines."""

    def test_pipeline_with_extraction(self):
        """Test that extraction works in pipeline execution."""

        @node(output_name="parsed", extract={"doc": ["file_path", "document_type"]})
        def parse(file_path: str, document_type: str) -> dict:
            return {"path": file_path, "type": document_type}

        @node(output_name="result")
        def process(parsed: dict) -> str:
            return f"{parsed['type']}:{parsed['path']}"

        pipeline = Pipeline(nodes=[parse, process])

        doc = Document(
            file_path="/pipeline/file.pdf",
            document_type="pdf",
            author="Test",
            created_at="2024-01-01",
            content="Test content",
        )

        result = pipeline.run(inputs={"doc": doc})
        assert result["result"] == "pdf:/pipeline/file.pdf"

    def test_pipeline_map_with_extraction(self):
        """Test that extraction works with pipeline.map()."""

        @node(output_name="parsed", extract={"doc": ["file_path"]})
        def parse(file_path: str) -> str:
            return file_path.upper()

        pipeline = Pipeline(nodes=[parse])

        docs = [
            Document(
                file_path="/file1.pdf",
                document_type="pdf",
                author="A",
                created_at="2024-01-01",
                content="",
            ),
            Document(
                file_path="/file2.txt",
                document_type="txt",
                author="B",
                created_at="2024-01-02",
                content="",
            ),
        ]

        results = pipeline.map(inputs={"doc": docs}, map_over="doc")
        assert results[0]["parsed"] == "/FILE1.PDF"
        assert results[1]["parsed"] == "/FILE2.TXT"

    def test_adapted_node_in_pipeline(self):
        """Test using with_extraction adapted node in pipeline."""

        @node(output_name="parsed")
        def parse(file_path: str, document_type: str) -> dict:
            return {"path": file_path, "type": document_type}

        adapted_parse = parse.with_extraction(doc=["file_path", "document_type"])

        pipeline = Pipeline(nodes=[adapted_parse])

        doc = Document(
            file_path="/adapted/pipeline.pdf",
            document_type="pdf",
            author="Test",
            created_at="2024-01-01",
            content="",
        )

        result = pipeline.run(inputs={"doc": doc})
        assert result["parsed"] == {"path": "/adapted/pipeline.pdf", "type": "pdf"}


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_multiple_source_objects(self):
        """Test extracting from multiple source objects."""

        @node(
            output_name="result",
            extract={"doc1": ["file_path"], "doc2": ["document_type"]},
        )
        def combine(file_path: str, document_type: str) -> str:
            return f"{document_type}:{file_path}"

        doc1 = Document(
            file_path="/path1",
            document_type="ignored",
            author="A",
            created_at="2024-01-01",
            content="",
        )
        doc2 = Document(
            file_path="/ignored",
            document_type="pdf",
            author="B",
            created_at="2024-01-02",
            content="",
        )

        result = combine(doc1=doc1, doc2=doc2)
        assert result == "pdf:/path1"

    def test_repr_with_extraction(self):
        """Test that repr includes extraction info."""

        @node(output_name="result", extract={"doc": ["file_path"]})
        def process(file_path: str) -> str:
            return file_path

        repr_str = repr(process)
        assert "extract=" in repr_str
        assert "doc" in repr_str
