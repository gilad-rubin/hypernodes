"""Minimal script to reproduce DaftEngine serialization bug with type-annotated custom classes.

This version uses explicit type annotations to better match the original error case.
"""

from typing import List
from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine


# Custom class used in type annotations (like the models in the original error)
class Document:
    """A document to process."""

    def __init__(self, text: str, id: int):
        self.text = text
        self.id = id

    def __repr__(self):
        return f"Document(id={self.id}, text='{self.text[:20]}...')"


# Processor class that has a method with type-annotated parameters
class DocumentProcessor:
    """A processor that explicitly type-hints the Document class."""

    def __init__(self, prefix: str = "PROCESSED"):
        self.prefix = prefix

    def process(self, doc: Document) -> Document:
        """Process a document - note the explicit Document type annotation."""
        return Document(
            text=f"{self.prefix}: {doc.text}",
            id=doc.id,
        )


# Node that accepts a list and maps over it
@node(output_name="documents")
def create_documents() -> List[Document]:
    """Create a list of documents."""
    return [
        Document("Hello World", 1),
        Document("Test Document", 2),
        Document("Another Doc", 3),
    ]


# Node that uses the processor with explicit type annotation
@node(output_name="processed_doc")
def process_document(document: Document, processor: DocumentProcessor) -> Document:
    """Process a single document using the processor."""
    return processor.process(document)


# Pipeline that uses map_over
def create_pipeline():
    """Create a pipeline with nested map operation."""
    inner_pipeline = Pipeline(
        nodes=[process_document],
        name="process_single",
    )

    # Use as_node with map_over to process multiple documents
    process_all = inner_pipeline.as_node(
        input_mapping={
            "document": "document",
            "processor": "processor",
        },
        output_mapping={
            "processed_doc": "result",
        },
        map_over="document",
    )

    outer_pipeline = Pipeline(
        nodes=[create_documents, process_all],
        engine=DaftEngine(collect=True),
        name="process_all_documents",
    )

    return outer_pipeline


def main():
    """Run the test case with map operations."""
    print("Creating DocumentProcessor instance...")
    processor = DocumentProcessor(prefix="PROCESSED")

    print("Building pipeline with map operations...")
    pipeline = create_pipeline()

    print("Running pipeline with DaftEngine...")
    inputs = {
        "processor": processor,  # Stateful parameter
    }

    try:
        result = pipeline.run(inputs=inputs)
        print(f"Success! Results:")
        for doc in result.get("result", []):
            print(f"  {doc}")
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"Error occurred: {type(e).__name__}")
        print(f"Error message: {e}")
        print(f"{'='*60}\n")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
