"""Example demonstrating DaftBackend with complex types (Pydantic models, Lists, etc.)

This example shows how DaftBackend automatically handles complex return types
that Daft cannot natively infer, by using Python object storage.
"""

from typing import List, Dict, Any
from pydantic import BaseModel
from hypernodes import node, Pipeline
from hypernodes.daft_backend import DaftBackend

print("=" * 60)
print("DaftBackend with Complex Types Example")
print("=" * 60)

# Define Pydantic models
class Document(BaseModel):
    """A simple document."""
    id: str
    text: str
    
    model_config = {"frozen": True}


class EncodedDocument(BaseModel):
    """A document with embedding."""
    id: str
    text: str
    embedding: List[float]
    
    model_config = {"frozen": True}


# Example 1: List of Pydantic Models
print("\n1. List of Pydantic Models")
print("-" * 40)

@node(output_name="documents")
def create_documents(texts: List[str]) -> List[Document]:
    """Create documents from texts."""
    return [Document(id=f"doc_{i}", text=text) for i, text in enumerate(texts)]

@node(output_name="count")
def count_documents(documents: List[Document]) -> int:
    """Count documents."""
    return len(documents)

pipeline = Pipeline(
    nodes=[create_documents, count_documents],
    backend=DaftBackend(),
    name="document_pipeline"
)

result = pipeline.run(inputs={"texts": ["Hello", "World", "Test"]})
print(f"Created {result['count']} documents:")
for doc in result['documents']:
    print(f"  - {doc.id}: {doc.text}")


# Example 2: Pydantic to Pydantic Transformation
print("\n\n2. Pydantic to Pydantic Transformation")
print("-" * 40)

@node(output_name="documents")
def load_documents(count: int) -> List[Document]:
    """Load documents."""
    return [Document(id=f"doc_{i}", text=f"Document {i}") for i in range(count)]

@node(output_name="encoded_documents")
def encode_documents(documents: List[Document]) -> List[EncodedDocument]:
    """Encode documents with fake embeddings."""
    return [
        EncodedDocument(
            id=doc.id,
            text=doc.text,
            embedding=[float(len(doc.text)), 1.0, 2.0]
        )
        for doc in documents
    ]

@node(output_name="avg_embedding_size")
def compute_avg_embedding(encoded_documents: List[EncodedDocument]) -> float:
    """Compute average first dimension of embeddings."""
    return sum(doc.embedding[0] for doc in encoded_documents) / len(encoded_documents)

pipeline = Pipeline(
    nodes=[load_documents, encode_documents, compute_avg_embedding],
    backend=DaftBackend(),
    name="encoding_pipeline"
)

result = pipeline.run(inputs={"count": 3})
print(f"Encoded {len(result['encoded_documents'])} documents")
print(f"Average embedding size: {result['avg_embedding_size']:.2f}")
for doc in result['encoded_documents']:
    print(f"  - {doc.id}: embedding={doc.embedding}")


# Example 3: Dict Return Types
print("\n\n3. Dict Return Types")
print("-" * 40)

@node(output_name="config")
def create_config(model_name: str, batch_size: int) -> Dict[str, Any]:
    """Create configuration dict."""
    return {
        "model_name": model_name,
        "batch_size": batch_size,
        "max_length": batch_size * 10,
        "device": "cuda"
    }

@node(output_name="summary")
def summarize_config(config: Dict[str, Any]) -> str:
    """Summarize configuration."""
    return f"Model: {config['model_name']}, Batch: {config['batch_size']}, Device: {config['device']}"

pipeline = Pipeline(
    nodes=[create_config, summarize_config],
    backend=DaftBackend(),
    name="config_pipeline"
)

result = pipeline.run(inputs={"model_name": "bert-base", "batch_size": 32})
print(f"Config: {result['config']}")
print(f"Summary: {result['summary']}")


# Example 4: Map with Pydantic Models
print("\n\n4. Map with Pydantic Models")
print("-" * 40)

@node(output_name="document")
def create_single_document(text: str, idx: int) -> Document:
    """Create a single document."""
    return Document(id=f"doc_{idx}", text=text)

@node(output_name="encoded")
def encode_single_document(document: Document) -> EncodedDocument:
    """Encode a single document."""
    # Fake embedding based on text length
    text_len = float(len(document.text))
    return EncodedDocument(
        id=document.id,
        text=document.text,
        embedding=[text_len, text_len / 2, text_len / 4]
    )

pipeline = Pipeline(
    nodes=[create_single_document, encode_single_document],
    backend=DaftBackend(),
    name="single_doc_pipeline"
)

# Map over multiple texts
texts = ["Short", "Medium length", "Very long text here"]
results = pipeline.map(
    inputs={
        "text": texts,
        "idx": list(range(len(texts)))
    },
    map_over=["text", "idx"]
)

print(f"Processed {len(results['encoded'])} documents:")
for doc in results['encoded']:
    print(f"  - {doc.id}: '{doc.text}' -> embedding={[f'{x:.1f}' for x in doc.embedding]}")


# Example 5: Nested Lists
print("\n\n5. Nested Lists")
print("-" * 40)

@node(output_name="matrix")
def create_matrix(rows: int, cols: int) -> List[List[int]]:
    """Create a matrix."""
    return [[i * cols + j for j in range(cols)] for i in range(rows)]

@node(output_name="flattened")
def flatten_matrix(matrix: List[List[int]]) -> List[int]:
    """Flatten matrix."""
    return [item for row in matrix for item in row]

@node(output_name="sum")
def sum_values(flattened: List[int]) -> int:
    """Sum all values."""
    return sum(flattened)

pipeline = Pipeline(
    nodes=[create_matrix, flatten_matrix, sum_values],
    backend=DaftBackend(),
    name="matrix_pipeline"
)

result = pipeline.run(inputs={"rows": 3, "cols": 4})
print(f"Matrix: {result['matrix']}")
print(f"Flattened: {result['flattened']}")
print(f"Sum: {result['sum']}")


print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)
print("\nKey Takeaway:")
print("DaftBackend automatically detects complex return types")
print("(List[Pydantic], Dict, nested structures) and uses")
print("Python object storage, making it work seamlessly with")
print("any HyperNodes pipeline without code changes!")
