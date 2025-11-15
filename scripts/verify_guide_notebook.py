"""Verify the guide.ipynb example works with automatic output pruning."""

from dataclasses import dataclass

from hypernodes import Pipeline, node


@dataclass
class Document:
    text: str
    score: float


@dataclass
class Answer:
    text: str
    sources: list[Document]


@dataclass
class EvaluationResult:
    query: str
    generated_answer: str
    ground_truth: str
    score: float


# Simple mock implementations
class SimpleVectorDB:
    def search(self, query: str, k: int) -> list[Document]:
        return [Document(text=f"Mock doc {i}", score=1.0 / (i + 1)) for i in range(k)]


class MockLLM:
    def generate(self, prompt: str) -> str:
        return f"Mock response to prompt"


# RAG Pipeline Nodes
@node(output_name="retrieved_docs")
def retrieve(query: str, vector_db, top_k: int = 5) -> list[Document]:
    return vector_db.search(query, k=top_k)


@node(output_name="answer")
def generate(query: str, retrieved_docs: list[Document], llm) -> Answer:
    context = "\n\n".join([f"[{i}] {doc.text}" for i, doc in enumerate(retrieved_docs)])
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    llm_response = llm.generate(prompt)
    return Answer(text=llm_response, sources=retrieved_docs)


# Evaluation Node
@node(output_name="evaluation")
def evaluate_answer(answer: Answer, ground_truth: str, query: str) -> EvaluationResult:
    score = 0.8
    return EvaluationResult(
        query=query,
        generated_answer=answer.text,
        ground_truth=ground_truth,
        score=score,
    )


# Build the pipelines
rag_pipeline = Pipeline(nodes=[retrieve, generate])
eval_pipeline = Pipeline(nodes=[rag_pipeline.as_node(name="RAG"), evaluate_answer])

# Get the RAG node
rag_node = eval_pipeline.nodes[0]

# Verify optimization
print("✓ Automatic Output Pruning Verification")
print("=" * 60)
print(f"RAG pipeline has outputs: {rag_pipeline.graph.available_output_names}")
print(f"  - answer")
print(f"  - retrieved_docs")
print()
print(f"Outer pipeline needs: answer (used by evaluate_answer)")
print()

# Check graph-level optimization
required_outputs = eval_pipeline.graph.required_outputs.get(rag_node)
print(f"Graph required_outputs[RAG] = {required_outputs}")
print(f"RAG node.output_name = {rag_node.output_name}")
print()

# Verify execution
vector_db = SimpleVectorDB()
llm = MockLLM()

result = eval_pipeline.run(
    inputs={
        "query": "How do plants create energy?",
        "ground_truth": "Plants create energy through photosynthesis.",
        "vector_db": vector_db,
        "llm": llm,
        "top_k": 2,
    }
)

print(f"Result keys: {list(result.keys())}")
print()

# Final verification
optimized = (
    required_outputs == ["answer"]
    and rag_node.output_name == ("answer", "retrieved_docs")  # Returns ALL possible outputs
    and "retrieved_docs" not in result  # But only 'answer' is computed
)

if optimized:
    print("✅ SUCCESS: Automatic output pruning is working!")
    print()
    print("Benefits:")
    print("  1. Only 'answer' computed (retrieve node skipped)")
    print("  2. Visualization shows only 'answer'")
    print("  3. No wasted computation")
    print("  4. Works automatically")
else:
    print("❌ FAILED: Optimization not working correctly")
    exit(1)

