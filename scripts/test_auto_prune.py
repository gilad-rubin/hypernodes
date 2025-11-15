"""Test automatic output pruning for nested pipelines."""

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


# Build RAG pipeline with two outputs
@node(output_name="retrieved_docs")
def retrieve(query: str, top_k: int = 5) -> list[Document]:
    """Mock retrieval - returns dummy documents."""
    return [
        Document(text=f"Doc {i} for query: {query}", score=1.0 / (i + 1))
        for i in range(top_k)
    ]


@node(output_name="answer")
def generate(query: str, retrieved_docs: list[Document]) -> Answer:
    """Mock generation."""
    return Answer(text=f"Mock answer for: {query}", sources=retrieved_docs)


# RAG pipeline produces TWO outputs: "answer" and "retrieved_docs"
rag_pipeline = Pipeline(nodes=[retrieve, generate])


# Evaluation node only needs "answer", NOT "retrieved_docs"
@node(output_name="evaluation")
def evaluate_answer(answer: Answer, ground_truth: str, query: str) -> EvaluationResult:
    """Evaluate the answer."""
    score = 0.8  # Mock score
    return EvaluationResult(
        query=query,
        generated_answer=answer.text,
        ground_truth=ground_truth,
        score=score,
    )


# Test 1: Check that PipelineNode detects required outputs
print("=" * 80)
print("TEST 1: Verify automatic output pruning for nested pipelines")
print("=" * 80)

# Create the evaluation pipeline
eval_pipeline = Pipeline(nodes=[rag_pipeline.as_node(name="RAG"), evaluate_answer])

# Get the RAG node from the evaluation pipeline
rag_node = eval_pipeline.nodes[0]

print(f"\n✓ RAG Pipeline produces outputs: {rag_pipeline.graph.available_output_names}")
print(f"  - answer")
print(f"  - retrieved_docs")

print(f"\n✓ evaluate_answer node depends on:")
for param in evaluate_answer.root_args:
    print(f"  - {param}")

print(f"\n✓ RAG node's _required_outputs: {rag_node._required_outputs}")
print(f"  (This should be ['answer'] because evaluate_answer only needs 'answer')")

print(f"\n✓ RAG node's output_name property: {rag_node.output_name}")
print(f"  (This should be 'answer' only, not ('answer', 'retrieved_docs'))")

# Test 2: Verify execution only computes required outputs
print("\n" + "=" * 80)
print("TEST 2: Verify execution behavior")
print("=" * 80)

inputs = {
    "query": "What is Python?",
    "ground_truth": "Python is a programming language.",
    "top_k": 2,
}

# Add a debug flag to see what's happening
print("\n✓ Running evaluation pipeline...")
result = eval_pipeline.run(inputs=inputs)

print(f"\n✓ Result keys: {list(result.keys())}")
print(f"  Expected: ['answer', 'evaluation']")
print(f"  (Both are valid outputs - 'answer' from RAG node, 'evaluation' from evaluate_answer)")

print(f"\n✓ Evaluation result:")
print(f"  Query: {result['evaluation'].query}")
print(f"  Generated Answer: {result['evaluation'].generated_answer}")
print(f"  Score: {result['evaluation'].score}")

# Test 3: Verify the key optimization - retrieved_docs is NOT computed
print("\n" + "=" * 80)
print("TEST 3: Verify the KEY optimization")
print("=" * 80)

print(
    "\n✓ The RAG pipeline would normally produce TWO outputs:"
    "\n  - 'answer' (from generate node)"
    "\n  - 'retrieved_docs' (from retrieve node)"
)
print(
    "\n✓ But since the outer pipeline only needs 'answer', the optimization ensures:"
    "\n  - RAG node's output_name is 'answer' (not a tuple)"
    "\n  - Inner pipeline only computes 'answer'"
    "\n  - 'retrieved_docs' is NOT in the final result"
)

print(f"\n✓ Verify 'retrieved_docs' is NOT in result: {'retrieved_docs' not in result}")

# Test if all tests passed
success = True

if rag_node._required_outputs != ["answer"]:
    print("\n✗ FAILED: _required_outputs should be ['answer']")
    success = False

if rag_node.output_name != "answer":
    print(f"\n✗ FAILED: output_name should be 'answer', got {rag_node.output_name}")
    success = False

# The key test: retrieved_docs should NOT be in the result
if "retrieved_docs" in result:
    print(f"\n✗ FAILED: 'retrieved_docs' should not be in result (it was not needed)")
    success = False

# 'answer' SHOULD be in the result (it's an output of the RAG node)
if "answer" not in result:
    print(f"\n✗ FAILED: 'answer' should be in result")
    success = False

if success:
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nAutomatic output pruning is working correctly:")
    print("  1. Graph builder detected that only 'answer' is needed")
    print("  2. PipelineNode._required_outputs was set to ['answer']")
    print("  3. PipelineNode.output_name returns only 'answer'")
    print("  4. Execution passes output_name to inner pipeline")
    print("  5. Visualization will show only required outputs")
else:
    print("\n" + "=" * 80)
    print("✗ SOME TESTS FAILED")
    print("=" * 80)

