"""Test that as_node(name=...) is prioritized over Pipeline(name=...) in visualization."""
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


@node(output_name="retrieved_docs")
def retrieve(query: str, vector_db, top_k: int = 5) -> list[Document]:
    return [Document(text="doc1", score=0.9)]


@node(output_name="answer")
def generate(query: str, retrieved_docs: list[Document], llm) -> Answer:
    return Answer(text="answer", sources=retrieved_docs)


@node(output_name="evaluation")
def evaluate_answer(answer: Answer, ground_truth: str, query: str) -> EvaluationResult:
    return EvaluationResult(
        query=query,
        generated_answer=answer.text,
        ground_truth=ground_truth,
        score=0.5,
    )


# Test 1: as_node(name="RAG") with NO Pipeline(name=...)
print("Test 1: as_node(name='RAG') with no Pipeline name")
rag_pipeline_unnamed = Pipeline(nodes=[retrieve, generate])
eval_pipeline_1 = Pipeline(
    nodes=[rag_pipeline_unnamed.as_node(name="RAG"), evaluate_answer]
)
viz1 = eval_pipeline_1.visualize(depth=2, return_type="graphviz")

# Check for "RAG" in cluster labels
has_rag_label = any("label=RAG" in line for line in viz1.body)
print(f"  ✓ 'RAG' found in visualization: {has_rag_label}")
assert has_rag_label, "Expected 'RAG' to appear as cluster label"

# Test 2: as_node(name="RAG") OVERRIDES Pipeline(name="inner_rag")
print("\nTest 2: as_node(name='RAG') overrides Pipeline(name='inner_rag')")
rag_pipeline_named = Pipeline(nodes=[retrieve, generate], name="inner_rag")
eval_pipeline_2 = Pipeline(
    nodes=[rag_pipeline_named.as_node(name="RAG"), evaluate_answer]
)
viz2 = eval_pipeline_2.visualize(depth=2, return_type="graphviz")

# Should show "RAG" (from as_node), NOT "inner_rag" (from Pipeline)
has_rag_label = any("label=RAG" in line for line in viz2.body)
has_inner_rag_label = any("label=inner_rag" in line for line in viz2.body)
print(f"  ✓ 'RAG' found in visualization: {has_rag_label}")
print(f"  ✓ 'inner_rag' NOT in visualization: {not has_inner_rag_label}")
assert has_rag_label, "Expected 'RAG' to appear as cluster label"
assert not has_inner_rag_label, "Expected 'inner_rag' to be overridden by 'RAG'"

# Test 3: Pipeline(name="MyPipeline") with NO as_node name
print("\nTest 3: Pipeline(name='MyPipeline') with no as_node name")
rag_pipeline_only_pipeline_name = Pipeline(nodes=[retrieve, generate], name="MyPipeline")
eval_pipeline_3 = Pipeline(
    nodes=[rag_pipeline_only_pipeline_name.as_node(), evaluate_answer]
)
viz3 = eval_pipeline_3.visualize(depth=2, return_type="graphviz")

# Should show "MyPipeline" (from Pipeline name)
has_my_pipeline_label = any("label=MyPipeline" in line for line in viz3.body)
print(f"  ✓ 'MyPipeline' found in visualization: {has_my_pipeline_label}")
assert has_my_pipeline_label, "Expected 'MyPipeline' to appear as cluster label"

# Test 4: No names at all - should show "pipeline"
print("\nTest 4: No names at all - should show 'pipeline'")
rag_pipeline_no_name = Pipeline(nodes=[retrieve, generate])
eval_pipeline_4 = Pipeline(nodes=[rag_pipeline_no_name.as_node(), evaluate_answer])
viz4 = eval_pipeline_4.visualize(depth=2, return_type="graphviz")

# Should show "pipeline" as fallback
has_pipeline_label = any("label=pipeline" in line for line in viz4.body)
print(f"  ✓ 'pipeline' found in visualization: {has_pipeline_label}")
assert has_pipeline_label, "Expected 'pipeline' to appear as cluster label"

print("\n✅ All tests passed! Name priority is: as_node(name=...) > Pipeline(name=...) > 'pipeline'")

