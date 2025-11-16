"""Test script to verify the depth=2 visualization fix."""
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
    """Retrieve most relevant documents from vector database."""
    return [Document(text="doc1", score=0.9)]


@node(output_name="answer")
def generate(query: str, retrieved_docs: list[Document], llm) -> Answer:
    """Generate answer using LLM with retrieved context."""
    return Answer(text="answer", sources=retrieved_docs)


@node(output_name="evaluation")
def evaluate_answer(answer: Answer, ground_truth: str, query: str) -> EvaluationResult:
    """Simple evaluation."""
    return EvaluationResult(
        query=query,
        generated_answer=answer.text,
        ground_truth=ground_truth,
        score=0.5,
    )


# Build RAG pipeline
rag_pipeline = Pipeline(nodes=[retrieve, generate])

# Use RAG pipeline as a node in evaluation pipeline
eval_pipeline = Pipeline(nodes=[rag_pipeline.as_node(name="RAG"), evaluate_answer])

# Test visualization with depth=2
print("Testing visualization with depth=2...")
viz = eval_pipeline.visualize(depth=2, return_type="graphviz")

# Check the graph structure
print("\nGraph nodes:", viz.body)

# Look for edges - we should see an edge from 'generate' to 'evaluate_answer'
edges = [line for line in viz.body if "->" in line]
print("\nEdges found:")
for edge in edges:
    print(f"  {edge}")

# Check if the fix worked by extracting node IDs and checking connections
# Parse node definitions to map IDs to names
node_map = {}
for line in viz.body:
    if '[label=<' in line and 'TR><TD><B>' in line:
        # Extract node ID
        node_id = line.split('[')[0].strip()
        # Extract node name
        if '<B>' in line and '</B>' in line:
            name = line.split('<B>')[1].split('</B>')[0]
            node_map[node_id] = name

print("\nNode ID mapping:")
for node_id, name in node_map.items():
    print(f"  {node_id} -> {name}")

# Find the IDs for generate and evaluate_answer
generate_id = None
evaluate_id = None
for node_id, name in node_map.items():
    if name == "generate":
        generate_id = node_id
    elif name == "evaluate_answer":
        evaluate_id = node_id

print(f"\nLooking for edge from generate ({generate_id}) to evaluate_answer ({evaluate_id})")

# Check if there's an edge from generate to evaluate_answer
has_correct_edge = False
for edge in edges:
    if generate_id and evaluate_id:
        if f"{generate_id} -> {evaluate_id}" in edge:
            print(f"✓ Found edge: {edge.strip()}")
            has_correct_edge = True

if has_correct_edge:
    print("\n✓ SUCCESS: Visualization correctly shows edge from 'generate' to 'evaluate_answer'!")
else:
    print("\n✗ FAILED: Missing edge from 'generate' to 'evaluate_answer'")

# Save to file for manual inspection
viz.render("test_viz_depth2", format="svg", cleanup=True)
print("\nVisualization saved to test_viz_depth2.svg")

