
import json
from typing import List, Any
from hypernodes import node, Pipeline
from hypernodes.viz.visualization_widget import PipelineWidget

# Mock types
class VectorStore: pass
class LLM: pass
class EvaluationPair: pass
class JudgeEvaluator: pass
class Document: pass
class EvaluationResult: pass

@node(output_name="retrieved_docs")
def retrieve(query: str, vector_store: VectorStore, top_k: int = 2) -> List[Document]:
    return []

@node(output_name="answer")
def generate(query: str, retrieved_docs: List[Document], llm: LLM) -> str:
    return "answer"

retrieval = Pipeline([retrieve, generate])
retrieval.bind(vector_store=VectorStore(), llm=LLM(), top_k=2)

@node(output_name="query")
def extract_query(eval_pair: EvaluationPair) -> str:
    return "query"

@node(output_name="evaluation_result")
def evaluate_answer(eval_pair: EvaluationPair, answer: str, judge: JudgeEvaluator) -> EvaluationResult:
    return EvaluationResult()

batch_evaluation = Pipeline([extract_query, retrieval.as_node(), evaluate_answer])
batch_evaluation.bind(judge=JudgeEvaluator())

@node(output_name="metrics")
def compute_metrics(evaluation_results: List[EvaluationResult]) -> dict:
    return {}

# Map batch_evaluation over eval_pairs
batch_node = batch_evaluation.as_node(
    input_mapping={"eval_pairs": "eval_pair"},
    output_mapping={"evaluation_result": "evaluation_results"},
    map_over="eval_pairs"
)

metrics_pipeline = Pipeline([batch_node, compute_metrics])

# Generate HTML
print("Generating visualization HTML...")
# passing depth=3 to verify that it expands correctly initially
widget = PipelineWidget(metrics_pipeline, depth=3)
html_content = widget._generate_html(depth=3)

output_path = "tests/viz_output.html"
with open(output_path, "w") as f:
    f.write(html_content)

print(f"HTML written to {output_path}")
