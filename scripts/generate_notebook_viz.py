"""
Recreates the notebook visualization scenarios and saves SVGs for inspection.

Generates depth=2 and depth=3 visualizations for the metrics pipeline
with nested evaluation/retrieval pipelines, mirroring the notebook cells.
"""

from pathlib import Path

from hypernodes import Pipeline, node


@node(output_name="query")
def extract_query(eval_pair: dict) -> str:
    return eval_pair["query"]


@node(output_name="retrieved_docs")
def retrieve(query: str, vector_store: object, top_k: int = 2):
    return [{"doc": query}] * top_k


@node(output_name="answer")
def generate_answer(query: str, retrieved_docs: list, llm: object):
    return {"text": f"answer for {query}", "sources": retrieved_docs}


@node(output_name="evaluation_result")
def evaluate_answer(answer: dict, eval_pair: dict):
    return {"evaluation_result": 1.0}


@node(output_name="metrics")
def compute_metrics(evaluation_results: list):
    return {"count": len(evaluation_results)}


def build_metrics_pipeline() -> Pipeline:
    retrieval = Pipeline(nodes=[retrieve, generate_answer], name="retrieval").bind(
        vector_store=object(), top_k=3, llm=object()
    )
    evaluation = Pipeline(
        nodes=[extract_query, retrieval.as_node(), evaluate_answer],
        name="evaluation",
    )
    evaluation_node = evaluation.as_node(
        name="batch_evaluation",
        map_over="eval_pairs",
        input_mapping={"eval_pairs": "eval_pair"},
        output_mapping={"evaluation_result": "evaluation_results"},
    )
    metrics = Pipeline(nodes=[evaluation_node, compute_metrics], name="metrics")
    return metrics


def save_svgs():
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    metrics = build_metrics_pipeline()

    depth2_path = out_dir / "metrics_depth2.svg"
    metrics.visualize(
        depth=2,
        filename=str(depth2_path.with_suffix("")),
        engine="graphviz",
        group_inputs=True,
        return_type="graphviz",
    )

    depth3_path = out_dir / "metrics_depth3.svg"
    metrics.visualize(
        depth=3,
        filename=str(depth3_path.with_suffix("")),
        engine="graphviz",
        group_inputs=True,
        return_type="graphviz",
    )

    print(f"Saved {depth2_path} and {depth3_path}")


if __name__ == "__main__":
    save_svgs()
