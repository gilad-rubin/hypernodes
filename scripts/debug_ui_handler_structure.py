#!/usr/bin/env python3
"""
Mock standalone example to debug UI handler structure.
Recreates the evaluation_demo.ipynb structure without external dependencies.
"""

import json
from dataclasses import dataclass

from hypernodes import Pipeline, node
from hypernodes.engines import SeqEngine
from hypernodes.viz.ui_handler import UIHandler

# ============================================================================
# Mock Data Models
# ============================================================================


@dataclass
class EvaluationPair:
    query: str
    ground_truth: str


# ============================================================================
# Mock Retrieval Pipeline (nested)
# ============================================================================


@node(output_name="embeddings")
def encode_query(query: str, model: str = "mock-model") -> list:
    """Mock encoding step"""
    return [0.1, 0.2, 0.3]


@node(output_name="retrieved_docs")
def retrieve(embeddings: list, index: str = "mock-index") -> list:
    """Mock retrieval step"""
    return [
        {"doc": "Document 1", "score": 0.95},
        {"doc": "Document 2", "score": 0.87},
    ]


@node(output_name="reranked_docs")
def rerank(retrieved_docs: list, reranker: str = "mock-reranker") -> list:
    """Mock reranking step"""
    return retrieved_docs[:1]  # Return top 1


# Build retrieval pipeline
retrieval_pipeline = Pipeline(
    nodes=[encode_query, retrieve, rerank], engine=SeqEngine()
).bind(model="text-embedding-3-small", index="faiss-index", reranker="cross-encoder")


# ============================================================================
# Mock Evaluation Pipeline (contains nested retrieval)
# ============================================================================


@node(output_name="eval_pair")
def extract_eval_pair(eval_pair: EvaluationPair) -> EvaluationPair:
    """Pass through evaluation pair"""
    return eval_pair


@node(output_name="query")
def extract_query(eval_pair: EvaluationPair) -> str:
    """Extract query from evaluation pair"""
    return eval_pair.query


@node(output_name="ground_truth")
def extract_ground_truth(eval_pair: EvaluationPair) -> str:
    """Extract ground truth from evaluation pair"""
    return eval_pair.ground_truth


# Create retrieval node with mapping
retrieval_node = retrieval_pipeline.as_node(
    input_mapping={"query": "query"}, output_mapping={"reranked_docs": "context"}
)


@node(output_name="generated_answer")
def generate_answer(context: list, query: str, llm: str = "gpt-4") -> str:
    """Mock LLM generation"""
    return f"Generated answer for: {query}"


@node(output_name="evaluation_result")
def evaluate(
    generated_answer: str, ground_truth: str, evaluator: str = "mock-eval"
) -> dict:
    """Mock evaluation"""
    return {"score": 0.85, "passed": True, "explanation": "Good match"}


# Build evaluation pipeline
evaluation_pipeline = Pipeline(
    nodes=[
        extract_eval_pair,
        extract_query,
        extract_ground_truth,
        retrieval_node,
        generate_answer,
        evaluate,
    ],
    engine=SeqEngine(),
).bind(llm="gpt-4o", evaluator="llm-evaluator")


# ============================================================================
# Mock Metrics Pipeline (contains nested evaluation)
# ============================================================================


@node(output_name="eval_pairs")
def pass_through_pairs(eval_pairs: list) -> list:
    """Pass through eval pairs"""
    return eval_pairs


# Create evaluation node with map_over
evaluation_node = evaluation_pipeline.as_node(
    map_over="eval_pairs",
    input_mapping={"eval_pairs": "eval_pair"},
    output_mapping={"evaluation_result": "results"},
)


@node(output_name="metrics")
def aggregate_metrics(results: list) -> dict:
    """Aggregate evaluation results"""
    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))
    avg_score = sum(r.get("score", 0) for r in results) / total if total > 0 else 0

    return {
        "total": total,
        "passed": passed,
        "accuracy": passed / total if total > 0 else 0,
        "average_score": avg_score,
    }


# Build metrics pipeline
metrics_pipeline = Pipeline(
    nodes=[pass_through_pairs, evaluation_node, aggregate_metrics], engine=SeqEngine()
)


# ============================================================================
# UI Handler Structure Extraction
# ============================================================================


def viz_node_to_dict(node) -> dict:
    """Convert a VizNode to a dictionary."""
    from hypernodes.viz.structures import (
        DataNode,
        DualNode,
        FunctionNode,
        GroupDataNode,
        PipelineNode,
    )

    base = {
        "id": node.id,
        "parent_id": node.parent_id,
        "type": type(node).__name__,
    }

    if isinstance(node, (FunctionNode, DualNode)):
        base.update(
            {
                "label": node.label,
                "function_name": node.function_name,
            }
        )
    elif isinstance(node, PipelineNode):
        base.update(
            {
                "label": node.label,
                "is_expanded": node.is_expanded,
            }
        )
    elif isinstance(node, DataNode):
        base.update(
            {
                "name": node.name,
                "type_hint": node.type_hint,
                "is_bound": node.is_bound,
                "source_id": node.source_id,
            }
        )
    elif isinstance(node, GroupDataNode):
        base.update(
            {
                "nodes": [viz_node_to_dict(n) for n in node.nodes],
                "is_bound": node.is_bound,
                "source_id": node.source_id,
            }
        )

    return base


def viz_edge_to_dict(edge) -> dict:
    """Convert a VizEdge to a dictionary."""
    return {
        "source": edge.source,
        "target": edge.target,
        "label": edge.label,
    }


def extract_ui_handler_structure(
    pipeline: Pipeline, depth: int = 1, group_inputs: bool = True
) -> dict:
    """
    Extract the UI handler structure for debugging purposes using the actual UIHandler.

    Args:
        pipeline: The pipeline to analyze
        depth: How deep to traverse nested pipelines
        group_inputs: Whether to group input nodes

    Returns:
        Dictionary with nodes, edges, and metadata from UIHandler
    """
    # Create UIHandler with specified depth
    handler = UIHandler(pipeline, depth=depth, group_inputs=group_inputs)

    # Get visualization data
    viz_data = handler.get_visualization_data()

    # Convert to serializable structure
    structure = {
        "depth": depth,
        "group_inputs": group_inputs,
        "expanded_nodes": list(handler.expanded_nodes),
        "nodes": [viz_node_to_dict(node) for node in viz_data.nodes],
        "edges": [viz_edge_to_dict(edge) for edge in viz_data.edges],
        "metadata": {
            "unfulfilled_args": list(pipeline.unfulfilled_args),
            "bound_inputs": dict(pipeline.bound_inputs),
            "root_args": list(pipeline.graph.root_args),
            "output_names": [str(n.output_name) for n in pipeline.nodes],
            "total_nodes": len(pipeline.nodes),
        },
    }

    return structure


def print_structure_summary(structure: dict, indent: int = 0):
    """Print a human-readable summary of the structure"""
    prefix = "  " * indent

    print(
        f"{prefix}UI Handler Structure (depth={structure['depth']}, group_inputs={structure['group_inputs']}):"
    )
    print(f"{prefix}  Root args: {structure['metadata']['root_args']}")
    print(f"{prefix}  Unfulfilled args: {structure['metadata']['unfulfilled_args']}")
    print(
        f"{prefix}  Bound inputs: {list(structure['metadata']['bound_inputs'].keys())}"
    )
    print(f"{prefix}  Total visualization nodes: {len(structure['nodes'])}")
    print(f"{prefix}  Total edges: {len(structure['edges'])}")
    print(f"{prefix}  Expanded node IDs: {len(structure['expanded_nodes'])} expanded")
    print()

    print(f"{prefix}  Visualization Nodes:")
    for node_info in structure["nodes"]:
        node_type = node_info["type"]
        node_id = node_info["id"]
        parent = node_info.get("parent_id", None)

        if node_type == "FunctionNode":
            print(f"{prefix}    [{node_type}] {node_info['function_name']}")
            print(f"{prefix}        ID: {node_id}")
            print(f"{prefix}        Label: {node_info['label']}")
            if parent:
                print(f"{prefix}        Parent: {parent}")

        elif node_type == "DualNode":
            print(
                f"{prefix}    [{node_type}] {node_info['function_name']} (batch-optimized)"
            )
            print(f"{prefix}        ID: {node_id}")
            print(f"{prefix}        Label: {node_info['label']}")
            if parent:
                print(f"{prefix}        Parent: {parent}")

        elif node_type == "PipelineNode":
            expand_status = "EXPANDED" if node_info["is_expanded"] else "COLLAPSED"
            print(f"{prefix}    [{node_type}] {node_info['label']} [{expand_status}]")
            print(f"{prefix}        ID: {node_id}")
            if parent:
                print(f"{prefix}        Parent: {parent}")

        elif node_type == "DataNode":
            bound_marker = " (BOUND)" if node_info["is_bound"] else ""
            print(f"{prefix}    [{node_type}] {node_info['name']}{bound_marker}")
            print(f"{prefix}        ID: {node_id}")
            if node_info["source_id"]:
                print(f"{prefix}        Source: {node_info['source_id']}")
            if node_info["type_hint"]:
                print(f"{prefix}        Type: {node_info['type_hint']}")
            if parent:
                print(f"{prefix}        Parent: {parent}")

        elif node_type == "GroupDataNode":
            bound_marker = " (BOUND)" if node_info["is_bound"] else ""
            print(
                f"{prefix}    [{node_type}] Group of {len(node_info['nodes'])} data nodes{bound_marker}"
            )
            print(f"{prefix}        ID: {node_id}")
            if node_info["source_id"]:
                print(f"{prefix}        Source: {node_info['source_id']}")
            if parent:
                print(f"{prefix}        Parent: {parent}")
            for sub_node in node_info["nodes"]:
                print(f"{prefix}          - {sub_node['name']}")

        print()

    if structure["edges"]:
        print(f"{prefix}  Edges:")
        for edge in structure["edges"]:
            label_str = f" [{edge['label']}]" if edge["label"] else ""
            print(f"{prefix}    {edge['source']} → {edge['target']}{label_str}")
        print()


# ============================================================================
# Main Execution
# ============================================================================


def main():
    print("=" * 80)
    print("MOCK EVALUATION DEMO - UI HANDLER STRUCTURE DEBUG")
    print("=" * 80)
    print()

    # Create mock evaluation pairs
    eval_pairs = [
        EvaluationPair(
            query="What is Hypernodes?",
            ground_truth="Hypernodes is a Python library for building robust, scalable data pipelines.",
        ),
        EvaluationPair(
            query="How does batch processing work?",
            ground_truth="Hypernodes supports batch processing through the .map() method.",
        ),
    ]

    print("\n" + "=" * 80)
    print("1. RETRIEVAL PIPELINE (base level)")
    print("=" * 80)
    structure = extract_ui_handler_structure(retrieval_pipeline, depth=1)
    print_structure_summary(structure)

    print("\n" + "=" * 80)
    print("2. EVALUATION PIPELINE (depth=1 - collapsed nested)")
    print("=" * 80)
    structure = extract_ui_handler_structure(evaluation_pipeline, depth=1)
    print_structure_summary(structure)

    print("\n" + "=" * 80)
    print("3. EVALUATION PIPELINE (depth=2 - expanded nested)")
    print("=" * 80)
    structure = extract_ui_handler_structure(evaluation_pipeline, depth=2)
    print_structure_summary(structure)

    print("\n" + "=" * 80)
    print("4. METRICS PIPELINE (depth=1 - collapsed nested)")
    print("=" * 80)
    structure = extract_ui_handler_structure(metrics_pipeline, depth=1)
    print_structure_summary(structure)

    print("\n" + "=" * 80)
    print("5. METRICS PIPELINE (depth=2 - show evaluation layer)")
    print("=" * 80)
    structure = extract_ui_handler_structure(metrics_pipeline, depth=2)
    print_structure_summary(structure)

    print("\n" + "=" * 80)
    print("6. METRICS PIPELINE (depth=3 - fully expanded)")
    print("=" * 80)
    structure = extract_ui_handler_structure(metrics_pipeline, depth=3)
    print_structure_summary(structure)

    # Save full JSON structure to file
    print("\n" + "=" * 80)
    print("Saving full JSON structures to outputs/")
    print("=" * 80)

    import os

    os.makedirs("outputs", exist_ok=True)

    structures = {
        "retrieval_depth1": extract_ui_handler_structure(retrieval_pipeline, depth=1),
        "evaluation_depth1": extract_ui_handler_structure(evaluation_pipeline, depth=1),
        "evaluation_depth2": extract_ui_handler_structure(evaluation_pipeline, depth=2),
        "metrics_depth1": extract_ui_handler_structure(metrics_pipeline, depth=1),
        "metrics_depth2": extract_ui_handler_structure(metrics_pipeline, depth=2),
        "metrics_depth3": extract_ui_handler_structure(metrics_pipeline, depth=3),
    }

    output_file = "outputs/ui_handler_structures_debug.json"
    with open(output_file, "w") as f:
        json.dump(structures, f, indent=2, default=str)

    print(f"✅ Saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
