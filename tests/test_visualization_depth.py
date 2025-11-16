"""Tests for visualization depth parameter, specifically nested pipeline expansion."""
from dataclasses import dataclass

import pytest

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
    """Retrieve documents."""
    return [Document(text="doc1", score=0.9)]


@node(output_name="answer")
def generate(query: str, retrieved_docs: list[Document], llm) -> Answer:
    """Generate answer."""
    return Answer(text="answer", sources=retrieved_docs)


@node(output_name="evaluation")
def evaluate_answer(answer: Answer, ground_truth: str, query: str) -> EvaluationResult:
    """Evaluate answer."""
    return EvaluationResult(
        query=query,
        generated_answer=answer.text,
        ground_truth=ground_truth,
        score=0.5,
    )


def test_depth_2_shows_edges_from_inner_to_outer_nodes():
    """Test that depth=2 visualization correctly shows edges from inner nodes to outer nodes.
    
    This is a regression test for a bug where expanded nested pipelines (depth > 1)
    would not show edges from their inner output-producing nodes to outer consuming nodes.
    
    Example:
        RAG Pipeline (expanded):
            - retrieve -> generate (produces "answer")
        Outer Pipeline:
            - evaluate_answer (consumes "answer")
        
        Expected: Edge from "generate" to "evaluate_answer"
    """
    # Build RAG pipeline
    rag_pipeline = Pipeline(nodes=[retrieve, generate])

    # Use RAG pipeline as a node in evaluation pipeline
    eval_pipeline = Pipeline(nodes=[rag_pipeline.as_node(name="RAG"), evaluate_answer])

    # Visualize with depth=2 to expand the nested pipeline
    viz = eval_pipeline.visualize(depth=2, return_type="graphviz")

    # Extract node IDs and map them to names
    node_map = {}
    for line in viz.body:
        if '[label=<' in line and 'TR><TD><B>' in line:
            node_id = line.split('[')[0].strip()
            if '<B>' in line and '</B>' in line:
                name = line.split('<B>')[1].split('</B>')[0]
                node_map[node_id] = name

    # Find the IDs for generate and evaluate_answer
    generate_id = None
    evaluate_id = None
    for node_id, name in node_map.items():
        if name == "generate":
            generate_id = node_id
        elif name == "evaluate_answer":
            evaluate_id = node_id

    assert generate_id is not None, "generate node not found in visualization"
    assert evaluate_id is not None, "evaluate_answer node not found in visualization"

    # Extract all edges
    edges = [line for line in viz.body if "->" in line]

    # Check if there's an edge from generate to evaluate_answer
    has_edge = any(f"{generate_id} -> {evaluate_id}" in edge for edge in edges)

    assert has_edge, (
        f"Missing edge from 'generate' to 'evaluate_answer' in depth=2 visualization. "
        f"Expected edge from {generate_id} to {evaluate_id}. "
        f"Found edges: {edges}"
    )


def test_depth_1_shows_collapsed_pipeline_node():
    """Test that depth=1 (default) shows the nested pipeline as a single collapsed node."""
    # Build RAG pipeline
    rag_pipeline = Pipeline(nodes=[retrieve, generate])

    # Use RAG pipeline as a node in evaluation pipeline
    eval_pipeline = Pipeline(nodes=[rag_pipeline.as_node(name="RAG"), evaluate_answer])

    # Visualize with depth=1 (default) - should show collapsed pipeline
    viz = eval_pipeline.visualize(depth=1, return_type="graphviz")

    # Extract node names
    node_names = []
    for line in viz.body:
        if '[label=<' in line and 'TR><TD><B>' in line:
            if '<B>' in line and '</B>' in line:
                name = line.split('<B>')[1].split('</B>')[0]
                node_names.append(name)

    # Should see RAG and evaluate_answer, but NOT retrieve or generate
    assert "RAG" in node_names, "RAG pipeline node not found"
    assert "evaluate_answer" in node_names, "evaluate_answer node not found"
    assert "retrieve" not in node_names, "retrieve should not be visible at depth=1"
    assert "generate" not in node_names, "generate should not be visible at depth=1"


def test_depth_2_shows_all_inner_nodes():
    """Test that depth=2 expands nested pipeline and shows all inner nodes."""
    # Build RAG pipeline
    rag_pipeline = Pipeline(nodes=[retrieve, generate])

    # Use RAG pipeline as a node in evaluation pipeline
    eval_pipeline = Pipeline(nodes=[rag_pipeline.as_node(name="RAG"), evaluate_answer])

    # Visualize with depth=2 - should expand and show inner nodes
    viz = eval_pipeline.visualize(depth=2, return_type="graphviz")

    # Extract node names
    node_names = []
    for line in viz.body:
        if '[label=<' in line and 'TR><TD><B>' in line:
            if '<B>' in line and '</B>' in line:
                name = line.split('<B>')[1].split('</B>')[0]
                node_names.append(name)

    # Should see inner nodes (retrieve, generate) and outer node (evaluate_answer)
    assert "retrieve" in node_names, "retrieve should be visible at depth=2"
    assert "generate" in node_names, "generate should be visible at depth=2"
    assert "evaluate_answer" in node_names, "evaluate_answer should be visible at depth=2"
    # RAG pipeline itself should NOT be visible as a node (it's expanded)
    assert "RAG" not in node_names, "RAG should not be a node at depth=2 (it's expanded)"


def test_name_priority_as_node_overrides_pipeline():
    """Test that as_node(name=...) is prioritized over Pipeline(name=...) in visualization.
    
    Name priority should be:
    1. as_node(name=...) - highest priority
    2. Pipeline(name=...)
    3. "pipeline" - default fallback
    """
    # Test 1: as_node(name="RAG") overrides Pipeline(name="inner_rag")
    rag_pipeline_named = Pipeline(nodes=[retrieve, generate], name="inner_rag")
    eval_pipeline = Pipeline(
        nodes=[rag_pipeline_named.as_node(name="RAG"), evaluate_answer]
    )
    viz = eval_pipeline.visualize(depth=2, return_type="graphviz")

    # Should show "RAG" (from as_node), NOT "inner_rag" (from Pipeline)
    has_rag_label = any("label=RAG" in line for line in viz.body)
    has_inner_rag_label = any("label=inner_rag" in line for line in viz.body)
    
    assert has_rag_label, "Expected 'RAG' to appear as cluster label"
    assert not has_inner_rag_label, "Expected 'inner_rag' to be overridden by 'RAG'"

    # Test 2: Pipeline(name="MyPipeline") is used when no as_node name provided
    rag_pipeline_only = Pipeline(nodes=[retrieve, generate], name="MyPipeline")
    eval_pipeline_2 = Pipeline(nodes=[rag_pipeline_only.as_node(), evaluate_answer])
    viz2 = eval_pipeline_2.visualize(depth=2, return_type="graphviz")

    has_my_pipeline = any("label=MyPipeline" in line for line in viz2.body)
    assert has_my_pipeline, "Expected 'MyPipeline' to appear as cluster label"

    # Test 3: Fallback to "pipeline" when no names provided
    rag_pipeline_no_name = Pipeline(nodes=[retrieve, generate])
    eval_pipeline_3 = Pipeline(nodes=[rag_pipeline_no_name.as_node(), evaluate_answer])
    viz3 = eval_pipeline_3.visualize(depth=2, return_type="graphviz")

    has_pipeline = any("label=pipeline" in line for line in viz3.body)
    assert has_pipeline, "Expected 'pipeline' to appear as cluster label"
