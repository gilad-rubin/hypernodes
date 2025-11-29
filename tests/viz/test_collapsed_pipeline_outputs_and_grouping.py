"""Test that collapsed pipelines show outputs and inputs are grouped correctly."""
import json
import subprocess
from pathlib import Path

from hypernodes import Pipeline, node
from hypernodes.viz.ui_handler import UIHandler
from hypernodes.viz.js.renderer import JSRenderer


REPO_ROOT = Path(__file__).resolve().parents[2]


@node(output_name="query")
def extract(eval_pair: dict) -> str:
    return eval_pair.get("query", "")


@node(output_name="docs")
def retrieve(query: str, num_results: int = 5) -> list[str]:
    return ["a", "b"]


@node(output_name="answer")
def answer(query: str, docs: list[str], model_name: str) -> str:
    return "ans"


@node(output_name=("score", "details"))
def evaluate(answer: str, expected: str) -> tuple[float, dict]:
    return 1.0, {}


def run_node(script: str) -> dict:
    result = subprocess.run(
        ["node", "-e", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip())


def test_collapsed_pipeline_has_outputs_in_function_outputs():
    """When a pipeline is collapsed, it should have outputs collected like function nodes."""
    retrieval = Pipeline(nodes=[retrieve], name="retrieval_step")
    rag = Pipeline(nodes=[extract, retrieval.as_node(), answer], name="rag_pipeline")
    outer = Pipeline(nodes=[rag.as_node(), evaluate], name="outer")

    handler = UIHandler(outer, depth=3)  # All expanded initially
    graph = handler.get_visualization_data(traverse_collapsed=True)
    rf = JSRenderer().render(graph, separate_outputs=False, show_types=True)

    # Find the 'answer' boundary node at root level with sourceId=rag_pipeline
    answer_nodes = [n for n in rf["nodes"] if n["data"].get("label") == "answer"]
    root_answer = [n for n in answer_nodes if n.get("parentNode") is None]
    
    assert len(root_answer) == 1, "Expected one 'answer' node at root level"
    assert root_answer[0]["data"].get("sourceId") == "rag_pipeline", \
        "Root 'answer' node should have sourceId=rag_pipeline"


def test_input_grouping_after_compression():
    """Inputs that target the same collapsed pipeline should be grouped."""
    retrieval = Pipeline(nodes=[retrieve], name="retrieval_step")
    rag = Pipeline(nodes=[extract, retrieval.as_node(), answer], name="rag_pipeline")
    outer = Pipeline(nodes=[rag.as_node(), evaluate], name="outer")

    handler = UIHandler(outer, depth=3)
    graph = handler.get_visualization_data(traverse_collapsed=True)
    rf = JSRenderer().render(graph, separate_outputs=False, show_types=True)

    # Test the groupInputs function
    script = (
        """
    const utils = require('./assets/viz/state_utils.js');
    const data = __DATA__;
    
    // Simulate collapsed rag_pipeline state
    const nodes = data.nodes.map(n => ({
        ...n,
        data: { 
            ...n.data, 
            isExpanded: n.data.nodeType === 'PIPELINE' ? false : undefined 
        }
    }));
    
    // Apply compression first
    const compressedEdges = utils.compressEdges(nodes, data.edges);
    
    // Then group inputs
    const result = utils.groupInputs(nodes, compressedEdges);
    
    // Count input groups
    const inputGroups = result.nodes.filter(n => n.data?.nodeType === 'INPUT_GROUP');
    const inputParams = inputGroups.flatMap(g => g.data.params || []);
    
    console.log(JSON.stringify({
        inputGroupCount: inputGroups.length,
        inputParams: inputParams,
        totalNodes: result.nodes.length,
        totalEdges: result.edges.length
    }));
    """
    ).replace("__DATA__", json.dumps(rf))

    result = run_node(script)
    
    # When rag_pipeline is collapsed, eval_pair, model_name, num_results should be grouped
    # (they all target rag_pipeline after compression)
    assert result["inputGroupCount"] >= 1, "Expected at least one input group"
    
    # Check that the expected inputs are in a group
    expected_grouped_inputs = {"eval_pair", "model_name", "num_results"}
    actual_grouped_inputs = set(result["inputParams"])
    
    # At least eval_pair, model_name, num_results should be grouped together
    assert expected_grouped_inputs <= actual_grouped_inputs, \
        f"Expected {expected_grouped_inputs} to be grouped, got {actual_grouped_inputs}"


def test_expected_not_grouped_with_rag_inputs():
    """'expected' input should NOT be grouped with rag_pipeline inputs because it targets 'evaluate'."""
    retrieval = Pipeline(nodes=[retrieve], name="retrieval_step")
    rag = Pipeline(nodes=[extract, retrieval.as_node(), answer], name="rag_pipeline")
    outer = Pipeline(nodes=[rag.as_node(), evaluate], name="outer")

    handler = UIHandler(outer, depth=3)
    graph = handler.get_visualization_data(traverse_collapsed=True)
    rf = JSRenderer().render(graph, separate_outputs=False, show_types=True)

    script = (
        """
    const utils = require('./assets/viz/state_utils.js');
    const data = __DATA__;
    
    // Simulate collapsed rag_pipeline state
    const nodes = data.nodes.map(n => ({
        ...n,
        data: { 
            ...n.data, 
            isExpanded: n.data.nodeType === 'PIPELINE' ? false : undefined 
        }
    }));
    
    const compressedEdges = utils.compressEdges(nodes, data.edges);
    const result = utils.groupInputs(nodes, compressedEdges);
    
    // Find 'expected' - it should remain as individual node
    const expectedNode = result.nodes.find(n => n.data?.label === 'expected' && n.data?.nodeType === 'DATA');
    const expectedInGroup = result.nodes.some(n => 
        n.data?.nodeType === 'INPUT_GROUP' && 
        (n.data?.params || []).includes('expected')
    );
    
    console.log(JSON.stringify({
        expectedExists: !!expectedNode,
        expectedInGroup: expectedInGroup
    }));
    """
    ).replace("__DATA__", json.dumps(rf))

    result = run_node(script)
    
    # 'expected' should NOT be in any input group (it targets 'evaluate', not 'rag_pipeline')
    assert not result["expectedInGroup"], "'expected' should not be grouped with rag_pipeline inputs"
    assert result["expectedExists"], "'expected' should exist as an individual node"

