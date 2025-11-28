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
def retrieve(query: str) -> list[str]:
    return ["a", "b"]


@node(output_name="answer")
def answer(query: str, docs: list[str], model_name: str) -> str:
    return "ans"


@node(output_name=("score", "details"))
def evaluate(answer: str, expected: str) -> tuple[float, dict]:
    return 1.0, {}


def run_node(script: str) -> list[dict]:
    result = subprocess.run(
        ["node", "-e", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip())


def test_double_collapsed_edges_rewire_to_visible_ancestors():
    retrieval = Pipeline(nodes=[retrieve], name="retrieval_step")
    rag = Pipeline(nodes=[extract, retrieval.as_node(), answer], name="rag_pipeline")
    outer = Pipeline(nodes=[rag.as_node(), evaluate], name="outer")

    handler = UIHandler(outer, depth=1)
    graph = handler.get_visualization_data(traverse_collapsed=True)

    rf = JSRenderer().render(graph, separate_outputs=False, show_types=True)

    script = (
        """
    const utils = require('./assets/viz/state_utils.js');
    const data = __DATA__;
    const nodes = data.nodes.map(n => ({...n, hidden: !!n.hidden}));
    const edges = data.edges;
    const res = utils.compressEdges(nodes, edges);
    console.log(JSON.stringify(res));
    """
    ).replace("__DATA__", json.dumps(rf))

    compressed = run_node(script)

    ids = {e["source"] for e in compressed} | {e["target"] for e in compressed}

    # All edge endpoints should be visible top-level nodes (no descendants of collapsed pipelines)
    assert all("__" not in i for i in ids)
    # We expect edges from inputs to rag_pipeline and rag_pipeline (or its output) to evaluate/expected
    assert any(e["target"] == "rag_pipeline" for e in compressed)
    assert any(
        e["target"] == "evaluate" and e["source"] in {"rag_pipeline", "answer"}
        for e in compressed
    )
