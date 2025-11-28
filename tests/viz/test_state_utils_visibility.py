import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_node(script: str) -> dict:
    result = subprocess.run(
        ["node", "-e", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout.strip())


def test_hidden_state_persists_across_toggles_for_collapsed_pipelines():
    script = r"""
    const path = require('path');
    const utils = require(path.join(process.cwd(), 'assets', 'viz', 'state_utils.js'));

    const baseNodes = [
      { id: 'pipe', type: 'custom', data: { nodeType: 'PIPELINE', label: 'pipe', isExpanded: false } },
      { id: 'pipe__inner', type: 'custom', parentNode: 'pipe', data: { nodeType: 'FUNCTION', label: 'inner' } },
    ];
    const baseEdges = [];
    const expansionState = new Map([['pipe', false]]);

    const run = (separateOutputs, showTypes) => {
      const state = utils.applyState(baseNodes, baseEdges, { expansionState, separateOutputs, showTypes, theme: 'dark' });
      const withVisibility = utils.applyVisibility(state.nodes, expansionState);
      const inner = withVisibility.find((n) => n.id === 'pipe__inner');
      return inner ? inner.hidden : null;
    };

    const hiddenBefore = run(false, true);
    const hiddenAfterTypesToggle = run(false, false);
    const hiddenAfterSeparateToggle = run(true, true);

    console.log(JSON.stringify({ before: hiddenBefore, types: hiddenAfterTypesToggle, separate: hiddenAfterSeparateToggle }));
    """

    result = run_node(script)

    assert result["before"] is True
    assert result["types"] is True
    assert result["separate"] is True
