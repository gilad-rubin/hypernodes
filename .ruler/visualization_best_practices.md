# Visualization System - Architecture & Debugging Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  React Flow + ELK Layout                                ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ ││
│  │  │ CustomNode  │  │ CustomEdge  │  │ PipelineGroup   │ ││
│  │  └─────────────┘  └─────────────┘  └─────────────────┘ ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │  state_utils.js                                         ││
│  │  applyState → applyVisibility → compressEdges →        ││
│  │  groupInputs                                            ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ HTML with embedded JSON
┌─────────────────────────────────────────────────────────────┐
│                    Python Backend                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ UIHandler   │→ │ JSRenderer  │→ │ html_generator.py   │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│         ▲                                                    │
│  ┌─────────────┐                                            │
│  │ GraphWalker │                                            │
│  └─────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
src/hypernodes/viz/
├── graph_walker.py      # Traverses pipeline DAG, generates flat node/edge structure
├── ui_handler.py        # Manages depth, expansion, serialization
├── structures.py        # Data classes: FunctionNode, PipelineNode, DataNode, VizEdge
├── js/
│   ├── renderer.py      # Transforms VisualizationGraph → React Flow format
│   └── html_generator.py # Generates complete HTML with React/ELK/Tailwind
├── graphviz/
│   └── renderer.py      # Static Graphviz SVG rendering

assets/viz/
├── state_utils.js       # Client-side state transformations (applyState, compressEdges, etc.)
├── theme_utils.js       # Theme detection and color parsing
├── reactflow.umd.js     # React Flow library
├── elk.bundled.js       # ELK layout library
└── custom.css           # Custom styling

tests/viz/
├── test_collapsed_pipeline_outputs_and_grouping.py  # Combined outputs, input grouping
├── test_collapsed_pipelines_no_dangling_edges.py    # Edge compression
├── test_separate_outputs.py                          # Both display modes
├── test_state_utils_visibility.py                    # Visibility logic
├── test_debug_tools.py                               # Debug utilities
└── ...
```

## Data Flow

### 1. Python Side (Build Time)

```python
# 1. GraphWalker traverses the pipeline
walker = GraphWalker(pipeline, depth=2, traverse_collapsed=True)
graph = walker.walk()  # Returns VisualizationGraph

# 2. UIHandler wraps for serialization
handler = UIHandler(pipeline, depth=2)
graph_data = handler.get_visualization_data(traverse_collapsed=True)

# 3. JSRenderer transforms to React Flow format
renderer = JSRenderer()
rf_data = renderer.render(graph_data, theme='dark', separate_outputs=False)

# 4. HTML generator embeds data
html = generate_widget_html(rf_data)
```

### 2. JavaScript Side (Runtime)

```javascript
// 1. Parse embedded data
const initialData = JSON.parse(document.getElementById('graph-data').textContent);

// 2. Apply state transformations
const stateResult = applyState(nodes, edges, { expansionState, separateOutputs, showTypes, theme });

// 3. Apply visibility (hide children of collapsed pipelines)
const nodesWithVis = applyVisibility(stateResult.nodes, expansionState);

// 4. Compress edges (remap to visible ancestors)
const compressedEdges = compressEdges(nodesWithVis, stateResult.edges);

// 5. Group inputs (combine inputs targeting same node)
const { nodes, edges } = groupInputs(nodesWithVis, compressedEdges);

// 6. ELK layout → React Flow render
```

---

## Debugging Guide

### Quick Debug Commands

```bash
# Generate test HTML
uv run python scripts/test_collapsed_pipeline.py

# Run all viz tests
uv run pytest tests/viz/ -v

# Run specific test
uv run pytest tests/viz/test_collapsed_pipelines_no_dangling_edges.py -v
```

### Browser Console Debug API

The visualization exposes a debug API via `HyperNodesVizState.debug`:

```javascript
// Enable verbose logging for edge compression and layout
HyperNodesVizState.debug.enableDebug()

// Analyze current graph state (nodes, edges, pipelines, dangling edges)
HyperNodesVizState.debug.analyzeState()

// Get current pipeline expansion state
HyperNodesVizState.debug.getExpansionState()

// Simulate edge compression with a specific expansion state
HyperNodesVizState.debug.simulateCompression({ 'rag_pipeline': false })

// Disable debug mode
HyperNodesVizState.debug.disableDebug()

// === Visual Debug Overlays ===

// Show debug overlays (node bounding boxes + edge connection points)
HyperNodesVizState.debug.showOverlays()

// Hide debug overlays
HyperNodesVizState.debug.hideOverlays()

// Inspect current layout (node positions, dimensions, edge paths)
HyperNodesVizState.debug.inspectLayout()
// Returns: { nodes: [{id, x, y, width, height, bottom}], edges: [...], edgePaths: [...] }

// Validate edge-node connections (checks if edges connect within node bounds)
HyperNodesVizState.debug.validateConnections()
// Returns: { valid: bool, issues: [{edge, type, issue, expected, actual}], summary: str }

// Get comprehensive debug report (runs all analyses)
HyperNodesVizState.debug.fullReport()
```

### Visual Debug Mode

Debug overlays can be enabled in three ways:

1. **UI Toggle**: Click the bug icon in the view controls panel
2. **Console**: `HyperNodesVizState.debug.showOverlays()`
3. **URL Parameter**: Add `?debug=overlays` or `?debug=true` to the URL

When enabled, the visualization shows:
- **Red dashed boxes** around each node with ID labels
- **Green circles** at edge source points
- **Blue circles** at edge target points
- **Coordinate labels** on edges showing (sourceX, sourceY) → (targetX, targetY)

### Python State Simulator

The `state_simulator` module replicates JavaScript state transformations in Python for testing:

```python
from hypernodes.viz import (
    UIHandler,
    simulate_state,
    verify_state,
    verify_edge_alignment,
    simulate_collapse_expand_cycle,
    diagnose_all_states,
)

# Get visualization graph
handler = UIHandler(pipeline, depth=99)
graph = handler.get_visualization_data(traverse_collapsed=True)

# Simulate a specific state (collapsed pipeline, combined outputs)
result = simulate_state(
    graph,
    expansion_state={"my_pipeline": False},
    separate_outputs=False,
)

# Verify edges are valid
alignment = verify_edge_alignment(result)
if not alignment["valid"]:
    print("Issues:", alignment["issues"])

# Test a complete collapse/expand cycle
cycle = simulate_collapse_expand_cycle(graph, "my_pipeline")
print(cycle["summary"])  # "Pipeline 'my_pipeline' collapse/expand cycle: PASS (0 total issues)"

# Diagnose all state combinations
all_states = diagnose_all_states(graph)
for key, state in all_states.items():
    if state["orphan_edges"]:
        print(f"{key}: orphan edges found!")
```

### Debugging Each Layer

#### 1. GraphWalker (`graph_walker.py`)

**What it does**: Traverses the pipeline DAG and creates nodes/edges.

**Debug approach**:
```python
from hypernodes.viz.graph_walker import GraphWalker

walker = GraphWalker(pipeline, depth=2, traverse_collapsed=True)
graph = walker.walk()

# Inspect nodes
for node in graph.nodes:
    print(f"{node.__class__.__name__}: {node.id} (parent={node.parent_id})")

# Inspect edges
for edge in graph.edges:
    print(f"  {edge.source} → {edge.target}")
```

**Common issues**:
- Missing boundary outputs → Check `_expand_pipeline_node` creates them
- Wrong parent assignment → Check `parent_id` parameter passing
- Missing type hints → Check `_extract_input_type` recursive lookup

#### 2. UIHandler (`ui_handler.py`)

**What it does**: Manages depth, expansion state, and serialization.

**Debug approach**:
```python
from hypernodes.viz.ui_handler import UIHandler

handler = UIHandler(pipeline, depth=2)
data = handler.get_visualization_data(traverse_collapsed=True)

# Check what nodes exist
print("Nodes:", [n.id for n in data.nodes])
print("Pipelines:", [n.id for n in data.nodes if hasattr(n, 'is_expanded')])
```

**Key parameters**:
- `depth`: How many levels to expand initially
- `traverse_collapsed=True`: Pre-fetch internal structure for interactive expand/collapse

#### 3. JSRenderer (`js/renderer.py`)

**What it does**: Transforms VisualizationGraph → React Flow node/edge format.

**Debug approach**:
```python
from hypernodes.viz.js.renderer import JSRenderer

renderer = JSRenderer()
rf_data = renderer.render(graph_data, theme="dark", separate_outputs=False)

# Inspect React Flow data
import json
print(json.dumps(rf_data, indent=2))
```

**Key outputs**:
- `nodes[].data.nodeType`: FUNCTION, PIPELINE, DATA, INPUT_GROUP, DUAL
- `nodes[].data.isExpanded`: For PIPELINE nodes
- `nodes[].data.sourceId`: For output DATA nodes (points to producer)
- `nodes[].parentNode`: For nested nodes
- `edges[].sourcePosition/targetPosition`: Should be "bottom"/"top"

#### 4. state_utils.js (Frontend)

**What it does**: Client-side state transformations.

**Debug with Node.js**:
```javascript
// scripts/debug_state.js
const utils = require('../assets/viz/state_utils.js');
const fs = require('fs');

const html = fs.readFileSync('outputs/test.html', 'utf-8');
const match = html.match(/<script id="graph-data"[^>]*>([\s\S]*?)<\/script>/);
const data = JSON.parse(match[1]);

// Test transformations
const result = utils.applyState(data.nodes, data.edges, {
    expansionState: new Map([["rag_pipeline", false]]),
    separateOutputs: false,
    showTypes: true,
    theme: "dark"
});
console.log(JSON.stringify(result, null, 2));
```

Run: `node scripts/debug_state.js`

**Key functions**:
- `applyState`: Applies theme, separateOutputs mode, combines outputs
- `applyVisibility`: Hides children of collapsed pipelines
- `compressEdges`: Remaps edges to visible ancestors when pipelines collapse
- `groupInputs`: Groups inputs targeting the same node

#### 5. Playwright Browser Testing

For interactive debugging:
```python
# Use Playwright MCP tools
mcp_playwright_browser_navigate(url="file:///path/to/test.html")
mcp_playwright_browser_wait_for(time=2)  # Wait for React/ELK to render
mcp_playwright_browser_console_messages()  # Check for errors
mcp_playwright_browser_evaluate(function="() => HyperNodesVizState.debug.analyzeState()")
```

---

## Key Patterns

### Boundary Outputs

When a pipeline is collapsed, its outputs appear at the parent level:

```
Expanded:                          Collapsed:
┌─ rag_pipeline ─────────┐        ┌─ rag_pipeline ────┐
│  generate_answer       │   →    │  → answer : str   │
│      → answer          │        └───────────────────┘
└────────────────────────┘
```

Boundary output nodes have `sourceId` pointing to the PIPELINE:
```javascript
{ id: "answer", data: { sourceId: "rag_pipeline", nodeType: "DATA" } }
```

### Edge Compression

When pipelines collapse, edges to internal nodes remap to the collapsed pipeline:

```javascript
// getVisibleAncestor walks up parent chain to find collapsed pipeline
const sourceVis = getVisibleAncestor(edge.source);  // e.g., "rag_pipeline"
const targetVis = getVisibleAncestor(edge.target);
// Remap: edge to internal node → edge to collapsed pipeline
```

### Input Grouping

Inputs targeting the same node are grouped into INPUT_GROUP:
```
Before: eval_pair → rag_pipeline, model_name → rag_pipeline
After:  [eval_pair, model_name, num_results] → rag_pipeline
```

---

## Common Issues & Fixes

| Issue | What to Check | Fix Location |
|-------|---------------|--------------|
| Missing edges after collapse | `compressEdges` output, `getVisibleAncestor` | `state_utils.js` |
| Hanging/dangling arrows | Handle positions, node visibility, `updateNodeInternals` | `html_generator.py` |
| Edge starts/ends outside node | Use `validateConnections()` to diagnose | `state_utils.js` debug API |
| Stale edge paths after layout | Layout version, edge ID updates | `html_generator.py` |
| Nodes not grouping | `groupInputs`, target matching | `state_utils.js` |
| Outputs not combined | `applyState`, `sourceId` values | `state_utils.js` |
| Wrong node positions | ELK layout, `parentNode` | `html_generator.py` |
| Types missing on inputs | `_extract_input_type` | `graph_walker.py` |
| Pipeline outputs not shown | `functionOutputs` collection | `state_utils.js` |

### Debugging Edge Alignment Issues

When edges appear to "hang" or not connect properly:

1. **Enable debug overlays**: `HyperNodesVizState.debug.showOverlays()` or toggle in UI
2. **Inspect layout**: `HyperNodesVizState.debug.inspectLayout()` to see node positions
3. **Validate connections**: `HyperNodesVizState.debug.validateConnections()` to check bounds
4. **Check edge paths**: Look for `delta.y` in issues to see how far off edges are
5. **Run full report**: `HyperNodesVizState.debug.fullReport()` for comprehensive analysis

Common causes:
- React Flow caching edge paths after node resize
- Layout not triggering `updateNodeInternals`
- Asynchronous ELK layout completing after render

---

## Do's and Don'ts

### DO's ✅

1. **Use `traverse_collapsed=True`** for interactive viz (pre-fetches internal structure)
2. **Filter boundary outputs based on expansion state** (expanded→hide, collapsed→show)
3. **Remap edges when hiding nodes** (use `compressEdges`)
4. **Test both display modes** (`separateOutputs=true/false`)
5. **Preserve expansion state across option changes** (store separately)
6. **Verify with Playwright** for interactive behavior

### DON'TS ❌

1. **Don't skip edges from expanded pipelines** without remapping
2. **Don't assume all DATA nodes with sourceId are function outputs** (check sourceNodeTypes)
3. **Don't filter edges before remapping** (order matters)
4. **Don't create zero-size handles** (use opacity-0 instead)
5. **Don't modify nodes in place** (use spread for React state)

---

## Running Tests

```bash
# All visualization tests
uv run pytest tests/viz/ -v

# Specific test categories
uv run pytest tests/viz/test_collapsed*.py -v      # Collapse behavior
uv run pytest tests/viz/test_separate_outputs.py -v # Display modes
uv run pytest tests/viz/test_debug_tools.py -v     # Debug utilities

# Edge alignment tests (requires Playwright)
pip install playwright && playwright install chromium
uv run pytest tests/viz/test_edge_alignment_playwright.py -v

# Python-only edge alignment tests (no browser)
uv run pytest tests/viz/test_edge_alignment_playwright.py::TestPythonEdgeAlignment -v
```

### Playwright Browser Tests

The `test_edge_alignment_playwright.py` file contains tests that validate edge-node connections in a real browser:

```python
# Run tests that verify edges connect properly after collapse/expand
pytest tests/viz/test_edge_alignment_playwright.py::TestPlaywrightEdgeAlignment -v

# These tests:
# 1. Generate HTML visualization
# 2. Open in headless Chromium
# 3. Click to collapse/expand pipelines
# 4. Use debug.validateConnections() to verify alignment
# 5. Assert no issues found
```

## Generate Test HTML

```python
from hypernodes.viz.ui_handler import UIHandler
from hypernodes.viz.js.renderer import JSRenderer
from hypernodes.viz.js.html_generator import generate_widget_html

handler = UIHandler(pipeline, depth=2)
graph_data = handler.get_visualization_data(traverse_collapsed=True)
renderer = JSRenderer()
rf_data = renderer.render(graph_data, theme="dark", separate_outputs=False, show_types=True)
html = generate_widget_html(rf_data)

with open("outputs/test.html", "w") as f:
    f.write(html)
```






