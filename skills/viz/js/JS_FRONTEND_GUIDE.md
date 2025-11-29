# HyperNodes Visualization: JS Frontend Guide

A comprehensive guide to working with the React Flow + ELK based visualization system for HyperNodes pipelines.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Dependencies & Versions](#dependencies--versions)
3. [Data Flow](#data-flow)
4. [Node Height Calculations](#node-height-calculations)
5. [Edge & Handle System](#edge--handle-system)
6. [Debugging Tools](#debugging-tools)
7. [Testing with Playwright](#testing-with-playwright)
8. [Common Issues & Solutions](#common-issues--solutions)
9. [Do's and Don'ts](#dos-and-donts)
10. [File Reference](#file-reference)

---

## Architecture Overview

The visualization system consists of two main parts:

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Backend                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ GraphWalker │→ │ UIHandler   │→ │ JSRenderer          │ │
│  │ (traverse)  │  │ (serialize) │  │ (React Flow fmt)    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                           ↓                  │
│                               ┌─────────────────────┐       │
│                               │ html_generator.py   │       │
│                               │ (embeds everything) │       │
│                               └─────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              ↓ HTML with embedded JSON
┌─────────────────────────────────────────────────────────────┐
│                        Browser                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  state_utils.js                                         ││
│  │  applyState → applyVisibility → compressEdges →        ││
│  │  groupInputs                                            ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │  ELK Layout → React Flow Render                         ││
│  │  (useLayout hook calculates positions)                  ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Custom Components                                       ││
│  │  CustomNode | CustomEdge | PipelineGroup                ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Location | Responsibility |
|-----------|----------|----------------|
| GraphWalker | `viz/graph_walker.py` | Traverse pipeline DAG, extract nodes/edges |
| UIHandler | `viz/ui_handler.py` | Manage depth, expansion, serialization |
| JSRenderer | `viz/js/renderer.py` | Transform to React Flow format |
| html_generator | `viz/js/html_generator.py` | Generate complete HTML with embedded JS |
| state_utils.js | `assets/viz/state_utils.js` | Client-side state transformations |
| theme_utils.js | `assets/viz/theme_utils.js` | Theme detection (VS Code dark/light) |

---

## Dependencies & Versions

The visualization uses **CDN-hosted libraries** embedded in the HTML output:

| Library | Version | CDN URL |
|---------|---------|---------|
| React | 18 | `unpkg.com/react@18` |
| React DOM | 18 | `unpkg.com/react-dom@18` |
| **React Flow** | **11.10.1** | `unpkg.com/@xyflow/react@11.10.1` |
| ELK (elkjs) | 0.9.3 | `unpkg.com/elkjs@0.9.3` |
| Dagre | - | `unpkg.com/dagre` |
| Web Worker | - | `unpkg.com/web-worker` |

> ⚠️ **IMPORTANT**: We use React Flow **v11.10.1**, NOT v12. The v12 API is different.
> 
> Key v11 vs v12 differences:
> - v11: `import ReactFlow from '@xyflow/react'`
> - v12: `import { ReactFlow } from '@xyflow/react'`
> - v11: `defaultViewport`, `onInit`
> - v12: `defaultViewport` → `initialViewport`, `onInit` → `onReady`

---

## Data Flow

### 1. Python Side (Generation Time)

```python
from hypernodes.viz import UIHandler
from hypernodes.viz.js.renderer import JSRenderer
from hypernodes.viz.js.html_generator import generate_widget_html

# 1. Create handler and get visualization data
handler = UIHandler(pipeline, depth=99)
graph_data = handler.get_visualization_data(traverse_collapsed=True)

# 2. Render to React Flow format
renderer = JSRenderer()
rf_data = renderer.render(
    graph_data, 
    theme="dark", 
    separate_outputs=False,  # Combined outputs mode
    show_types=True
)

# 3. Generate HTML
html = generate_widget_html(rf_data)
```

### 2. JavaScript Side (Runtime)

```javascript
// Embedded in html_generator.py output

// 1. Parse initial data
const initialData = JSON.parse(document.getElementById('graph-data').textContent);

// 2. State transformations (state_utils.js)
const stateResult = applyState(nodes, edges, { 
    expansionState, 
    separateOutputs, 
    showTypes, 
    theme 
});

// 3. Visibility (hide children of collapsed pipelines)
const nodesWithVis = applyVisibility(stateResult.nodes, expansionState);

// 4. Edge compression (remap to visible ancestors)
const compressedEdges = compressEdges(nodesWithVis, stateResult.edges);

// 5. Input grouping (combine inputs to same node)
const { nodes, edges } = groupInputs(nodesWithVis, compressedEdges);

// 6. ELK layout calculates positions
const layoutedNodes = await runElkLayout(nodes, edges);

// 7. React Flow renders
<ReactFlow nodes={layoutedNodes} edges={edges} ... />
```

---

## Node Height Calculations

### The Critical Issue

**ELK layout must know node heights BEFORE rendering**, but actual heights depend on DOM content. If ELK calculates `height=64px` but the DOM renders `height=98px`, edges will connect to wrong positions.

### Height Formula (Current Implementation)

Located in `html_generator.py`, `useLayout` hook:

```javascript
// Function nodes with outputs (combined mode)
if (n.data.nodeType === 'FUNCTION' && n.data.outputs?.length > 0) {
    // Breakdown:
    // - Header section: ~60px (py-2.5 padding + text content)
    // - First output: ~38px
    // - Additional outputs: 24px each
    height = 60 + 38 + ((n.data.outputs.length - 1) * 24);
}

// Collapsed pipeline nodes with outputs
if (n.data.nodeType === 'PIPELINE' && !n.data.isExpanded && n.data.outputs?.length > 0) {
    // Same formula as function nodes
    height = 60 + 38 + ((n.data.outputs.length - 1) * 24);
}
```

### Visual Breakdown

```
┌─────────────────────────────┐
│ py-2.5 (10px)               │
│ ┌─────────────────────────┐ │
│ │ Node Name + Icon        │ │  ~40px content
│ └─────────────────────────┘ │
│ py-2.5 (10px)               │
├─────────────────────────────┤  ← border-t (1px)
│ pt-2 (8px)                  │
│ ┌─────────────────────────┐ │
│ │ Output 1  : type        │ │  ~22px
│ └─────────────────────────┘ │
│ gap-1 (4px)                 │
│ ┌─────────────────────────┐ │
│ │ Output 2  : type        │ │  ~22px
│ └─────────────────────────┘ │
│ pb-2 (8px)                  │
└─────────────────────────────┘
  Handle (at bottom-center)
```

### How to Verify Heights

Use the browser debug API:

```javascript
HyperNodesVizState.debug.inspectLayout()
// Returns: { nodes: [{id, x, y, width, height, bottom}], ... }
```

Or measure directly:

```javascript
const node = document.querySelector('[data-id="my_node"]');
const domHeight = node.getBoundingClientRect().height;
console.log(`DOM height: ${domHeight}px`);
```

---

## Edge & Handle System

### Handle Positioning

React Flow edges connect via **handles** - invisible anchor points on nodes.

Our handles are **invisible** (not `display: none`):
```css
/* Custom node handles */
.react-flow__handle { 
    @apply !w-2 !h-2 !opacity-0;  /* 8x8px, invisible but present */
}
```

### Source Handle Position

For **source handles** (edge starts), we position at node bottom-center:

```javascript
// In html_generator.py, nodeTypes definition
<Handle 
    type="source" 
    position="bottom" 
    id="source-handle"
    className="!w-2 !h-2 !opacity-0"
/>
```

The edge `sourceY` = `nodeY + nodeHeight - (handleHeight / 2)`

### Target Handle Position

For **target handles** (edge ends), we position at node top-center:

```javascript
<Handle 
    type="target" 
    position="top" 
    id="target-handle"
    className="!w-2 !h-2 !opacity-0"
/>
```

### SSR-Style Handle Definition

React Flow supports `handles` property on nodes for server-side rendering:

```javascript
// Not currently used but available
node.handles = [
    { id: 'source', type: 'source', position: 'bottom', x: 0.5, y: 1.0, width: 8, height: 8 },
    { id: 'target', type: 'target', position: 'top', x: 0.5, y: 0.0, width: 8, height: 8 }
];
```

### Edge Path Calculation

React Flow calculates edge paths using:
1. Source node position + height → sourceY
2. Target node position → targetY
3. `pathfindingEdge` or `bezier` curve algorithms

The `d` attribute of edge `<path>` elements contains the SVG path:
```javascript
// Parse source coordinates from path
const path = document.querySelector('.react-flow__edge path');
const d = path.getAttribute('d');  // "M100,200 C100,250 200,150 200,100"
// M100,200 → source at (100, 200)
```

---

## Debugging Tools

### Browser Debug API

The visualization exposes debugging tools via `window.HyperNodesVizState.debug`:

```javascript
// Enable verbose console logging
HyperNodesVizState.debug.enableDebug()

// Analyze current graph state
HyperNodesVizState.debug.analyzeState()
// → { nodes: [...], edges: [...], pipelines: [...], danglingEdges: [...] }

// Get pipeline expansion states
HyperNodesVizState.debug.getExpansionState()
// → { "rag_pipeline": false, "retrieval_step": true }

// Simulate edge compression
HyperNodesVizState.debug.simulateCompression({ "rag_pipeline": false })

// Validate edge-node connections (THE KEY DEBUGGING TOOL)
HyperNodesVizState.debug.validateConnections()
// → { valid: true/false, issues: [...], summary: "..." }

// Get detailed layout info
HyperNodesVizState.debug.inspectLayout()
// → { nodes: [{id, x, y, width, height, bottom}], edges: [...], edgePaths: [...] }

// Show visual debug overlays (node boxes + edge points)
HyperNodesVizState.debug.showOverlays()

// Get full diagnostic report
HyperNodesVizState.debug.fullReport()
```

### Visual Debug Overlays

Enable overlays to see:
- **Red dashed boxes** around each node with ID labels
- **Green circles** at edge source points
- **Blue circles** at edge target points
- **Coordinate labels** showing (sourceX, sourceY) → (targetX, targetY)

Enable via:
1. **Console**: `HyperNodesVizState.debug.showOverlays()`
2. **UI**: Click bug icon in view controls
3. **URL**: Add `?debug=overlays`

### Python State Simulator

Test state transformations without a browser:

```python
from hypernodes.viz import (
    UIHandler,
    simulate_state,
    verify_edge_alignment,
    simulate_collapse_expand_cycle,
)

handler = UIHandler(pipeline, depth=99)
graph = handler.get_visualization_data(traverse_collapsed=True)

# Simulate collapsed state
result = simulate_state(
    graph,
    expansion_state={"my_pipeline": False},
    separate_outputs=False,
)

# Verify edges are valid
alignment = verify_edge_alignment(result)
if not alignment["valid"]:
    print("Issues:", alignment["issues"])

# Test full collapse/expand cycle
cycle = simulate_collapse_expand_cycle(graph, "my_pipeline")
print(cycle["summary"])
```

---

## Testing with Playwright

### Setup

```bash
pip install playwright
playwright install chromium
```

### Basic Test Pattern

```python
import tempfile
from playwright.sync_api import sync_playwright

from hypernodes.viz.js.html_generator import generate_widget_html

def test_edge_coordinates():
    # Generate HTML
    html = generate_widget_html(rf_data)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        f.write(html.encode())
        html_path = f.name
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"file://{html_path}")
        
        # Wait for React Flow to render
        page.wait_for_selector('.react-flow__node', timeout=10000)
        page.wait_for_timeout(2000)  # Wait for ELK layout
        
        # Click to collapse a pipeline
        collapse_btn = page.locator('[data-id="my_pipeline"] button').first
        collapse_btn.click()
        page.wait_for_timeout(1000)  # Wait for re-layout
        
        # Validate connections using debug API
        validation = page.evaluate(
            "() => HyperNodesVizState.debug.validateConnections()"
        )
        
        assert validation["valid"], f"Issues: {validation['issues']}"
        
        browser.close()
```

### Key Selectors

```javascript
// Nodes
'[data-id="node_name"]'                    // Specific node by ID
'.react-flow__node'                         // All nodes
'.react-flow__node[data-id*="pipeline"]'   // Nodes with "pipeline" in ID

// Edges
'.react-flow__edge'                         // All edges
'.react-flow__edge path'                    // Edge SVG paths
'[data-testid*="source_target"]'           // Edge by source/target

// Buttons
'[data-id="my_pipeline"] button'           // Collapse/expand button

// Handles
'.react-flow__handle--source'              // Source handles
'.react-flow__handle--target'              // Target handles
```

### Extracting Coordinates

```javascript
// Get node position from transform
const node = document.querySelector('[data-id="my_node"]');
const transform = node.style.transform;  // "translate(100px, 200px)"
const match = transform.match(/translate\(([^,]+)px,\s*([^)]+)px\)/);
const [x, y] = [parseFloat(match[1]), parseFloat(match[2])];

// Get edge source Y from path
const path = document.querySelector('.react-flow__edge path');
const d = path.getAttribute('d');  // "M100,200 C..."
const sourceY = parseFloat(d.match(/M([^,]+),([^\s]+)/)[2]);
```

---

## Common Issues & Solutions

### Issue: Edge doesn't connect to node after collapse

**Symptom**: Edge appears to "float" above or below the collapsed node.

**Cause**: ELK height calculation doesn't match actual DOM height.

**Solution**: Update height formula in `html_generator.py`:

```javascript
// Before (wrong)
height = 40 + (outputs.length * 24);  // = 64px for 1 output

// After (correct)
height = 60 + 38 + ((outputs.length - 1) * 24);  // = 98px for 1 output
```

**Verification**:
```javascript
HyperNodesVizState.debug.validateConnections()
```

### Issue: Edges "jump" after collapse/expand

**Symptom**: Edges briefly show in wrong position, then correct themselves.

**Cause**: React Flow caches edge positions. Need to force re-render.

**Solution**: Use `expansionKey` suffix on edge IDs:

```javascript
// In html_generator.py
const expansionKey = Object.entries(expansionState)
    .map(([k, v]) => `${k}:${v}`)
    .join(',');

edges = edges.map(e => ({
    ...e,
    id: `${e.id}__${expansionKey}`  // Forces re-render on state change
}));
```

### Issue: Handle positions not updating

**Symptom**: `updateNodeInternals` called but handles stay in old positions.

**Cause**: Handle positions are relative to node container. If container height is wrong, handle is outside visible area.

**Solution**: Ensure ELK height matches DOM height (see first issue).

### Issue: Missing edges when pipeline collapses

**Symptom**: Edge from input to internal node disappears when pipeline collapses.

**Cause**: `compressEdges()` can't find visible ancestor for internal node.

**Solution**: Ensure `getVisibleAncestor()` walks up parent chain correctly:

```javascript
function getVisibleAncestor(nodeId, nodesMap, expansionState) {
    let current = nodeId;
    while (current) {
        const node = nodesMap.get(current);
        if (node?.data?.isHidden === false) return current;
        current = node?.parentNode;
    }
    return null;
}
```

---

## Do's and Don'ts

### ✅ DO's

1. **DO measure actual DOM heights** when updating ELK height formulas
   ```javascript
   const actual = node.getBoundingClientRect().height;
   ```

2. **DO use `traverse_collapsed=True`** for interactive visualizations
   ```python
   graph_data = handler.get_visualization_data(traverse_collapsed=True)
   ```

3. **DO validate edge alignment** after any height/positioning changes
   ```javascript
   HyperNodesVizState.debug.validateConnections()
   ```

4. **DO test both display modes** (`separateOutputs=true` and `false`)

5. **DO use invisible handles** (`!opacity-0`) not `display: none`

6. **DO force edge re-render** via ID changes when expansion state changes

7. **DO wait for ELK layout** in Playwright tests:
   ```python
   page.wait_for_timeout(2000)  # After page load
   page.wait_for_timeout(1000)  # After collapse/expand
   ```

### ❌ DON'Ts

1. **DON'T assume ELK heights match DOM heights** - always verify

2. **DON'T use `display: none` on handles** - breaks edge calculations

3. **DON'T skip waiting for layout in tests** - leads to flaky tests

4. **DON'T modify React Flow nodes in place** - use spread for state updates:
   ```javascript
   // Wrong
   node.data.isExpanded = false;
   
   // Right
   const updatedNode = { ...node, data: { ...node.data, isExpanded: false } };
   ```

5. **DON'T forget about React Flow version** - we use v11.10.1, NOT v12

6. **DON'T filter edges before compression** - order matters:
   ```javascript
   // Wrong: filter, then compress
   // Right: compress, then filter
   ```

7. **DON'T trust Tailwind classes alone for sizing** - verify actual computed values

---

## File Reference

### Python (Generation)

| File | Purpose |
|------|---------|
| `src/hypernodes/viz/graph_walker.py` | Traverse pipeline DAG |
| `src/hypernodes/viz/ui_handler.py` | Manage depth, expansion, serialization |
| `src/hypernodes/viz/structures.py` | Data classes: FunctionNode, PipelineNode, etc. |
| `src/hypernodes/viz/js/renderer.py` | Transform to React Flow format |
| `src/hypernodes/viz/js/html_generator.py` | **THE MAIN FILE** - generates HTML with embedded JS |
| `src/hypernodes/viz/state_simulator.py` | Python-side state transformation testing |

### JavaScript (Runtime)

| File | Purpose |
|------|---------|
| `assets/viz/state_utils.js` | applyState, compressEdges, groupInputs |
| `assets/viz/theme_utils.js` | Theme detection (VS Code integration) |

### Tests

| File | Purpose |
|------|---------|
| `tests/viz/test_edge_alignment_playwright.py` | Playwright browser tests for edge validation |
| `tests/viz/test_edge_coordinates.py` | Coordinate verification after collapse |
| `tests/viz/test_collapse_coordinates.py` | Comprehensive collapse coordinate tests |
| `tests/viz/test_debug_tools.py` | Debug API tests |
| `tests/viz/test_state_utils_visibility.py` | Visibility logic tests |
| `tests/viz/test_collapsed_pipelines_no_dangling_edges.py` | Edge compression tests |

### Example Test Scripts

```bash
# Run all viz tests
uv run pytest tests/viz/ -v

# Run specific test
uv run pytest tests/viz/test_edge_alignment_playwright.py -v

# Run Python-only tests (no browser)
uv run pytest tests/viz/test_edge_alignment_playwright.py::TestPythonEdgeAlignment -v

# Generate test HTML for manual inspection
uv run python scripts/test_viz_comprehensive.py
```

---

## Quick Reference Card

### Debug Commands (Browser Console)

```javascript
// Enable debug logging
HyperNodesVizState.debug.enableDebug()

// The one command you need most often
HyperNodesVizState.debug.validateConnections()

// See everything
HyperNodesVizState.debug.fullReport()

// Visual overlays
HyperNodesVizState.debug.showOverlays()
```

### Height Formula (html_generator.py)

```javascript
// Nodes with combined outputs
height = 60 + 38 + ((outputs.length - 1) * 24)
// 1 output  → 98px
// 2 outputs → 122px
// 3 outputs → 146px
```

### Key Locations to Edit

| What | Where |
|------|-------|
| Node height calculation | `html_generator.py` → `useLayout` hook (~lines 800-860) |
| Edge compression logic | `assets/viz/state_utils.js` → `compressEdges()` |
| Custom node rendering | `html_generator.py` → `nodeTypes` definition |
| Debug API | `html_generator.py` → `HyperNodesVizState.debug` object |
