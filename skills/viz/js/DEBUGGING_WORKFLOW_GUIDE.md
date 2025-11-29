# Visualization Debugging Workflow Guide

A practical guide for debugging visualization issues in HyperNodes, from initial discovery to fix verification.

---

## Table of Contents

1. [Debugging Philosophy](#debugging-philosophy)
2. [The Debugging Workflow](#the-debugging-workflow)
3. [Step 1: Reproduce the Issue](#step-1-reproduce-the-issue)
4. [Step 2: Add Visibility to the Problem](#step-2-add-visibility-to-the-problem)
5. [Step 3: Write a Failing Test](#step-3-write-a-failing-test)
6. [Step 4: Trace Through the Code](#step-4-trace-through-the-code)
7. [Step 5: Fix and Verify](#step-5-fix-and-verify)
8. [Adding Debug Tools for New Features](#adding-debug-tools-for-new-features)
9. [Playwright Testing Patterns](#playwright-testing-patterns)
10. [Common Pitfalls](#common-pitfalls)
11. [Quick Debugging Recipes](#quick-debugging-recipes)

---

## Debugging Philosophy

### Key Principles

1. **Make the invisible visible** - Add logging, overlays, and data exposure before trying to fix
2. **Write a failing test first** - If you can't write a test that fails, you don't understand the bug
3. **Don't trust assumptions** - Verify property names, values, and data flow at each step
4. **Compare expected vs actual** - Always know what you expect and what you're getting

### The Three Layers

Visualization bugs can occur at three layers:

```
┌─────────────────────────────────────────┐
│ 1. Python Generation (build time)       │  ← Data structure issues
│    UIHandler → JSRenderer → HTML        │
├─────────────────────────────────────────┤
│ 2. JavaScript Transformation (runtime)  │  ← State transformation bugs
│    fallbackApplyState → mapToElk        │
├─────────────────────────────────────────┤
│ 3. React/CSS Rendering (visual)         │  ← Layout/style issues
│    CustomNode → React Flow → DOM        │
└─────────────────────────────────────────┘
```

---

## The Debugging Workflow

```
┌─────────────────┐
│ 1. REPRODUCE    │  Create minimal reproduction case
└────────┬────────┘
         ↓
┌─────────────────┐
│ 2. VISIBILITY   │  Add logging/debug tools to see what's happening
└────────┬────────┘
         ↓
┌─────────────────┐
│ 3. FAILING TEST │  Write Playwright test that detects the issue
└────────┬────────┘
         ↓
┌─────────────────┐
│ 4. TRACE        │  Follow data through Python → JS → DOM
└────────┬────────┘
         ↓
┌─────────────────┐
│ 5. FIX & VERIFY │  Apply fix, ensure test passes
└─────────────────┘
```

---

## Step 1: Reproduce the Issue

### Create a Minimal Test Case

```python
# scripts/debug_my_issue.py
from hypernodes import Pipeline, node
from hypernodes.viz import UIHandler
from hypernodes.viz.js.renderer import JSRenderer
from hypernodes.viz.js.html_generator import generate_widget_html

# Minimal reproduction - use the SIMPLEST pipeline that shows the bug
@node(output_name="result")
def problematic_node(x: str) -> str:
    return x

pipeline = Pipeline(nodes=[problematic_node])

# Generate with the specific settings that trigger the bug
handler = UIHandler(pipeline, depth=99)
graph_data = handler.get_visualization_data(traverse_collapsed=True)
renderer = JSRenderer()
rf_data = renderer.render(
    graph_data,
    theme="dark",
    separate_outputs=False,  # Try both True and False
    show_types=True,         # Try both True and False
)
html = generate_widget_html(rf_data)

# Save for inspection
with open('outputs/debug_issue.html', 'w') as f:
    f.write(html)
print('Saved: outputs/debug_issue.html')
print('Open in browser and check console/debug tools')
```

Run it:
```bash
uv run python scripts/debug_my_issue.py
```

### Open and Inspect

1. Open `outputs/debug_issue.html` in browser
2. Open DevTools (F12)
3. Check Console for errors
4. Enable debug mode: `HyperNodesVizState.debug.enableDebug()`

---

## Step 2: Add Visibility to the Problem

### Use Existing Debug Tools

```javascript
// Browser console - always start here
HyperNodesVizState.debug.enableDebug()      // Enable verbose logging
HyperNodesVizState.debug.analyzeState()     // See current state
HyperNodesVizState.debug.showOverlays()     // Visual debug overlays
HyperNodesVizState.debug.fullReport()       // Everything at once
```

### Add Temporary Console Logging

In `html_generator.py`, add logging at key points:

```javascript
// In mapToElk function
console.log('mapToElk node:', n.id, {
    nodeType: n.data?.nodeType,
    outputs: n.data?.outputs,
    separateOutputs: n.data?.separateOutputs,
    showTypes: n.data?.showTypes,
});

// In fallbackApplyState
console.log('fallbackApplyState:', {
    separateOutputs,
    outputNodes: [...outputNodes],
    functionOutputs,
});
```

### Check Python Data Before JS

```python
# In your debug script, inspect intermediate data
import json

print("=== Graph Data ===")
for node in graph_data.nodes:
    print(f"  {node.id}: {type(node).__name__}")

print("\n=== RF Data (React Flow format) ===")
for n in rf_data['nodes']:
    print(f"  {n['id']}: {n['data']}")
```

---

## Step 3: Write a Failing Test

### The Test Should Detect the Bug

```python
# tests/viz/test_my_issue.py
import pytest
import tempfile
from hypernodes import Pipeline, node
from hypernodes.viz import UIHandler
from hypernodes.viz.js.renderer import JSRenderer
from hypernodes.viz.js.html_generator import generate_widget_html


class TestMyIssue:
    """Test for the specific issue I'm debugging."""

    @pytest.fixture
    def browser_page(self):
        """Create a Playwright browser page."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            pytest.skip("Playwright not installed")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            yield page
            browser.close()

    def _generate_and_open(self, page, pipeline, **render_kwargs):
        """Helper to generate HTML and open in browser."""
        handler = UIHandler(pipeline, depth=99)
        graph_data = handler.get_visualization_data(traverse_collapsed=True)
        renderer = JSRenderer()
        rf_data = renderer.render(graph_data, theme="dark", **render_kwargs)
        html = generate_widget_html(rf_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html)
            temp_path = f.name
        
        page.goto(f"file://{temp_path}")
        page.wait_for_timeout(2000)  # Wait for ELK layout
        return temp_path

    def test_the_specific_issue(self, browser_page):
        """This test should FAIL before the fix and PASS after."""
        @node(output_name="my_output")
        def my_node(x: str) -> str:
            return x
        
        pipeline = Pipeline(nodes=[my_node])
        self._generate_and_open(
            browser_page, 
            pipeline,
            separate_outputs=False,
            show_types=True,
        )
        
        # Use JavaScript to check for the issue
        result = browser_page.evaluate("""() => {
            // Detect the specific problem
            const issues = [];
            
            // Example: Check if something is wrong
            document.querySelectorAll('.some-selector').forEach(el => {
                const expected = 'something';
                const actual = el.getAttribute('data-something');
                if (actual !== expected) {
                    issues.push({
                        expected,
                        actual,
                        issue: 'Values do not match'
                    });
                }
            });
            
            return {
                issues,
                hasIssues: issues.length > 0,
            };
        }""")
        
        print(f"Issues found: {result['issues']}")
        
        # This assertion should FAIL before your fix
        assert not result['hasIssues'], f"Issues detected: {result['issues']}"
```

### Run the Test

```bash
# Run with verbose output
uv run pytest tests/viz/test_my_issue.py -v -s

# If it passes, your test doesn't detect the bug!
# If it fails, you've got a good test
```

---

## Step 4: Trace Through the Code

### Trace Data Flow

```
Python                          JavaScript
───────────────────────────────────────────────────────────
1. UIHandler.get_visualization_data()
   └─ Creates VisualizationGraph with nodes/edges
   
2. JSRenderer.render()
   └─ Transforms to React Flow format
   └─ Nodes have: { id, data: { nodeType, label, ... } }
   
                                3. fallbackApplyState()
                                   └─ Adds outputs to function nodes
                                   └─ Creates: { name, type } (NOT label, typeHint!)
                                   
                                4. mapToElk()
                                   └─ Calculates width/height
                                   └─ Uses n.data.outputs
                                   
                                5. React Flow renders
                                   └─ CustomNode component
                                   └─ CSS applied
```

### Key Inspection Points

**Python side:**
```python
# Check what Python generates
print(json.dumps(rf_data['nodes'][0], indent=2))
```

**JavaScript side:**
```javascript
// Check what JS receives
console.log('Initial nodes:', initialNodes);

// Check after transformation
const transformed = fallbackApplyState(nodes, edges, options);
console.log('After transform:', transformed.nodes);

// Check in mapToElk
console.log('In mapToElk:', n.id, n.data);
```

### Find the Disconnect

The bug is usually where **expected** diverges from **actual**:

```javascript
// Add assertions that will throw if assumption is wrong
const outputs = n.data.outputs || [];
console.assert(outputs.length > 0, `Expected outputs for ${n.id}`);
console.assert('name' in outputs[0], `Expected 'name' property in output`);
```

---

## Step 5: Fix and Verify

### Apply the Fix

Based on what you found, apply the fix in the appropriate layer:

| Layer | File | Common Fixes |
|-------|------|--------------|
| Python | `renderer.py`, `graph_walker.py` | Data structure, property names |
| JS Transform | `html_generator.py` (JS section) | State transformation logic |
| CSS/Rendering | `html_generator.py` (component section) | Layout, styling, visibility |

### Verify the Fix

```bash
# 1. Run your specific test
uv run pytest tests/viz/test_my_issue.py -v -s

# 2. Run all viz tests to check for regressions
uv run pytest tests/viz/ -v --ignore=tests/viz/test_edge_alignment_playwright.py

# 3. Visual verification
uv run python scripts/debug_my_issue.py
# Open outputs/debug_issue.html and check manually
```

### Clean Up

1. Remove temporary console.log statements
2. Keep the test (it's now a regression test!)
3. Update documentation if you found something non-obvious

---

## Adding Debug Tools for New Features

### Pattern: Add to HyperNodesVizState.debug

In `html_generator.py`, find the debug object and add your tool:

```javascript
window.HyperNodesVizState = {
    // ... existing state ...
    
    debug: {
        // ... existing debug methods ...
        
        // ADD YOUR NEW DEBUG TOOL HERE
        inspectMyFeature: () => {
            const nodes = getNodesFromSomewhere();
            const analysis = nodes.map(n => ({
                id: n.id,
                myProperty: n.data?.myProperty,
                computed: calculateSomething(n),
            }));
            console.table(analysis);
            return analysis;
        },
    }
};
```

### Pattern: Add Visual Overlay

In the `DebugOverlay` component:

```javascript
const DebugOverlay = ({ nodes, edges, enabled, theme }) => {
    // ... existing code ...
    
    // Add new visualization
    return html`
        <div>
            <!-- Existing overlays -->
            
            <!-- Your new overlay -->
            ${nodes.map(n => {
                if (!shouldShowMyOverlay(n)) return null;
                return html`
                    <div 
                        style=${{
                            position: 'absolute',
                            left: n.position.x,
                            top: n.position.y,
                            // ... styling
                        }}
                    >
                        My debug info: ${n.data.myProperty}
                    </div>
                `;
            })}
        </div>
    `;
};
```

### Pattern: Add Debug Tab

The debug panel has tabs. Add a new one:

```javascript
const [activeTab, setActiveTab] = useState('bounds'); // 'bounds', 'widths', 'texts', 'mytab'

// In the tab buttons
<button onClick=${() => setActiveTab('mytab')}>MY TAB</button>

// In the content area
${activeTab === 'mytab' ? html`
    <div>
        <h3>My Debug Info</h3>
        <table>
            ${nodes.map(n => html`
                <tr>
                    <td>${n.id}</td>
                    <td>${n.data.myProperty}</td>
                </tr>
            `)}
        </table>
    </div>
` : null}
```

---

## Playwright Testing Patterns

### Basic Test Structure

```python
class TestMyFeature:
    @pytest.fixture
    def browser_page(self):
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            yield page
            browser.close()

    def test_something(self, browser_page):
        # 1. Generate and open HTML
        # 2. Wait for render
        # 3. Evaluate JavaScript to check something
        # 4. Assert result
        pass
```

### Waiting Patterns

```python
# Wait for specific element
page.wait_for_selector('.react-flow__node', timeout=10000)

# Wait fixed time (after ELK layout)
page.wait_for_timeout(2000)

# Wait for text to appear
page.wait_for_selector('text=expected text')

# Wait for network idle (if loading external resources)
page.wait_for_load_state('networkidle')
```

### Interaction Patterns

```python
# Click a node
page.click('[data-id="my_node"]')

# Click collapse button
page.click('[data-id="my_pipeline"] button')

# Hover
page.hover('[data-id="my_node"]')

# Type in input
page.fill('input[name="search"]', 'query')
```

### Evaluation Patterns

```python
# Simple value
result = page.evaluate("() => document.title")

# Complex analysis
result = page.evaluate("""() => {
    const data = [];
    document.querySelectorAll('.my-class').forEach(el => {
        data.push({
            text: el.innerText,
            width: el.offsetWidth,
            isVisible: el.offsetParent !== null,
        });
    });
    return data;
}""")

# Use debug API
result = page.evaluate("() => HyperNodesVizState.debug.analyzeState()")
```

### Screenshot for Manual Verification

```python
# Take screenshot for debugging
page.screenshot(path='outputs/debug_screenshot.png')

# Screenshot specific element
page.locator('[data-id="my_node"]').screenshot(path='outputs/node.png')
```

### Checking Visual Properties

```python
# Check if element is visually clipped
result = page.evaluate("""() => {
    const el = document.querySelector('.my-element');
    return {
        isClipped: el.scrollWidth > el.clientWidth,
        scrollWidth: el.scrollWidth,
        clientWidth: el.clientWidth,
    };
}""")

# Check computed style
result = page.evaluate("""() => {
    const el = document.querySelector('.my-element');
    const style = getComputedStyle(el);
    return {
        width: style.width,
        overflow: style.overflow,
        display: style.display,
    };
}""")

# Check bounding box
bbox = page.locator('.my-element').bounding_box()
assert bbox['width'] > 100, "Element too narrow"
```

---

## Common Pitfalls

### Pitfall 1: Property Name Mismatch

**Symptom**: Data exists but code can't find it

**Example**: `fallbackApplyState` creates `{ name, type }` but code looks for `{ label, typeHint }`

**Prevention**: Always log the actual object structure:
```javascript
console.log('Output object:', JSON.stringify(output, null, 2));
```

**Fix Pattern**:
```javascript
// Handle both conventions
const value = obj.name || obj.label || '';
```

### Pitfall 2: Timing Issues

**Symptom**: Test passes sometimes, fails sometimes

**Cause**: Not waiting for async operations (ELK layout, React render)

**Prevention**: Always wait after actions:
```python
page.click('[data-id="pipeline"]')
page.wait_for_timeout(1000)  # Wait for re-layout
```

### Pitfall 3: DOM vs Visual State

**Symptom**: DOM shows correct text but it's visually clipped

**Cause**: CSS overflow:hidden clips content without changing DOM

**Detection**:
```javascript
// Wrong - always has full text
const hasText = el.innerText.includes('full text');

// Right - check actual visible width
const isClipped = el.scrollWidth > el.clientWidth;
```

### Pitfall 4: State Not Flowing Through

**Symptom**: Setting changes in one place but no effect elsewhere

**Cause**: State transformations happen in order, later stages don't see earlier changes

**Debug**:
```javascript
// Log at each transformation stage
console.log('After applyState:', nodes);
console.log('After applyVisibility:', nodes);
console.log('After compressEdges:', edges);
```

### Pitfall 5: React Flow Version Differences

**Symptom**: Code from docs doesn't work

**Cause**: We use v11.10.1, not v12

**Prevention**: Always check React Flow version in imports:
```javascript
// v11 (what we use)
import ReactFlow from '@xyflow/react'

// v12 (different API)
import { ReactFlow } from '@xyflow/react'
```

### Pitfall 6: ELK Height/Width vs DOM Height/Width

**Symptom**: Edges connect to wrong positions

**Cause**: ELK calculates size before render, DOM might render differently

**Debug**:
```javascript
// Compare ELK vs DOM
const elkHeight = node.height;  // From mapToElk
const domHeight = document.querySelector(`[data-id="${node.id}"]`).offsetHeight;
console.log(`ELK: ${elkHeight}, DOM: ${domHeight}, diff: ${domHeight - elkHeight}`);
```

---

## Quick Debugging Recipes

### Recipe: "Why is my output truncated?"

```bash
# 1. Generate test HTML
uv run python scripts/test_truncation_uniform.py

# 2. Open in browser, enable debug
# HyperNodesVizState.debug.enableDebug()

# 3. Check node width vs content width
# In console:
const node = document.querySelector('[data-id="my_function"]');
const output = node.querySelector('[class*="border-t"]');
console.log('Node width:', node.offsetWidth);
console.log('Content width:', output?.scrollWidth);

# 4. Check if outputs have correct properties
HyperNodesVizState.debug.analyzeState().nodes
    .filter(n => n.data?.nodeType === 'FUNCTION')
    .forEach(n => console.log(n.id, n.data.outputs));
```

### Recipe: "Why is my edge disconnected?"

```javascript
// 1. Enable overlays
HyperNodesVizState.debug.showOverlays()

// 2. Validate connections
HyperNodesVizState.debug.validateConnections()

// 3. Check specific edge
const edge = document.querySelector('[data-testid="edge-source-target"]');
const path = edge.querySelector('path');
console.log('Path d:', path.getAttribute('d'));
```

### Recipe: "Why doesn't my pipeline collapse?"

```javascript
// 1. Check expansion state
HyperNodesVizState.debug.getExpansionState()

// 2. Check if pipeline node has correct data
const pipeline = HyperNodesVizState.debug.analyzeState().nodes
    .find(n => n.id === 'my_pipeline');
console.log('isExpanded:', pipeline.data?.isExpanded);
console.log('nodeType:', pipeline.data?.nodeType);

// 3. Simulate collapse
HyperNodesVizState.debug.simulateCompression({ 'my_pipeline': false })
```

### Recipe: "Why is showTypes not working?"

```javascript
// 1. Check if flag is passed
const meta = JSON.parse(document.getElementById('graph-data').textContent).meta;
console.log('show_types in meta:', meta.show_types);

// 2. Check nodes
HyperNodesVizState.debug.analyzeState().nodes
    .forEach(n => console.log(n.id, 'showTypes:', n.data?.showTypes));

// 3. Check if it's reaching width calculation
// Add console.log in mapToElk:
// console.log(n.id, 'showTypes:', n.data.showTypes);
```

---

## Summary Checklist

When debugging visualization issues:

- [ ] Created minimal reproduction script
- [ ] Opened HTML in browser and checked console
- [ ] Enabled debug mode: `HyperNodesVizState.debug.enableDebug()`
- [ ] Used `analyzeState()` to see current data
- [ ] Wrote failing Playwright test
- [ ] Traced data through Python → JS → DOM
- [ ] Found where expected ≠ actual
- [ ] Applied fix
- [ ] Test passes
- [ ] All viz tests still pass
- [ ] Visual verification looks good
- [ ] Removed debug console.logs
- [ ] Updated documentation if needed

