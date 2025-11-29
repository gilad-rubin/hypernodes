# Node Width Calculation & Text Truncation Guide

A guide to understanding and debugging the dynamic width calculation system for HyperNodes visualization nodes.

---

## Table of Contents

1. [Overview](#overview)
2. [Layout Constants](#layout-constants)
3. [Width Calculation Logic](#width-calculation-logic)
4. [Text Truncation](#text-truncation)
5. [Common Property Name Mismatch Bug](#common-property-name-mismatch-bug)
6. [Testing for Truncation Issues](#testing-for-truncation-issues)
7. [Debugging Checklist](#debugging-checklist)

---

## Overview

Node widths are calculated **dynamically** based on content:
- Node label (function/pipeline name)
- Output names (when `separateOutputs=false`)
- Type hints (when `showTypes=true`)

All widths are capped by `MAX_NODE_WIDTH` to ensure visual consistency.

### Key Principle

> **The node width should accommodate the LONGEST text it displays.**
> 
> For function nodes with combined outputs, this is:
> `max(nodeLabel, outputName + typeHint)`

---

## Layout Constants

Located in `html_generator.py`, inside the JavaScript section:

```javascript
// === LAYOUT CONSTANTS ===

// Truncation limit for type hints (both display and width calculation)
const TYPE_HINT_MAX_CHARS = 25;

// Truncation limit for node labels (function names, pipeline names)
const NODE_LABEL_MAX_CHARS = 25;

// Character width estimate for monospace font (text-xs ~12px)
const CHAR_WIDTH_PX = 7;

// Base padding for DATA/INPUT nodes: px-3 (12px) * 2 + icon (12px) + gaps (16px)
const NODE_BASE_PADDING = 52;

// Base padding for FUNCTION/PIPELINE nodes (simpler without label)
const FUNCTION_NODE_BASE_PADDING = 48;

// Maximum node width to ensure uniform appearance
const MAX_NODE_WIDTH = 280;
```

### Padding Breakdown

**DATA/INPUT nodes** (`NODE_BASE_PADDING = 52px`):
```
┌─ Node ──────────────────────────────────────┐
│ px-3 │ icon │ gap │ text │ gap │ type │ px-3 │
│ 12px │ 12px │ 8px │ var  │ 8px │ var  │ 12px │
└─────────────────────────────────────────────┘
```

**FUNCTION/PIPELINE nodes** (`FUNCTION_NODE_BASE_PADDING = 48px`):
```
┌─ Header ────────────────────────────────────┐
│ px-3 │           node name            │ px-3 │
│ 12px │            var                 │ 12px │
├─────────────────────────────────────────────┤
│ px-2 │ → │ output name │ : │ type │ px-2 │
│  8px │ 2 │    var      │ 2 │ var  │  8px │
└─────────────────────────────────────────────┘
```

---

## Width Calculation Logic

### For DATA/INPUT Nodes

```javascript
const labelLen = Math.min(n.data.label ? n.data.label.length : 0, NODE_LABEL_MAX_CHARS);
const typeLen = (n.data.showTypes && n.data.typeHint) 
    ? Math.min(n.data.typeHint.length, TYPE_HINT_MAX_CHARS) + 2  // +2 for ": "
    : 0;
width = Math.min(MAX_NODE_WIDTH, (labelLen + typeLen) * CHAR_WIDTH_PX + NODE_BASE_PADDING);
```

### For FUNCTION/PIPELINE Nodes with Combined Outputs

**⚠️ CRITICAL**: This is where the property name mismatch bug can occur!

```javascript
const labelLen = Math.min(n.data.label ? n.data.label.length : 0, NODE_LABEL_MAX_CHARS);
let maxContentLen = labelLen;

// Consider output widths if combined (separateOutputs=false)
// Note: outputs from fallbackApplyState have { name, type }, NOT { label, typeHint }
const outputs = n.data.outputs || [];
if (!n.data.separateOutputs && outputs.length > 0) {
    outputs.forEach(o => {
        // Handle BOTH property naming conventions
        const outName = o.name || o.label || '';
        const outType = o.type || o.typeHint || '';
        const outLabelLen = Math.min(outName.length, NODE_LABEL_MAX_CHARS);
        const outTypeLen = (n.data.showTypes && outType) 
            ? Math.min(outType.length, TYPE_HINT_MAX_CHARS) + 2 
            : 0;
        const totalLen = outLabelLen + outTypeLen + 4;  // +4 for arrow "→ " and spacing
        if (totalLen > maxContentLen) maxContentLen = totalLen;
    });
}
width = Math.min(MAX_NODE_WIDTH, maxContentLen * CHAR_WIDTH_PX + FUNCTION_NODE_BASE_PADDING);
```

### Width Calculation Flow

```
1. Python: JSRenderer creates nodes WITHOUT outputs data
   └─ Function node: { nodeType: "FUNCTION", label: "retrieve" }
   └─ Output node: { nodeType: "DATA", label: "retrieved_documents", sourceId: "retrieve" }

2. JavaScript: fallbackApplyState() adds outputs to function nodes
   └─ Function node now has: { outputs: [{ name: "retrieved_documents", type: "str" }] }
   └─ Note: uses 'name' and 'type', NOT 'label' and 'typeHint'

3. JavaScript: mapToElk() calculates width using n.data.outputs
   └─ Must handle both property naming conventions!
```

---

## Text Truncation

### Display Truncation

```javascript
// For type hints
const truncateTypeHint = (type) => type && type.length > TYPE_HINT_MAX_CHARS 
    ? type.substring(0, TYPE_HINT_MAX_CHARS) + '...' 
    : type;

// For node labels
const truncateLabel = (label) => label && label.length > NODE_LABEL_MAX_CHARS
    ? label.substring(0, NODE_LABEL_MAX_CHARS) + '...'
    : label;
```

### What Gets Truncated

| Content | Truncation | Max Chars |
|---------|------------|-----------|
| Node labels | Yes | 25 (`NODE_LABEL_MAX_CHARS`) |
| Output names | Yes | 25 (`NODE_LABEL_MAX_CHARS`) |
| Type hints | Yes | 25 (`TYPE_HINT_MAX_CHARS`) |
| Input names | Yes | 25 (`NODE_LABEL_MAX_CHARS`) |

### Priority: Names Over Types

Output/input **names** should be fully visible if possible. Type hints can be truncated more aggressively since they have a tooltip.

```
✅ GOOD: "→ retrieved_documents : Dict[str, Li..."
❌ BAD:  "→ retrieved_d... : str"
```

---

## Common Property Name Mismatch Bug

### The Problem

`fallbackApplyState()` creates outputs with `{ name, type }`:

```javascript
// In fallbackApplyState()
functionOutputs[n.data.sourceId].push({ 
    name: n.data.label,     // ← 'name' not 'label'
    type: n.data.typeHint   // ← 'type' not 'typeHint'
});
```

But width calculation might look for `{ label, typeHint }`:

```javascript
// ❌ WRONG - these will always be undefined!
const outLabelLen = Math.min(o.label ? o.label.length : 0, NODE_LABEL_MAX_CHARS);
const outTypeLen = (n.data.showTypes && o.typeHint) ? ...

// ✅ CORRECT - handle both conventions
const outName = o.name || o.label || '';
const outType = o.type || o.typeHint || '';
```

### Symptom

- Function node is sized only for its label, not its outputs
- Output text gets clipped/truncated even when it's short
- Example: "retrieve" node (8 chars) with "retrieved_documents" output (19 chars)
  - Before fix: node width = 104px (based on "retrieve")
  - After fix: node width = 244px (based on "retrieved_documents")

### How to Detect

```javascript
// In browser console
const funcNodes = HyperNodesVizState.debug.analyzeState().nodes
    .filter(n => n.data?.nodeType === 'FUNCTION');
    
funcNodes.forEach(n => {
    console.log(`${n.id}: outputs =`, n.data.outputs);
    if (n.data.outputs?.[0]) {
        console.log('  Has name?', 'name' in n.data.outputs[0]);
        console.log('  Has label?', 'label' in n.data.outputs[0]);
    }
});
```

---

## Testing for Truncation Issues

### Playwright Test Pattern

```python
def test_output_not_truncated(self, browser_page):
    """Test that output names are fully visible."""
    @node(output_name="retrieved_documents")
    def retrieve(query: str) -> str:
        return query
    
    pipeline = Pipeline(nodes=[retrieve])
    html = self._generate_combined_html(pipeline)
    self._save_and_open(browser_page, html)
    
    # Check if output text is clipped by node container
    result = browser_page.evaluate("""() => {
        const outputSections = document.querySelectorAll('[class*="border-t"]');
        const issues = [];
        
        outputSections.forEach(section => {
            const row = section.querySelector('[class*="flex"][class*="items-center"]');
            if (!row) return;
            
            // Find parent node container
            let nodeContainer = section.parentElement;
            while (nodeContainer && !nodeContainer.classList.contains('react-flow__node')) {
                nodeContainer = nodeContainer.parentElement;
            }
            if (!nodeContainer) return;
            
            const nodeWidth = nodeContainer.offsetWidth;
            const contentWidth = row.scrollWidth;
            
            // Issue if content is wider than node
            if (contentWidth > nodeWidth + 5) {
                issues.push({
                    nodeWidth,
                    contentWidth,
                    text: row.innerText,
                    issue: 'Node too narrow for output content'
                });
            }
        });
        
        return { issues, hasIssues: issues.length > 0 };
    }""")
    
    assert not result['hasIssues'], f"Issues: {result['issues']}"
```

### Key Test Scenarios

1. **Short function name, long output name**
   ```python
   @node(output_name="retrieved_documents")  # 19 chars
   def fn(x: str) -> str:  # "fn" is 2 chars
       return x
   ```

2. **Multiple outputs with varying lengths**
   ```python
   @node(output_name=("short", "very_long_output_name"))
   def multi(x: str) -> tuple:
       return ("", "")
   ```

3. **With and without type hints**
   - `show_types=True`: Width includes type
   - `show_types=False`: Width based only on name

4. **Collapsed pipeline with outputs**
   - Same rules apply to collapsed PIPELINE nodes

### Visual Verification

```python
# Generate test HTML
from hypernodes.viz import UIHandler
from hypernodes.viz.js.renderer import JSRenderer
from hypernodes.viz.js.html_generator import generate_widget_html

handler = UIHandler(pipeline, depth=99)
graph_data = handler.get_visualization_data(traverse_collapsed=True)
renderer = JSRenderer()
rf_data = renderer.render(graph_data, theme='dark', separate_outputs=False, show_types=True)
html = generate_widget_html(rf_data)

with open('outputs/test.html', 'w') as f:
    f.write(html)
```

---

## Debugging Checklist

### When Output Text is Truncated

1. **Check property names in outputs**
   ```javascript
   // Console
   const node = HyperNodesVizState.debug.analyzeState().nodes
       .find(n => n.id === 'your_function');
   console.log('Outputs:', node.data.outputs);
   // Should show: [{ name: "...", type: "..." }]
   ```

2. **Verify width calculation sees outputs**
   - Add `console.log` in mapToElk:
   ```javascript
   console.log(`${n.id}: outputs =`, n.data.outputs, 'maxContentLen =', maxContentLen);
   ```

3. **Check showTypes and separateOutputs flags**
   ```javascript
   console.log('showTypes:', n.data.showTypes);
   console.log('separateOutputs:', n.data.separateOutputs);
   ```

4. **Measure actual vs calculated width**
   ```javascript
   const node = document.querySelector('[data-id="your_function"]');
   console.log('DOM width:', node.offsetWidth);
   // Compare to ELK calculated width
   ```

### When Node is Too Wide

1. Check if MAX_NODE_WIDTH is being applied
2. Verify truncation is happening for long texts
3. Check padding constants aren't too large

### Common Fixes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Output text clipped | Property mismatch | Use `o.name \|\| o.label` |
| Node too narrow | showTypes not considered | Check conditional type length |
| All nodes same width | Using MAX instead of Math.min | Check width = Math.min(...) |
| Width ignores outputs | separateOutputs check wrong | Verify `!n.data.separateOutputs` |

---

## Quick Reference

### Width Formula

```javascript
// DATA/INPUT nodes
width = Math.min(MAX_NODE_WIDTH, (labelLen + typeLen) * CHAR_WIDTH_PX + NODE_BASE_PADDING)

// FUNCTION/PIPELINE nodes (with outputs)
width = Math.min(MAX_NODE_WIDTH, maxContentLen * CHAR_WIDTH_PX + FUNCTION_NODE_BASE_PADDING)

// Where:
// - CHAR_WIDTH_PX = 7
// - NODE_BASE_PADDING = 52
// - FUNCTION_NODE_BASE_PADDING = 48
// - MAX_NODE_WIDTH = 280
// - TYPE_HINT_MAX_CHARS = 25
// - NODE_LABEL_MAX_CHARS = 25
```

### Example Calculations

| Content | Calculation | Width |
|---------|-------------|-------|
| "retrieve" (8 chars) | 8 × 7 + 48 = 104 | 104px |
| "retrieved_documents" (19 chars) | 19 × 7 + 48 = 181 | 181px |
| "retrieved_documents : str" (23+3 chars + 4 arrow) | 30 × 7 + 48 = 258 | 258px |
| Very long text (40 chars) | Math.min(280, 40 × 7 + 48) | 280px (capped) |

### Key Files

| File | What to Change |
|------|----------------|
| `html_generator.py` | Width calculation in `mapToElk`, constants, truncation functions |
| `tests/viz/test_output_truncation_playwright.py` | Playwright tests for visual truncation |
| `scripts/test_truncation_uniform.py` | Generate test HTML files |

