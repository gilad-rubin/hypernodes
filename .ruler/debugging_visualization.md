# Debugging Visualization Issues

This guide documents how to debug and fix complex visualization issues in `hypernodes`, specifically focusing on layout synchronization and coordinate verification.

## Case Study: The "Hanging Arrow" Issue

### Symptom
When collapsing a pipeline node, the edges connected to it would sometimes appear "hanging" or disconnected, pointing to the old location of the node boundary instead of the new one. This visual glitch would resolve itself after a manual refresh or theme toggle.

### Root Cause
The issue was a **race condition** between CSS animations and the JavaScript layout engine:
1.  **User Action**: Click collapse.
2.  **JS**: React triggers a state update.
3.  **CSS**: A `transition` (300ms) starts animating the node's height.
4.  **JS**: React Flow/ELK attempts to recalculate the layout.
5.  **Problem**: The layout calculation happened *during* or *before* the animation finished. React Flow measured the handles at their *intermediate* positions, not their final positions.

### The Fix: `onTransitionEnd`
Instead of using `setTimeout` (which is unreliable) or forcing reflows, we use the native `onTransitionEnd` event.

**Implementation:**
In `CustomNode` (and other node components), we add a handler to the root element:

```jsx
const handleTransitionEnd = useCallback((e) => {
    // Ensure we only react to the root element's transition, not children
    if (e.target === e.currentTarget) {
        updateNodeInternals(id);
    }
}, [id, updateNodeInternals]);

return (
    <div 
        className="..." 
        onTransitionEnd={handleTransitionEnd}
    >
        {/* content */}
    </div>
);
```

This ensures `updateNodeInternals(id)` is called **exactly** when the animation completes and the DOM is in its final state.

## Debugging Tools

We have implemented a built-in **Debug Overlay** to diagnose these exact issues.

### Enabling Debug Mode
1.  Open the generated HTML visualization.
2.  Click the **Bug Icon** (🐞) in the top-left control panel.
3.  Or, append `?debug=true` to the URL.

### What It Shows
*   **Node Bounds Panel**: A table on the right showing the exact `Y`, `Height`, and `Bottom` coordinates of every node.
*   **Edge Coordinates**: Hovering or looking at edges will show their Source (`S`) and Target (`T`) coordinates (e.g., `S:(150, 240)`).
*   **Visual Guides**: Bounding boxes around nodes and connection points.

## Verification Procedure

To verify layout fixes, do not rely solely on visual inspection. Use the **Coordinate Match Test**:

1.  **Reproduce**: Run the test script (e.g., `scripts/test_separate_expanded.py`).
2.  **Action**: Perform the UI action (e.g., collapse the node).
3.  **Inspect**:
    *   Look at the **Edge Source Y** coordinate (from the edge label).
    *   Look at the **Node Bottom Y** coordinate (from the Node Bounds table).
4.  **Verify**: These two numbers must match **exactly** (or within 1px).
    *   ✅ `Edge Y: 240` == `Node Bottom: 240` (Pass)
    *   ❌ `Edge Y: 240` != `Node Bottom: 300` (Fail - Arrow is hanging)

## Regression Testing
Always ensure a reproduction script exists for visual bugs.
*   **Current Test**: `scripts/test_separate_expanded.py`
*   **Usage**: `uv run python scripts/test_separate_expanded.py`

## Case Study: Theme Initialization Glitch

### Symptom
The visualization briefly flashes the dark theme (dark nodes) before switching to the correct light theme, or starts with a mixed theme (light background, dark nodes) until toggled.

### Root Cause
Theme detection was running **asynchronously** in a `useEffect` hook. The initial state was initialized with a fallback (defaulting to dark), causing the first render to use the wrong theme.

### The Fix: Synchronous Detection
We moved the `detectHostTheme` logic to run **synchronously** during the initial state setup.

**Implementation:**
```javascript
// Define detection logic outside component or before useState
const detectHostTheme = () => { ... };

const App = () => {
    // Initialize state by calling the function directly
    const [detectedTheme, setDetectedTheme] = useState(() => detectHostTheme());
    // ...
};
```
This ensures `activeTheme` is correct on the very first render pass.

