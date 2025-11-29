# Hanging Arrow Bug Fix

## The Problem

When collapsing a pipeline node in the React Flow visualization, edges would "hang" approximately **41 pixels below** the collapsed node's bottom edge. This created a visual bug where arrows appeared disconnected from their source nodes.

### Symptoms
- Edge source Y coordinate: 281
- Node bottom Y coordinate: 240
- Mismatch: **41 pixels**

### Key Observation
Manually toggling the theme (dark/light) would immediately fix the issue. This was the critical clue that led to the solution.

---

## Root Cause Analysis

React Flow caches edge paths based on node handle positions. When a pipeline node collapses:

1. The node's DOM element shrinks (e.g., from 828px to 68px height)
2. The Handle component moves to a new position (near the new bottom)
3. **But React Flow doesn't automatically recalculate** the edge bezier paths
4. Edges continue using the old cached path coordinates

The theme toggle works because it triggers a full re-render of all node and edge components, forcing React Flow to recalculate paths using current handle positions.

---

## Approaches Tried

### 1. Edge ID Versioning ❌
**Idea:** Add a version suffix to edge IDs to force React Flow to treat them as "new" edges.

```javascript
id: `${e.id}_g${edgeGeneration}`
```

**Result:** Failed. React Flow still used cached path calculations even with new IDs.

---

### 2. `updateNodeInternals()` ❌
**Idea:** Call React Flow's `updateNodeInternals()` for collapsed nodes to force handle recalculation.

```javascript
const pipelineIds = nodes.filter(n => n.data?.nodeType === 'PIPELINE').map(n => n.id);
pipelineIds.forEach(id => updateNodeInternals(id));
```

**Result:** Failed. This API is designed for when handles are added/removed, not when handles move due to node resize.

---

### 3. Node Data Refresh Key ❌
**Idea:** Add a `_refreshKey` to node data that changes on expansion, hoping React Flow would detect node changes.

```javascript
data: { ...n.data, _refreshKey: nodeRefreshKey }
```

**Result:** Failed. React Flow doesn't watch for arbitrary data changes to trigger edge recalculation.

---

### 4. Hide and Re-show Edges ❌
**Idea:** Hide edges during expansion state change, wait for DOM to update, then show edges with new IDs.

```javascript
setEdgesVisible(false);
setTimeout(() => {
  setEdgeGeneration(g => g + 1);
  setEdgesVisible(true);
}, 150);
```

**Result:** Failed. The timing was unreliable - edges were still calculated before nodes finished resizing.

---

### 5. Wait for Layout Completion ❌
**Idea:** Track `layoutVersion` and only regenerate edges after ELK layout completes.

```javascript
if (pendingEdgeRegen && !isLayouting && layoutVersion !== prevLayoutVersionRef.current) {
  // Regenerate edges
}
```

**Result:** Failed. ELK layout doesn't re-run on collapse (only visibility changes, not structure changes), so `layoutVersion` never updated.

---

### 6. Rapid Theme Toggle ✅
**Idea:** Mimic what the manual theme toggle does - briefly toggle theme to force full recalculation.

```javascript
useEffect(() => {
  if (expansionKey !== prevExpansionKeyRef.current) {
    prevExpansionKeyRef.current = expansionKey;
    
    const timer = setTimeout(() => {
      const currentTheme = theme;
      const otherTheme = currentTheme === 'dark' ? 'light' : 'dark';
      
      // Toggle to other theme, then immediately back
      setManualTheme(otherTheme);
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          setManualTheme(currentTheme);
        });
      });
    }, 100);
    
    return () => clearTimeout(timer);
  }
}, [expansionKey, theme, setManualTheme]);
```

**Result:** SUCCESS! The theme toggle is imperceptible but forces React Flow to fully recalculate all edge paths.

---

## Final Solution

The fix works by:

1. **Detecting expansion state changes** via `expansionKey` (a string of collapsed pipeline IDs)
2. **Waiting 100ms** for DOM to settle after collapse/expand
3. **Toggling theme briefly** (dark→light→dark or vice versa)
4. **Using double `requestAnimationFrame`** to ensure the toggle-back happens after React commits

### Why This Works

Theme changes cause:
- All node components to re-render with new style props
- All edge components to re-render with new style props
- React Flow's internal stores to update with fresh node dimensions
- Edge bezier paths to be recalculated from current handle positions

### Results

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Edge source Y | 281 | 233 |
| Node bottom Y | 240 | 240 |
| Mismatch | 41px (bug!) | 7px (handle offset, correct) |

The 7px remaining offset is the **correct handle position** - handles are positioned slightly inside the node border, not exactly at the edge.

---

## Debugging Tools Added

As part of this investigation, we added debug overlays to the visualization:

### Enable Debug Mode
- URL parameter: `?debug=true`
- UI toggle: "Debug overlays" button in view controls
- Console: `HyperNodesVizState.debug.showOverlays()`

### Debug Features
- **Edge coordinate labels**: `S:(x,y) T:(x,y)` showing source and target coordinates
- **Node bounds table**: Shows Y position, height, and bottom for each node
- Helps verify that edge source Y matches node bottom (within handle offset tolerance)

---

## Lessons Learned

1. **React Flow edge caching is aggressive** - it doesn't automatically recalculate paths on node resize
2. **`updateNodeInternals()` is limited** - only works for handle add/remove, not position changes
3. **The simplest fix is often the best** - mimicking what manual interaction does (theme toggle) was more reliable than trying to work around React Flow's internals
4. **Debug tools are invaluable** - the coordinate overlays made it easy to verify the fix worked

---

## Files Changed

| File | Change |
|------|--------|
| `src/hypernodes/viz/js/html_generator.py` | Added expansion state effect with theme toggle fix |

