# Progress Bar System - Complete Improvements

## Summary

Fixed all progress bar issues for nested pipelines and map operations, providing clear hierarchy visualization and per-node progress tracking.

## Issues Fixed

### 1. **Nested Progress Bars Positioning**
**Problem:** Progress bars for nested pipelines were displayed at the same position level, causing overlap and visual confusion.

**Solution:** Added `position` parameter to tqdm based on nesting depth:
- Root pipeline: `position=0`
- Nodes at depth+1: unique positions for each node
- Clear vertical hierarchy

### 2. **Parent Pipeline Progress Stuck**
**Problem:** When a nested pipeline completed, the parent pipeline's progress bar wasn't updated, appearing stuck at 67%.

**Solution:** Implemented `on_nested_pipeline_end()` that updates parent pipeline progress.

### 3. **Telemetry Logs Interfering**
**Problem:** Logfire console output interrupted progress bars.

**Solution:** Configure logfire with `console=False`:
```python
logfire.configure(send_to_logfire=True, console=False)
```

### 4. **Progress Bars Disappearing**
**Problem:** Bars disappeared after completion.

**Solution:** Set `leave=True` for all bars to keep them visible.

### 5. **Confusing Hierarchy Display**
**Problem:** Nested pipelines created duplicate bars (placeholder + actual), making hierarchy unclear.

**Solution:** Removed duplicate placeholder bars:
- `on_nested_pipeline_start()` no longer creates separate bars
- Each pipeline creates its own bar via `on_pipeline_start()`
- Clear top-to-bottom hierarchy

### 6. **Map Operations: Single Generic Progress Bar**
**Problem:** `.map()` showed only one "Map" bar (0 to N items). Needed per-node visibility.

**Solution:** Implemented per-node progress bars:
- One bar per node showing 0 to N items
- Real-time stats: items/s, cache %
- Each node at unique position

## Implementation Details

### Key Changes in `ProgressCallback`

1. **Position Management**
   - Pipeline bars: `position=ctx.depth`
   - Node bars: `position=ctx.depth + 1 + idx` (unique per node)

2. **Map Operation Bars**
   ```python
   on_map_start():
     - Create pipeline bar for overall progress
     - Create one bar per node with unique positions
     - Store in ctx for updates during execution
   
   on_node_end() when in_map:
     - Update corresponding node bar
     - Track cache hits, calculate rates
   
   on_map_end():
     - Update final descriptions with stats
     - Close all bars
   ```

3. **Skip Regular Bars During Map**
   - `on_pipeline_start()` and `on_node_start()` skip when `_in_map=True`
   - Prevents duplicate bars during map execution

### Key Changes in `Pipeline.map()`

**Critical Fix:** Set metadata before `on_map_start()`
```python
# Before calling on_map_start, set metadata so callbacks can access node_ids
ctx.set_pipeline_metadata(self.id, {
    "total_nodes": len(self.execution_order),
    "node_ids": node_ids
})
```

## Example Output

### Nested Pipeline
```
Pipeline: main_pipeline ✓ (0.83s):  100% [3/3]
  ├─ load_data ✓ (0.10s):  100% [1/1]
  Pipeline: preprocessing ✓ (0.52s):  100% [2/2]
    ├─ clean_data ✓ (0.20s):  100% [1/1]
    ├─ normalize ✓ (0.31s):  100% [1/1]
  ├─ aggregate ✓ (0.21s):  100% [1/1]
```

### Map Operation
```
Pipeline: map_pipeline (map) ✓ (1.58s, 6.3 items/s):  100% [10/10]
  ├─ square ✓ (1.58s, 6.3 items/s, 0.0% cached):  100% [10/10]
  ├─ check_even ✓ (1.58s, 6.3 items/s, 0.0% cached):  100% [10/10]
```

## Testing

Run the test script:
```bash
uv run python scripts/test_hierarchy_and_map.py
```

Expected behavior:
- ✓ Clear top-to-bottom hierarchy for nested pipelines
- ✓ No duplicate/confusing bars
- ✓ Per-node progress bars for map operations
- ✓ Each map bar goes from 0 to N (number of items)
- ✓ All bars remain visible after completion
- ✓ Real-time statistics displayed

## Files Modified

1. `src/hypernodes/telemetry/progress.py`
   - Added position management
   - Implemented per-node map bars
   - Removed duplicate nested pipeline bars
   - Skip regular bars during map operations

2. `src/hypernodes/pipeline.py`
   - Set metadata before `on_map_start()` call

3. `notebooks/telemetry_examples.ipynb`
   - Updated logfire configuration
