# Nested Progress Bars - Fixes Applied

## Issues Fixed

### 1. **Nested Progress Bars Positioning**
**Problem:** Progress bars for nested pipelines were displayed at the same position level, causing overlap and visual confusion.

**Solution:** Added `position` parameter to tqdm based on nesting depth:
- Root pipeline: `position=0`
- Nested elements (nodes, nested pipelines): `position=depth+1`

### 2. **Parent Pipeline Progress Stuck at 67%**
**Problem:** When a nested pipeline completed, the parent pipeline's progress bar wasn't updated, causing it to appear stuck (showing 2/3 = 67% instead of 100%).

**Solution:** Implemented `on_nested_pipeline_end()` callback method that:
- Updates the nested pipeline's placeholder bar
- **Crucially updates the parent pipeline's progress bar** (`pipeline_bar.update(1)`)
- Cleans up the nested bar properly

### 3. **Telemetry Logs Interfering with Progress Bars**
**Problem:** Logfire console output (timestamps like "12:00:07.754 pipeline:pipeline_4479057216") was printed in between progress bars, disrupting the visual display.

**Solution:** Configure logfire with `console=False` to suppress console output:
```python
logfire.configure(send_to_logfire=True, console=False)
```

### 4. **Progress Bars Not Cleaning Up**
**Problem:** Nested progress bars remained visible after completion, cluttering the display.

**Solution:** Set `leave=False` for all nested elements:
- Nested pipelines: `leave=False`
- Individual nodes: `leave=False`
- Map operations: `leave=False`
- Only root pipeline: `leave=True` (or `leave=(depth==0)`)

## Key Changes to `ProgressCallback`

### Modified `_create_bar()` method:
```python
def _create_bar(self, desc: str, total: Optional[int] = None, 
                position: int = 0, leave: bool = True, **kwargs):
    return self.tqdm(desc=desc, total=total, position=position, leave=leave, **kwargs)
```

### Added `on_nested_pipeline_start()`:
```python
def on_nested_pipeline_start(self, parent_id: str, child_pipeline_id: str, ctx: CallbackContext) -> None:
    indent = "  " * (ctx.depth + 1)
    bar = self._create_bar(
        desc=f"{indent}├─ {child_pipeline_id}",
        total=1,
        position=ctx.depth + 1,
        leave=False,
    )
    ctx.set(f"progress_bar:nested:{child_pipeline_id}", bar)
```

### Added `on_nested_pipeline_end()`:
```python
def on_nested_pipeline_end(self, parent_id: str, child_pipeline_id: str, duration: float, ctx: CallbackContext) -> None:
    # Update nested pipeline bar
    bar = ctx.get(f"progress_bar:nested:{child_pipeline_id}")
    if bar:
        indent = "  " * (ctx.depth + 1)
        bar.set_description(f"{indent}├─ {child_pipeline_id} ✓ ({duration:.2f}s)")
        bar.update(1)
        bar.close()

    # **CRITICAL FIX:** Update parent pipeline bar
    pipeline_bar = ctx.get(f"progress_bar:{parent_id}")
    if pipeline_bar:
        pipeline_bar.update(1)  # This was missing!
```

## Testing

Run the test script:
```bash
uv run python scripts/test_nested_progress.py
```

Or test in Jupyter notebook:
```python
# notebooks/telemetry_examples.ipynb - Cell 16
```

## Expected Behavior

✓ Root pipeline bar shows at position 0  
✓ Nested pipeline and nodes show at position 1 (indented)  
✓ Parent pipeline updates correctly when nested pipeline completes  
✓ No telemetry logs interrupt the progress bars  
✓ Nested bars clean up when done (only root bar remains)  
✓ All bars complete at 100%  

## References

- tqdm nested progress bars documentation: Use `position` parameter for manual control
- tqdm `leave` parameter: `leave=False` removes bar after completion
- Logfire configuration: `console=False` suppresses console output
