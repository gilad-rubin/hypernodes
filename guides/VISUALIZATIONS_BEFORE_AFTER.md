# Visualization Fixes - Before & After

## Summary of All Fixes Applied

### ✅ Fixed Issues:

1. **No outer outline for top-level pipeline** - Only nested pipelines now have clusters
2. **No text labels on edges** - Cleaner arrows without parameter names
3. **Better spacing** - Increased `ranksep` to 0.8 and `nodesep` to 0.5 for better arrow length
4. **Fixed PipelineNode crash** - Handled `.as_node()` wrapped pipelines correctly
5. **Clean type names** - Removed module prefixes like `__main__.Passage` → `Passage`
6. **Disabled grouped inputs** - Changed default `min_arg_group_size` from 2 to None
7. **Handle tuple outputs** - Fixed crash when nodes have multiple outputs

## How to Verify in Your Notebook

Run these cells in `v2.ipynb` to see the fixed visualizations:

### Test 1: Simple Pipeline with Nested Components
```python
# This should now work without the AttributeError
single_encode.visualize()
```

**Expected improvements:**
- ✅ No outer "pipeline" cluster box
- ✅ No text on arrows
- ✅ Better spacing between nodes
- ✅ Type names show as just "Passage" not "__main__.Passage"
- ✅ No confusing "group_..." nodes

### Test 2: Pipeline with .as_node()
```python
# This previously crashed with AttributeError
encode_and_index.visualize()
```

**Expected improvements:**
- ✅ Works without crashing
- ✅ Shows "nested_pipeline" for wrapped pipelines
- ✅ Clean output without grouped inputs

### Test 3: Complex Nested Pipeline
```python
# Full search pipeline
full_pipeline.visualize()
```

**Expected improvements:**
- ✅ Clear hierarchy with only nested pipelines having boxes
- ✅ Much better spacing in both TB and LR orientations
- ✅ No edge labels cluttering the view

### Test 4: Try LR Orientation (Better Arrow Length)
```python
# Test with left-to-right layout
single_encode.visualize(orient="LR")
```

**Expected improvements:**
- ✅ Much better spacing between nodes
- ✅ Arrows are no longer cramped

## What Changed in Code

The fixes are in `/Users/giladrubin/python_workspace/hypernodes/src/hypernodes/visualization.py`:

1. **Line 204-205, 238-239**: Added regex to remove module prefixes from type names
2. **Line 384-391**: Added check for `__name__` attribute to handle PipelineNode
3. **Line 394-396**: Handle tuple output names
4. **Line 518**: Changed default `min_arg_group_size` from 2 to None
5. **Line 562-565**: Increased spacing (ranksep: 0.8, nodesep: 0.5)
6. **Line 647-649**: Removed outer cluster for root pipeline
7. **Line 692-702**: Removed edge labels

## Testing

All test cases in `scripts/test_viz_fixes.py` pass successfully:

```bash
uv run python scripts/test_viz_fixes.py
```

Output files are saved to `outputs/` directory:
- `test1_single_encode.svg` - Simple pipeline
- `test3_encode_and_index.svg` - Pipeline with .as_node()
- `test4_full_pipeline.svg` - Complex nested pipeline

## Next Steps

1. Open your notebook `v2.ipynb`
2. Re-run the visualization cells
3. You should see all the improvements immediately

The visualizations will now be clean, properly spaced, and work correctly with `.as_node()` wrapped pipelines!
