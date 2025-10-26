# Modal Backend Map Operation Pickling Fix

## Issue

When using `Pipeline.map()` with the `ModalBackend`, the operation would fail with a pickling error:

```
TypeError: cannot pickle 'EncodedFile' instances
```

This occurred because the `CallbackContext` stored unpicklable objects (progress bars, tracing spans, etc.) that needed to be serialized and sent to the remote Modal execution environment.

## Root Cause

The `ModalBackend._serialize_payload()` method filtered `CallbackContext.data` to only include simple types (str, int, float, bool, list, dict, None), but it didn't exclude keys that contained unpicklable objects like:

- **Progress bars** (`progress_bar:*`) - tqdm/rich objects with file handles
- **Tracing spans** (`span:*`, `current_span`) - OpenTelemetry span objects
- **Map operation state** (`map_span`, `map_item_span:*`, `map_node_bars`) - Complex nested objects

When `Pipeline.map()` was called, these objects were present in the context from the local execution, and the serialization would fail when trying to pickle them for remote execution.

## Solution

Updated `ModalBackend._serialize_payload()` to explicitly exclude unpicklable callback-internal state by filtering out keys with specific prefixes:

```python
excluded_prefixes = (
    "progress_bar:",
    "span:",
    "map_span",
    "map_item_span:",
    "current_span",
    "map_node_bars",
)
ctx_data = {
    k: v
    for k, v in ctx.data.items()
    if not any(k.startswith(prefix) for prefix in excluded_prefixes)
    and isinstance(v, (str, int, float, bool, list, dict, type(None)))
}
```

This ensures that only serializable context data is sent to the remote Modal environment, while callback-internal state remains local.

## Files Modified

- `src/hypernodes/backend.py` - Updated `_serialize_payload()` method (lines 1768-1781)

## Tests Added

- `tests/test_modal_map.py` - New test file with `test_modal_map_operation()` that verifies:
  - Single pipeline run works
  - Map operation with multiple items works
  - Results are correct

## Verification

All existing tests pass (57 tests), and the new test passes. The fix has been verified with:

1. Unit test: `pytest tests/test_modal_map.py -v`
2. Integration script: `python scripts/test_modal_map_fix.py`

## Impact

This fix enables `Pipeline.map()` to work correctly with `ModalBackend` when callbacks (like `ProgressCallback`) are attached to the pipeline. Users can now:

- Use progress bars with Modal backend map operations
- Use tracing/telemetry with Modal backend
- Combine all callback features with remote Modal execution
