# Modal + Daft Serialization Issue - Investigation Summary

## Problem Statement

When running `uv run modal run scripts/test_modal_failure_repro.py` from the repository root, the execution fails with:

```
ModuleNotFoundError: No module named 'test_modal_failure_repro'
```

However, running `cd scripts && uv run modal run test_modal_failure_repro.py` works fine.

## Root Cause Analysis

### The Fundamental Issue

Modal and Daft have conflicting serialization behaviors:

1. **Modal's Behavior**: When running a script from the repo root, Modal imports it as a module (e.g., `test_modal_failure_repro`), not as `__main__`. All classes defined in the script get `__module__ = "test_modal_failure_repro"`.

2. **Daft's Behavior**: When creating DataFrame operations, Daft captures type information from function annotations and creates Expression objects. These Expressions are serialized (via cloudpickle) and sent to worker processes.

3. **The Conflict**: The serialized Expressions contain references to classes like `Passage`, `Prediction`, etc., with `__module__ = "test_modal_failure_repro"`. When Daft workers try to deserialize these Expressions, cloudpickle attempts to import `test_modal_failure_repro`, which doesn't exist in the worker environment.

### Why It Works from `scripts/`

When run from `scripts/`, the module name becomes just `test_modal_failure_repro` (without the `scripts.` prefix), and Modal's module resolution happens to work differently in that context.

### Timeline of the Failure

1. **Local Machine**: Script is imported by Modal as `test_modal_failure_repro`
2. **Local Machine**: Classes are defined with `__module__ = "test_modal_failure_repro"`
3. **Local Machine**: Pipeline is created with nodes referencing these classes in annotations
4. **Local Machine**: Modal serializes the entire pipeline (with embedded class references) to send to worker
5. **Worker Machine**: Modal deserializes the pipeline
6. **Worker Machine**: DaftEngine creates Expression objects from node annotations
7. **Worker Machine**: Daft sends Expressions to UDF worker subprocess
8. **UDF Worker**: Attempts to deserialize Expression → tries to `import test_modal_failure_repro` → **FAILS**

## Attempted Solutions

### 1. Runtime Class Fixing in `DaftEngine.run()`
**Status**: ❌ Too Late

We added `_fix_all_script_classes()` to fix classes at the start of `run()`. This successfully changed `__module__` to `"__main__"` for all detected classes, but the fix happened AFTER the pipeline was already serialized by Modal. The serialized pipeline already contained references to the unfixed classes.

### 2. Initialization-Time Fixing in `DaftEngine.__init__()`
**Status**: ❌ Not Applicable

Added `_fix_script_classes_at_init()` to fix classes when the engine is instantiated. However, this runs on the worker after the pipeline is deserialized, so it's still too late.

### 3. Module-Level Helper Function `fix_script_classes_for_modal()`
**Status**: ⚠️ Partial Solution

Created a helper function users can call at the top of their scripts. However:
- It's difficult to reliably identify which classes need fixing (too broad = breaks stdlib, too narrow = misses user classes)
- It still happens after classes are defined, so any code that references the classes before the fix will capture the old module names
- Requires users to remember to call it in every script

### 4. Output Fixing with `_fix_output_tree()`
**Status**: ✅ Already Implemented

The output fixing (line 2617 in DaftEngine) already applies `_fix_output_tree()` to returned values, ensuring output objects are serializable. This wasn't the issue.

## Current State

- ✅ **Local Execution**: Works perfectly (classes are `__main__`)
- ✅ **Output Serialization**: Properly handled via `_fix_output_tree()`
- ✅ **Stateful Input Fixing**: Working correctly
- ❌ **Modal Execution from Repo Root**: Still fails due to Expression serialization
- ✅ **Modal Execution from scripts/**: Works (different module resolution)

## Recommended Solutions

### Short-Term Workaround

**Run scripts from their containing directory:**

```bash
cd scripts
uv run modal run test_modal_failure_repro.py
```

This sidesteps the module naming issue.

### Medium-Term Solution

**Mount `scripts/` into Modal and add to PYTHONPATH:**

```python
modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTHONPATH": "/root/scripts"})
    .add_local_dir("scripts/", remote_path="/root/scripts")
    .add_local_dir("src/hypernodes", remote_path="/root/hypernodes")
)
```

This makes script modules importable on workers.

### Long-Term Solutions

1. **Restructure as Package**: Move script code into proper Python packages that can be installed/imported normally.

2. **Use Different Execution Model**: Instead of sending the entire pipeline to Modal, send only data and configuration, with the pipeline defined on the worker side.

3. **Upstream Fixes**: 
   - Request Daft to handle non-importable modules more gracefully
   - Request Modal to provide better control over script import behavior

## Technical Deep Dive

### What We Learned

1. **Class References are Captured Early**: Even though we fix `__module__` attributes, references to classes in function annotations are captured when functions are defined, not when they're called.

2. **Serialization Happens at Multiple Levels**:
   - Modal serializes the pipeline object
   - Daft serializes Expression objects
   - UDF workers deserialize both
   - Each level can fail independently

3. **cloudpickle is Tricky**: Even with `register_pickle_by_value()`, cloudpickle still tries to import modules referenced in `__module__` attributes during deserialization.

4. **The `__module__` Attribute is Immutable for Some Types**: Built-in types and some C-extension types can't have their `__module__` changed, which is why we see "cannot set '__module__' attribute" warnings.

### Files Modified

- `src/hypernodes/integrations/daft/engine.py`: Added extensive debugging, class fixing logic
- `src/hypernodes/engines.py`: Exported `fix_script_classes_for_modal()`
- `scripts/test_modal_failure_repro.py`: Added call to `fix_script_classes_for_modal()`
- `tests/test_daft_modal_cwd_issue.py`: Created local reproduction test (✅ Passes locally)
- `scripts/diagnostic_module_names.py`: Created diagnostic script

### Debug Output Analysis

When classes ARE fixed (from `_fix_all_script_classes` in `run()`):
```
[_fix_all_script_classes] Fixed test_modal_failure_repro.Passage -> __main__.Passage
[_fix_all_script_classes] Fixed test_modal_failure_repro.Prediction -> __main__.Prediction
...
```

But the error still occurs in the UDF worker during Expression deserialization, proving the fix happens too late in the execution flow.

## Conclusion

This is a fundamental architectural incompatibility between Modal's script execution model and Daft's Expression serialization. The issue cannot be fully resolved within DaftEngine alone. The recommended approach is to:

1. Use the short-term workaround (run from `scripts/`)
2. Plan to restructure code as proper packages for production use
3. Consider alternative execution strategies that don't involve serializing entire pipelines

The investigation has been valuable in understanding these serialization boundaries and has resulted in better debug tooling and documentation for future similar issues.


