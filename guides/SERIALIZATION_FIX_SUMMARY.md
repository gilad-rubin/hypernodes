# Daft Serialization Fix - Summary

## Problem

When using DaftEngine with distributed execution (e.g., Modal) or with `use_process=True` (multiprocessing), custom classes defined in script modules would fail with:

```
ModuleNotFoundError: No module named 'test_script'
```

The issue occurred because:
1. Script modules aren't available in UDF worker processes
2. Daft's internal Expression objects stored references to original classes
3. These references were pickled and sent to workers, which couldn't import the modules

## Root Cause

The problem had multiple layers:
1. **Type annotations**: Function annotations contained references to script-module classes
2. **Stateful values**: Objects passed as parameters contained script-module classes
3. **Nested objects**: Stateful objects containing other stateful objects weren't fully fixed
4. **Daft's Expression objects**: Created before fixes were applied, storing original class references

## Solution

Implemented a **comprehensive, multi-layered approach**:

### 1. Proactive Module Registration (`_register_all_script_modules()`)

**Location**: Added at the beginning of `DaftEngine.run()` (line 839)

**Purpose**: Register all script modules with cloudpickle BEFORE Daft creates any Expression objects.

**How it works**:
- Recursively walks through all pipeline inputs
- Collects module names from all custom class instances
- Checks if each module is importable
- Registers non-importable modules with `cloudpickle.register_pickle_by_value()`

**Impact**: Forces cloudpickle to serialize entire modules by value, not by reference.

### 2. Type Annotation Fixing

**Location**: Modified type hint extraction (lines 1497-1508)

**Purpose**: Fix all type annotations before they're used to create Daft DataTypes.

**Changes**:
```python
# OLD: Just extracted type hints
type_hints = get_type_hints(func)
return_type = type_hints.get("return", None)

# NEW: Fix ALL type hints
type_hints = get_type_hints(func)
fixed_type_hints = {}
for hint_name, hint_type in type_hints.items():
    fixed_type_hints[hint_name] = _fix_annotation(hint_type)
type_hints = fixed_type_hints
return_type = type_hints.get("return", None)
```

**Impact**: Daft's DataTypes now contain fixed classes with `__module__ = "__main__"`.

### 3. Deep Recursive Object Fixing

**New helper functions** (lines 390-638):

#### `_is_builtin_type(obj)`
Checks if an object is a builtin type that doesn't need fixing.

#### `_ensure_module_pickled_by_value(module_name)`
Registers a specific module with cloudpickle for by-value serialization.

#### `_fix_object_deeply(obj, visited, max_depth, current_depth)`
**The comprehensive solution** - recursively walks through an object's entire attribute tree:
- Fixes nested stateful objects (e.g., `reranker._encoder`)
- Handles lists, tuples, dicts, and sets
- Prevents infinite loops with visited set and max depth
- Fixes class references in attributes

#### `_register_fixed_class_in_sys_modules(original_cls, fixed_cls)`
Registers fixed classes in `sys.modules['__main__']` so workers can find them.

#### `_prepare_stateful_value_for_daft(value, debug)`
**Main orchestration function** that applies all fixes:
1. Deep recursive fixing of nested objects
2. Module registration with cloudpickle
3. sys.modules registration for class lookup

### 4. Updated Serialization Entry Points

**Updated `_ensure_stateful_value_pickleable()`** (line 2406):
```python
# OLD: Shallow fixing of instance __class__
def _ensure_stateful_value_pickleable(self, value):
    # ... shallow __class__ modification ...
    return _fix_instance_class(value)

# NEW: Comprehensive deep fixing
def _ensure_stateful_value_pickleable(self, value):
    return _prepare_stateful_value_for_daft(
        value, debug=getattr(self, "debug", False)
    )
```

**Updated `_build_stateful_udf()`** (line 2343-2348):
```python
# Comprehensively prepare all stateful values for serialization
prepared_stateful_values = {
    key: self._ensure_stateful_value_pickleable(value)
    for key, value in stateful_values.items()
}
```

## Key Files Modified

1. `/src/hypernodes/integrations/daft/engine.py`:
   - Added `_register_all_script_modules()` method (lines 2322-2404)
   - Added comprehensive helper functions (lines 390-638)
   - Updated `run()` to call module registration (line 839)
   - Updated type hint extraction to fix annotations (lines 1499-1505)
   - Simplified `_ensure_stateful_value_pickleable()` (lines 2406-2417)
   - Updated `_build_stateful_udf()` to prepare values (lines 2343-2348)

## Testing

### Created Test Scripts

1. **test_daft_serialization_bug.py**: Local test with simulated unimportable module ✅
2. **test_progressive_complexity.py**: 8 levels of progressive complexity ✅
3. **test_exact_repro.py**: Exact reproduction of user's failing script ✅
4. **test_final_verification.py**: Final verification test ✅

### Test Results

- ✅ All existing DaftEngine tests pass (12/12)
- ✅ All complex types tests pass (7/7)
- ✅ Exact reproduction now works from parent directory
- ✅ Works from script directory (already working)
- ✅ Works with nested stateful objects
- ✅ Works with complex type annotations

### Before/After Comparison

**Before**:
```bash
❯ uv run python scripts/test_exact_repro.py
# From hypernodes/ directory: ✗ FAILED - ModuleNotFoundError
# From scripts/ directory: ✓ SUCCESS
```

**After**:
```bash
❯ uv run python scripts/test_exact_repro.py
# From hypernodes/ directory: ✓ SUCCESS
# From scripts/ directory: ✓ SUCCESS
```

## Benefits

1. **Working directory independent**: Works regardless of where script is run from
2. **Handles nested objects**: Fixes objects containing other objects
3. **Comprehensive**: Fixes annotations, stateful values, and nested attributes
4. **Proactive**: Registers modules before Daft creates expressions
5. **Robust**: Multiple layers of defense against serialization issues
6. **Non-breaking**: All existing tests pass

## Architecture Philosophy

The fix follows a **defense-in-depth** strategy:

1. **Layer 1**: Register modules early (before Daft operations)
2. **Layer 2**: Fix type annotations (before DataType creation)
3. **Layer 3**: Deep-fix stateful values (before UDF creation)
4. **Layer 4**: Register classes in sys.modules (for worker lookup)

This multi-layered approach ensures serialization works even if one layer isn't sufficient for a particular edge case.
