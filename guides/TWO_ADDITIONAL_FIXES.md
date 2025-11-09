# Two Additional Fixes Applied

## Fix 1: Make Node Callable with Positional Arguments ✅

### Problem
When using nodes in Daft wrappers, calling them with positional arguments like:
```python
return encode_query(query, self.encoder)
```
Would fail because Node's `__call__` only accepted keyword arguments.

### Solution
**File**: `src/hypernodes/node.py` (lines 62-75)

Changed:
```python
def __call__(self, **kwargs) -> Any:
    return self.func(**kwargs)
```

To:
```python
def __call__(self, *args, **kwargs) -> Any:
    return self.func(*args, **kwargs)
```

### Tests
✅ `tests/test_node_callable_fix.py` - 2/2 tests pass

---

## Fix 2: Remove Ellipsis (...) from Generated Code ✅

### Problem
Generated code contained ellipsis for long lists:
```python
"recall_k_list": [[20, 50, 100, 200, 300, 400, ...]],  # ❌ Invalid!
```

Caused runtime error:
```
TypeError: '<=' not supported between instances of 'ellipsis' and 'int'
```

### Solution
**File**: `src/hypernodes/integrations/daft/engine.py` (lines 2143-2145)

Changed from `reprlib.repr()` (which truncates) to `repr()` (full representation):
```python
elif isinstance(value, (list, tuple)):
    return repr(value)  # ✅ Full list, no truncation
```

### Tests
✅ `tests/test_daft_ellipsis_fix.py` - 2/2 tests pass

---

## All Three Fixes Complete! 🎉

1. ✅ **Column Preservation** - Indexes preserved across map operations
2. ✅ **Node Callable** - Nodes work with positional arguments  
3. ✅ **No Ellipsis** - Generated code has complete lists

Your Hebrew retrieval pipeline is ready! 🚀
