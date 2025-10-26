# Modal Backend Deserialization Fix

## Issue

When running pipelines on Modal with GPU and deserializing results on a local CPU-only machine, the following error occurred:

```python
AttributeError: module 'cloudpickle' has no attribute 'Unpickler'
```

This happened in `ModalBackend._deserialize_results()` at line 1823:

```python
class CPUUnpickler(self.cloudpickle.Unpickler):  # ❌ cloudpickle has no Unpickler
```

## Root Cause

The `cloudpickle` module does not expose an `Unpickler` class. It only provides high-level functions like `dumps()` and `loads()`. The `Unpickler` class is part of the standard library's `pickle` module.

## Solution

Changed the custom unpickler to inherit from `pickle.Unpickler` instead of `cloudpickle.Unpickler`:

```python
class CPUUnpickler(pickle.Unpickler):  # ✓ Use standard pickle.Unpickler
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            # Override torch's _load_from_bytes to force CPU mapping
            def _load_from_bytes_cpu(b):
                return torch.load(
                    io.BytesIO(b),
                    map_location=torch.device('cpu'),
                    weights_only=False
                )
            return _load_from_bytes_cpu
        return super().find_class(module, name)
```

## Files Modified

- **src/hypernodes/backend.py**:
  - Added `import pickle` (line 7)
  - Changed `CPUUnpickler` base class from `self.cloudpickle.Unpickler` to `pickle.Unpickler` (line 1824)

## How It Works

1. When deserializing results from Modal, if PyTorch is available but CUDA is not, we need to map GPU tensors to CPU
2. The custom `CPUUnpickler` inherits from `pickle.Unpickler` and overrides `find_class()`
3. When unpickling encounters `torch.storage._load_from_bytes`, we intercept it and force `map_location='cpu'`
4. This allows PyTorch tensors created on GPU to be loaded on CPU-only machines

## Testing

Created `scripts/test_modal_deserialize.py` to verify:
- ✓ `pickle.Unpickler` works correctly
- ✓ `ModalBackend._deserialize_results()` works with standard data
- ✓ PyTorch tensor CPU mapping works (when torch is available)

All tests pass successfully.

## Impact

Users can now:
- Run ML/DL pipelines on Modal with GPU
- Deserialize PyTorch tensors on local CPU-only machines
- Use embedding generation, model inference, and retrieval pipelines without errors

## Related Memories

This fix builds on the previous Modal backend work:
- CUDA tensor deserialization (memory cf48886d)
- Callback context serialization (memory ad32d5ab)
- Execution strategies (memory 87ce29cf)
