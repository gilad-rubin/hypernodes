# How to Use the Fix in Your Jupyter Notebook

## The Fix

The `AttributeError: module 'cloudpickle' has no attribute 'Unpickler'` has been fixed in `src/hypernodes/backend.py`.

## To Use in Jupyter

Since you've already imported the module, you need to reload it. Add this cell and run it:

```python
# Reload the hypernodes module to get the fix
import importlib
import hypernodes.backend
importlib.reload(hypernodes.backend)

# Re-import to get the updated classes
from hypernodes import ModalBackend, Pipeline, node
from hypernodes.telemetry import ProgressCallback

print("✓ Module reloaded with fix")
```

Alternatively, you can restart the kernel and re-run all cells from the beginning.

## What Was Fixed

- Changed `CPUUnpickler` to inherit from `pickle.Unpickler` instead of `cloudpickle.Unpickler`
- Added `import pickle` to the backend module
- The fix allows PyTorch tensors from GPU execution on Modal to be deserialized on local CPU-only machines

## Verification

After reloading, your pipeline should work without the AttributeError. The deserialization will now correctly handle CUDA tensors on CPU-only machines.
