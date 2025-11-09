# PyTorch + Daft Compatibility Guide

## Problem

When using PyTorch/HuggingFace models with Daft DataFrames, you may encounter:
- **SIGSEGV (Segmentation Fault)** crashes (exit code 139)
- **Leaked semaphore** warnings
- Process hangs or deadlocks

## Root Causes

### 1. PyTorch "Poison Fork" Issue
PyTorch and CUDA have a known issue with the `fork` multiprocessing method. When a process forks after CUDA initialization, child processes inherit corrupted CUDA state, causing segmentation faults.

**Reference**: https://pytorch.org/docs/stable/notes/multiprocessing.html

### 2. HuggingFace Tokenizer Parallelism
Tokenizers use internal parallelism that conflicts with process forking:
```
The current process just got forked, after parallelism has already been used.
Disabling parallelism to avoid deadlocks...
```

### 3. Daft's Multiprocessing
Daft's native runner uses multiprocessing internally, which can trigger these issues even when you don't explicitly use multiprocessing.

## Solutions

### Option 1: Use HypernodesEngine (Recommended for PyTorch)

For PyTorch/CUDA workloads, use `HypernodesEngine` instead of `DaftEngine`:

```python
from hypernodes import Pipeline, node, HypernodesEngine, DiskCache

pipeline = Pipeline(
    nodes=[...],
    engine=HypernodesEngine(
        map_executor="sequential",  # Avoid multiprocessing
        node_executor="sequential"
    ),
    cache=DiskCache(path=".cache")
)
```

**Pros**:
- No multiprocessing issues with PyTorch
- Simple and reliable
- Works with all PyTorch/CUDA models

**Cons**:
- Sequential execution (slower for independent operations)
- No automatic parallelization

### Option 2: Configure Multiprocessing Start Method

Set `spawn` method before importing PyTorch:

```python
import os
import multiprocessing

# MUST be at the very top of your script
multiprocessing.set_start_method('spawn', force=True)

# Configure environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Now import PyTorch/transformers
import torch
from transformers import AutoModel
```

**Pros**:
- May allow multiprocessing with PyTorch
- Recommended by PyTorch documentation

**Cons**:
- Must be set before ANY torch imports
- Slower than `fork` (entire process state is copied)
- May still have issues with Daft's internal multiprocessing

### Option 3: Use Daft's Native `@daft.cls` Pattern

For advanced users, use Daft's native stateful class pattern:

```python
import daft

@daft.cls(max_concurrency=1, use_process=False)
class ModelEncoder:
    def __init__(self, model_path: str):
        # Model loaded once per worker
        self.model = load_model(model_path)

    @daft.method(return_dtype=daft.DataType.python())
    def __call__(self, text):
        # Handle dict input (Daft serializes Pydantic to dict)
        if isinstance(text, dict):
            text = text["text"]
        return self.model.encode(text)

encoder = ModelEncoder("bert-base-uncased")
df = daft.from_pydict({"text": ["hello", "world"]})
df = df.select(encoder(df["text"]).alias("embedding"))
```

**Key Points**:
- Use `max_concurrency=1` to prevent multiple model instances
- Use `use_process=False` to use threading instead of multiprocessing
- Handle dict inputs (Daft serializes Pydantic models to dicts)

**Pros**:
- Leverages Daft's optimization
- Model loaded once and reused
- Proper isolation with `@daft.cls`

**Cons**:
- More complex implementation
- Requires understanding Daft's serialization
- May still encounter multiprocessing issues

## DaftEngine Improvements

The `DaftEngine` class now includes automatic PyTorch compatibility:

```python
from hypernodes.engines import DaftEngine

# Automatic spawn method configuration
engine = DaftEngine(force_spawn_method=True)  # default

# Disable if you've already configured multiprocessing
engine = DaftEngine(force_spawn_method=False)
```

The engine will:
1. Check current multiprocessing start method
2. Set to `spawn` if not already set
3. Warn if incompatible method already set
4. Provide helpful error messages

## Recommendations

### For Development/Testing
- Use `HypernodesEngine` with sequential executors
- Simple, reliable, no multiprocessing issues

### For Production with Small Models
- Use `HypernodesEngine` with `node_executor="threaded"`
- Some parallelism without multiprocessing issues

### For Production with Large-Scale Processing
- Use Daft's `@daft.cls` pattern
- Requires careful implementation
- Test thoroughly for stability

### Avoid
- Using `DaftEngine` with PyTorch/CUDA models (unstable)
- Mixing `fork` multiprocessing with PyTorch
- Loading models before multiprocessing setup

## Environment Variables

Always set these when using PyTorch with multiprocessing:

```python
import os

# Disable tokenizer parallelism (prevents fork warnings)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable OpenMP threading (prevents conflicts)
os.environ["OMP_NUM_THREADS"] = "1"

# Disable MKL threading (prevents conflicts)
os.environ["MKL_NUM_THREADS"] = "1"
```

## Testing

Test your setup with:

```bash
# Should complete without SIGSEGV
uv run python your_script.py

# Check exit code (should be 0, not 139)
echo $?
```

## References

- [PyTorch Multiprocessing Best Practices](https://pytorch.org/docs/stable/notes/multiprocessing.html)
- [HuggingFace Tokenizers Issue #258](https://github.com/huggingface/tokenizers/issues/258)
- [Daft Documentation](https://docs.getdaft.io/)

## Examples

See:
- `scripts/daft_crash_test.py` - Original issue demonstration
- `scripts/daft_native_test.py` - Daft native `@daft.cls` implementation
- `tests/test_daft_backend.py` - HyperNodes test suite
