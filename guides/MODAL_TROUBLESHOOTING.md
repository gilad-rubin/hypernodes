# Modal Connection Issues - Troubleshooting Guide

## Problem

You're seeing these errors when running your Hebrew retrieval pipeline with Modal:

```
ConnectionError: Connection lost
ClientOSError: [Errno 32] Broken pipe
TimeoutError: Deadline exceeded
```

## Root Causes

### 1. **Hypernodes Not in Python Path** (Most Likely)

When you add local directories to Modal:

```python
.add_local_dir(str(hypernodes_dir), remote_path="/root/hypernodes")
```

The code is copied but **not installed as a Python package**. When the remote function tries to import hypernodes, it fails.

### 2. **Long-Running Pipeline**

Your pipeline might take longer than Modal's heartbeat timeout, causing connection drops.

### 3. **Image Build Issues**

Dependencies might not be fully installed or compatible.

## Solutions

### Solution 1: Fix Python Path (RECOMMENDED)

Update your Modal image to include hypernodes in PYTHONPATH:

```python
import os
from pathlib import Path
import modal

hypernodes_dir = Path("/Users/giladrubin/python_workspace/hypernodes")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({
        "HF_HOME": "/root/models",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/root/hypernodes/src:$PYTHONPATH",  # ← ADD THIS
    })
    .uv_pip_install(
        # ... your dependencies ...
    )
    .add_local_dir(
        str(hypernodes_dir / "src"),  # ← Copy src folder
        remote_path="/root/hypernodes/src",
    )
)
```

### Solution 2: Install Hypernodes as Package

Properly install hypernodes in the image:

```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        # ... your dependencies ...
    )
    .add_local_dir(
        str(hypernodes_dir / "src" / "hypernodes"),
        remote_path="/root/hypernodes_pkg/hypernodes",
    )
    .run_commands(
        "cd /root/hypernodes_pkg && "
        "echo 'from setuptools import setup, find_packages; "
        "setup(name=\"hypernodes\", packages=find_packages())' > setup.py && "
        "pip install -e ."
    )
)
```

### Solution 3: Increase Timeout

If your pipeline is legitimately slow, increase the timeout:

```python
backend = ModalBackend(
    image=image,
    gpu="A10G",
    timeout=7200,  # 2 hours instead of 1 hour
    map_execution="sequential",
    volumes={
        "/root/models": models_volume,
        "/root/data": data_volume,
    },
)
```

## Step-by-Step Fix

### Step 1: Test Modal Setup

Run the diagnostic script to identify the issue:

```bash
uv run python scripts/diagnose_modal_issue.py
```

This will tell you:
- ✓ If basic Modal execution works
- ✓ If slow execution causes timeouts
- ✓ If dependencies are missing

### Step 2: Test Hypernodes on Modal

Run the fix test script:

```bash
uv run python scripts/fix_modal_hebrew_example.py
```

This tests if hypernodes imports work on Modal. If it fails, you'll see the exact import error.

### Step 3: Update Your Image

Based on the test results, update your image definition. Here's the recommended setup:

```python
# At the top of your notebook/script
import os
from pathlib import Path
import modal
from hypernodes import Pipeline, DiskCache
from hypernodes.backend import ModalBackend

# Paths
hypernodes_dir = Path("/Users/giladrubin/python_workspace/hypernodes")

# Create volumes
models_volume = modal.Volume.from_name("mafat-models", create_if_missing=True)
data_volume = modal.Volume.from_name("mafat-data", create_if_missing=True)

# Define image with proper PYTHONPATH
image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({
        "HF_HOME": "/root/models",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/root/hypernodes/src:$PYTHONPATH",  # Key fix!
    })
    .uv_pip_install(
        # Core
        "cloudpickle>=3.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pydantic>=2.0.0",
        # ML
        "pylate",
        "transformers",
        "sentence-transformers",
        "FlagEmbedding",
        "torch",
        "optimum",
        "rank-bm25",
        # Data
        "pyarrow",
        "pytrec_eval",
        # Utils
        "diskcache",
        "networkx",
        "graphviz",
        "rich",
        "tqdm",
    )
    .add_local_dir(
        str(hypernodes_dir / "src"),
        remote_path="/root/hypernodes/src",
    )
)

# Create backend with longer timeout
backend = ModalBackend(
    image=image,
    gpu="A10G",
    timeout=7200,  # 2 hours - adjust based on your needs
    map_execution="sequential",
    volumes={
        "/root/models": models_volume,
        "/root/data": data_volume,
    },
)

# Use in pipeline
pipeline = your_pipeline.with_backend(backend)
results = pipeline.run(inputs=your_inputs)
```

### Step 4: Run Your Pipeline

After fixing the image, run your Hebrew retrieval pipeline again.

## Quick Verification

Before running the full pipeline, test with a minimal example:

```python
from hypernodes import Pipeline, node

@node(output_name="result")
def test_node(x: int) -> int:
    return x + 1

test_pipeline = Pipeline(nodes=[test_node]).with_backend(backend)
result = test_pipeline.run(inputs={"x": 5})
print(result)  # Should print: {'result': 6}
```

If this works, your setup is correct!

## Common Mistakes

### ❌ Wrong: Adding entire hypernodes directory
```python
.add_local_dir(str(hypernodes_dir), remote_path="/root/hypernodes")
```

### ✓ Correct: Adding src folder with PYTHONPATH
```python
.env({"PYTHONPATH": "/root/hypernodes/src:$PYTHONPATH"})
.add_local_dir(str(hypernodes_dir / "src"), remote_path="/root/hypernodes/src")
```

### ❌ Wrong: Short timeout for long pipeline
```python
backend = ModalBackend(image=image, timeout=300)  # 5 minutes
```

### ✓ Correct: Generous timeout
```python
backend = ModalBackend(image=image, timeout=7200)  # 2 hours
```

## Still Having Issues?

### Check Modal Dashboard
1. Go to https://modal.com
2. Check "Apps" section
3. Look for your app's logs
4. See exact error messages

### Check Jupyter Kernel
Connection timeouts can also be caused by:
- Jupyter kernel timeout (increase in settings)
- Network issues between laptop and Modal
- Modal service issues

### Progressive Testing
Test with smaller datasets first:

```python
# Start small
inputs = {
    "num_passages": 10,
    "num_queries": 2,
    "top_k": 5,
}

# Once working, increase
inputs = {
    "num_passages": 100,
    "num_queries": 10,
    "top_k": 20,
}
```

## Expected Behavior

✓ **First run**: 1-2 minutes for image build + cold start, then your pipeline execution time
✓ **Subsequent runs**: ~5-10 seconds overhead, then execution time
✓ **With caching**: Much faster as cached results are reused
✓ **Progress**: You should see progress output if using ProgressCallback

## Summary

The most likely issue is **hypernodes not being in the Python path on Modal**. Fix this by:

1. Add `PYTHONPATH` environment variable to image
2. Copy the `src` folder (not entire repo)
3. Increase timeout if needed
4. Test with diagnostic scripts

**Quick fix script**: `uv run python scripts/fix_modal_hebrew_example.py`

This will test your setup and tell you exactly what's wrong.
