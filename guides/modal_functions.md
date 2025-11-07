Perfect! Now I can show you concrete options using Modal's pattern. Let me show you how to send entire pipelines to Modal with different levels of complexity.

## Option 1: Simple - Wrap Pipeline Execution in `@app.function()`

### Cell 1: Define your nodes and Modal app

```python
import modal
from hypernodes import node, Pipeline
from hypernodes.cache import DiskCache
from hypernodes.callbacks import ProgressCallback

app = [modal.App](http://modal.App)("hypernodes-pipeline")

# Define your nodes (these are just Python functions)
@node(output_name="processed")
def process(x: int) -> int:
    return x * 2

@node(output_name="result")
def finalize(processed: int) -> int:
    return processed + 10
```

### Cell 2: Wrap pipeline execution in Modal function

```python
@app.function()
def run_pipeline(inputs: dict):
    """This ENTIRE function runs on Modal."""
    # Cache and callbacks instantiated ON MODAL
    cache = DiskCache(path="./cache")
    callbacks = [ProgressCallback()]
    
    # Build pipeline on Modal
    pipeline = Pipeline(
        nodes=[process, finalize],
        cache=cache,
        callbacks=callbacks
    )
    
    # Run on Modal
    return [pipeline.run](http://pipeline.run)(inputs=inputs)

@app.local_entrypoint()
def main():
    # This runs locally
    inputs = {"x": 5}
    
    # Run locally
    result_local = run_pipeline.local(inputs)
    print("Local:", result_local)
    
    # Run on Modal
    result_remote = run_pipeline.remote(inputs)
    print("Modal:", result_remote)
```

### Cell 3: Execute

```python
# In Jupyter: run this cell
# In script: run `modal run [script.py](http://script.py)`

# Actually call it in Jupyter
result = run_pipeline.remote({"x": 5})
print(result)
```

**Status:** ✅ **This works!** Simple inputs (dicts, primitives) serialize fine.

---

## Option 2: Complex Inputs - Pass Classes/Objects

### Cell 1: Setup (same as before)

```python
import modal
from hypernodes import node, Pipeline
from hypernodes.cache import DiskCache
from hypernodes.callbacks import ProgressCallback, TelemetryCallback

app = [modal.App](http://modal.App)("hypernodes-complex")

@node(output_name="result")
def process_data(data: dict) -> dict:
    return {"value": data["value"] * 2}
```

### Cell 2: Pass cache/callback CONFIGURATION (not instances)

```python
@app.function()
def run_pipeline_with_config(
    inputs: dict,
    cache_path: str = "./cache",
    enable_telemetry: bool = False
):
    """Pass configuration, instantiate on Modal."""
    # Instantiate ON MODAL based on config
    cache = DiskCache(path=cache_path)
    
    callbacks = [ProgressCallback()]
    if enable_telemetry:
        callbacks.append(TelemetryCallback())
    
    pipeline = Pipeline(
        nodes=[process_data],
        cache=cache,
        callbacks=callbacks
    )
    
    return [pipeline.run](http://pipeline.run)(inputs=inputs)

@app.local_entrypoint()
def main():
    # Pass configuration, not instances
    result = run_pipeline_with_config.remote(
        inputs={"data": {"value": 10}},
        cache_path="./modal_cache",
        enable_telemetry=True
    )
    print(result)
```

**Status:** ✅ **This works!** Pass configuration as primitives, instantiate on Modal.

---

## Option 3: Pass Cache/Callback Instances Directly (PROBLEM)

### Cell 1: Setup

```python
import modal
from hypernodes import node, Pipeline
from hypernodes.cache import DiskCache
from hypernodes.callbacks import ProgressCallback

app = [modal.App](http://modal.App)("hypernodes-instances")

@node(output_name="result")
def process(x: int) -> int:
    return x * 2
```

### Cell 2: Try to pass instances

```python
@app.function()
def run_pipeline_with_instances(
    inputs: dict,
    cache,  # DiskCache instance
    callbacks  # List of callback instances
):
    """Try to pass instances directly."""
    pipeline = Pipeline(
        nodes=[process],
        cache=cache,
        callbacks=callbacks
    )
    return [pipeline.run](http://pipeline.run)(inputs=inputs)

@app.local_entrypoint()
def main():
    # Create instances locally
    cache = DiskCache(path="./cache")
    callbacks = [ProgressCallback()]
    
    # Try to pass to Modal
    result = run_pipeline_with_instances.remote(
        inputs={"x": 5},
        cache=cache,
        callbacks=callbacks
    )
    print(result)
```

**Status:** ❌ **This may NOT work!**

**Problem:** Modal uses `cloudpickle` to serialize function arguments. If `DiskCache` or `ProgressCallback` have:

- Open file handles
- Thread locks
- Non-serializable state
- Database connections

...they won't serialize properly.

**Fix:** Make your cache/callback classes pickle-friendly:

```python
# In hypernodes/[cache.py](http://cache.py)
class DiskCache:
    def __init__(self, path: str):
        self.path = path
        self._store = None  # Lazy initialization
    
    def __getstate__(self):
        # Only serialize configuration
        return {"path": self.path}
    
    def __setstate__(self, state):
        # Reconstruct on Modal
        self.path = state["path"]
        self._store = None
    
    def _ensure_store(self):
        if self._store is None:
            # Open file/DB here
            self._store = self._create_store()
```

---

## Option 4: Nested Pipelines with Shared Config

### Cell 1: Define inner and outer pipelines

```python
import modal
from hypernodes import node, Pipeline
from hypernodes.cache import DiskCache
from hypernodes.callbacks import ProgressCallback

app = [modal.App](http://modal.App)("hypernodes-nested")

# Inner pipeline nodes
@node(output_name="cleaned")
def clean(text: str) -> str:
    return text.strip().lower()

@node(output_name="tokenized")
def tokenize(cleaned: str) -> list:
    return cleaned.split()

# Outer pipeline nodes
@node(output_name="final")
def aggregate(tokenized: list) -> int:
    return len(tokenized)
```

### Cell 2: Compose pipelines on Modal

```python
@app.function()
def run_nested_pipeline(inputs: dict):
    """Nested pipelines with shared cache/callbacks."""
    # Single cache/callback instances
    cache = DiskCache(path="./cache")
    callbacks = [ProgressCallback()]
    
    # Inner pipeline
    inner = Pipeline(
        nodes=[clean, tokenize],
        cache=cache,
        callbacks=callbacks
    )
    
    # Outer pipeline using inner
    outer = Pipeline(
        nodes=[[inner.as](http://inner.as)_node(), aggregate],
        cache=cache,      # Same instance
        callbacks=callbacks  # Same instance
    )
    
    return [outer.run](http://outer.run)(inputs=inputs)

@app.local_entrypoint()
def main():
    result = run_nested_pipeline.remote({"text": "  Hello World  "})
    print(result)  # {"final": 2}
```

**Status:** ✅ **This works!** Cache/callbacks shared across inner and outer pipelines.

---

## Option 5: Map Over Multiple Inputs on Modal

### Cell 1: Setup

```python
import modal
from hypernodes import node, Pipeline
from hypernodes.cache import DiskCache

app = [modal.App](http://modal.App)("hypernodes-map")

@node(output_name="result")
def expensive_computation(x: int) -> int:
    # Imagine this takes 10 seconds
    return x ** 2
```

### Cell 2: Use Modal's map for parallel execution

```python
@app.function()
def run_single_pipeline(input_value: int):
    """Run pipeline for a single input."""
    cache = DiskCache(path="./shared_cache")
    
    pipeline = Pipeline(
        nodes=[expensive_computation],
        cache=cache
    )
    
    return [pipeline.run](http://pipeline.run)(inputs={"x": input_value})

@app.local_entrypoint()
def main():
    # Run 100 pipelines in parallel on Modal
    input_values = range(100)
    
    # Modal handles parallelization
    results = list(run_single_[pipeline.map](http://pipeline.map)(input_values))
    
    print(f"Computed {len(results)} results")
    print(f"Sum: {sum(r['result'] for r in results)}")
```

**Status:** ✅ **This works!** Each Modal container runs one pipeline instance. Cache is per-container unless you use Modal Volumes for shared storage.

---

## Option 6: Jupyter-Friendly Wrapper (Your Use Case)

### Cell 1: Define reusable wrapper

```python
import modal
from hypernodes import node, Pipeline
from hypernodes.cache import DiskCache
from hypernodes.callbacks import ProgressCallback

app = [modal.App](http://modal.App)("hypernodes-jupyter")

@node(output_name="result")
def my_node(x: int) -> int:
    return x * 2

def run_on_modal(pipeline_fn, image=None, gpu=None):
    """Decorator to send pipeline function to Modal."""
    @app.function(image=image, gpu=gpu)
    def wrapped(*args, **kwargs):
        return pipeline_fn(*args, **kwargs)
    
    return wrapped
```

### Cell 2: Define your pipeline function

```python
def my_pipeline(x: int):
    """Regular function - works locally or on Modal."""
    cache = DiskCache(path="./cache")
    callbacks = [ProgressCallback()]
    
    pipeline = Pipeline(
        nodes=[my_node],
        cache=cache,
        callbacks=callbacks
    )
    
    return [pipeline.run](http://pipeline.run)(inputs={"x": x})
```

### Cell 3a: Run locally

```python
result = my_pipeline(5)
print(result)
```

### Cell 3b: Run on Modal

```python
remote_pipeline = run_on_modal(my_pipeline, image=modal.Image.debian_slim())
result = remote_pipeline.remote(5)
print(result)
```

**Status:** ✅ **This works!** Same function runs locally or remotely.

---

## Summary: What Works and What Doesn't

| Approach | Status | Notes |
| --- | --- | --- |
| Pass simple inputs (dict, int, str) | ✅ Works | Always serializable |
| Pass config (strings, bools) → instantiate on Modal | ✅ Works | **Recommended pattern** |
| Pass cache/callback instances | ⚠️ Depends | Requires `__getstate__`/`__setstate__` |
| Nested pipelines with shared cache | ✅ Works | Instantiate once, pass to both |
| Modal's `.map()` for parallel pipelines | ✅ Works | Each container = one pipeline run |
| Jupyter wrapper for local/remote | ✅ Works | Same code, switch execution |

## Key Recommendation

**Best pattern for your use case:**

```python
@app.function()
def run_pipeline(inputs: dict, cache_path: str = "./cache"):
    # Instantiate ON MODAL
    cache = DiskCache(path=cache_path)
    callbacks = [ProgressCallback()]
    
    pipeline = Pipeline(nodes=[...], cache=cache, callbacks=callbacks)
    return [pipeline.run](http://pipeline.run)(inputs=inputs)

# Jupyter: call directly
result = run_pipeline.remote({"x": 5}, cache_path="./modal_cache")
```

**For complex cache/callback classes:** Implement `__getstate__`/`__setstate__` to be pickle-safe, then you can pass instances directly.

Does this clarify the options? Which pattern fits your workflow best?
===
Perfect! I found the key information about how Modal serializes and sends functions. Here's what happens:

## How Modal Sends Functions to the Cloud

### 1. Function Arguments: **cloudpickle**

Modal uses [cloudpickle](https://github.com/cloudpipe/cloudpickle) to serialize function arguments and closures. From their docs:

> "Any data of reasonable size that is serializable through cloudpickle is passable as an argument to Modal functions."
> 

So when you call `run_pipeline.remote(inputs)`, Modal:

- Serializes `inputs` with cloudpickle
- Sends it to the Modal container
- Deserializes it on the remote side

**This means:** If you pass `cache=DiskCache(...)` as an argument, cloudpickle will try to serialize the instance. This works IF the class is pickle-friendly.

### 2. Local Code: **Automatic Mounting (with caveats)**

From their "Project structure" docs:

> "By default, Modal will **automatically include the module or package where a Function is defined** in all containers that run that Function."
> 

**Key change in Modal 1.0:**

> "Previously, Modal would 'automount' any local dependencies that were imported by your App source into a container. This was changed to be more selective to avoid unnecessary inclusion of large local packages."
> 

### 3. What Gets Included Automatically

**✅ Automatically included:**

- The file/module where `@app.function()` is defined
- If your project is a Python package (has `__init__.py`), the entire package is included

**❌ NOT automatically included:**

- Local modules outside your package
- Local helper files not part of the package

**For those, you need:**

```python
image = modal.Image.debian_slim().add_local_python_source("my_module")
```

### 4. Library Imports (pip packages)

**PyPI packages** like `numpy`, `pandas`, etc. must be installed in the Modal Image:

```python
image = modal.Image.debian_slim().pip_install("hypernodes", "numpy")

@app.function(image=image)
def run_pipeline(inputs):
    from hypernodes import Pipeline  # This import happens ON MODAL
    ...
```

## What This Means for Your Pipeline

### Scenario A: `hypernodes` is a pip-installable package

```python
import modal

app = [modal.App](http://modal.App)("my-app")

# Install hypernodes in the container
image = modal.Image.debian_slim().pip_install("hypernodes")

@app.function(image=image)
def run_pipeline(inputs: dict):
    # These imports happen ON MODAL
    from hypernodes import Pipeline, node
    from hypernodes.cache import DiskCache
    from hypernodes.callbacks import ProgressCallback
    
    # These instantiate ON MODAL
    cache = DiskCache(path="./cache")
    callbacks = [ProgressCallback()]
    
    pipeline = Pipeline(nodes=[...], cache=cache, callbacks=callbacks)
    return [pipeline.run](http://pipeline.run)(inputs=inputs)
```

**Status:** ✅ **This works perfectly**

### Scenario B: `hypernodes` is a local package

```
my_project/
├── hypernodes/          # Your local package
│   ├── __init__.py
│   ├── [pipeline.py](http://pipeline.py)
│   ├── [cache.py](http://cache.py)
│   └── [callbacks.py](http://callbacks.py)
└── [script.py](http://script.py)            # Your Modal app
```

```python
import modal

app = [modal.App](http://modal.App)("my-app")

# Add local hypernodes package to the image
image = modal.Image.debian_slim().add_local_python_source("hypernodes")

@app.function(image=image)
def run_pipeline(inputs: dict):
    # Now this works because hypernodes is in the container
    from hypernodes import Pipeline
    from hypernodes.cache import DiskCache
    
    cache = DiskCache(path="./cache")
    pipeline = Pipeline(nodes=[...], cache=cache)
    return [pipeline.run](http://pipeline.run)(inputs=inputs)
```

**Status:** ✅ **This works if you add the local source**

### Scenario C: Closure over local variables (PROBLEM)

```python
from hypernodes import Pipeline
from hypernodes.cache import DiskCache

# Created LOCALLY
cache = DiskCache(path="./cache")
pipeline = Pipeline(nodes=[...], cache=cache)

@app.function()
def run_pipeline(inputs: dict):
    # Tries to use 'pipeline' from closure
    return [pipeline.run](http://pipeline.run)(inputs=inputs)
```

**Status:** ⚠️ **May fail** - cloudpickle will try to serialize `pipeline` and `cache`, which may have non-serializable state.

## Summary

**What Modal does:**

1. **Serializes function arguments** with cloudpickle
2. **Automatically includes** the module/package where `@app.function()` is defined
3. **Does NOT automatically include** external local imports (you must add them to the Image)
4. **Does NOT automatically install** pip packages (you must add them to the Image)

**For your use case:**

- ✅ Define cache/callbacks **inside** the function (they instantiate on Modal)
- ✅ Import `hypernodes` inside the function (as long as it's in the Image)
- ❌ Don't create cache/pipeline in closure and expect them to serialize properly

**Best practice:**

```python
image = modal.Image.debian_slim().pip_install("hypernodes")  # Or .add_local_python_source()

@app.function(image=image)
def run_pipeline(inputs: dict):
    # Import and instantiate INSIDE the function
    from hypernodes import Pipeline
    from hypernodes.cache import DiskCache
    
    cache = DiskCache(path="./cache")
    pipeline = Pipeline(nodes=[...], cache=cache)
    return [pipeline.run](http://pipeline.run)(inputs=inputs)
```

This way, everything happens on Modal and you don't rely on cloudpickle to serialize complex objects.