# Fluent API for Pipeline Configuration

HyperNodes supports a fluent/builder API for configuring pipelines, making complex configurations more readable and composable.

## Methods

### `.with_backend(backend)`

Configure the execution backend.

```python
from hypernodes import Pipeline, LocalBackend, ModalBackend

# Local execution
pipeline = Pipeline(nodes=[...]).with_backend(LocalBackend())

# Remote execution on Modal
pipeline = Pipeline(nodes=[...]).with_backend(
    ModalBackend(image=my_image, gpu="A100")
)
```

### `.with_cache(cache)`

Configure result caching.

```python
from hypernodes import Pipeline, DiskCache

pipeline = Pipeline(nodes=[...]).with_cache(
    DiskCache(path="./cache")
)
```

### `.with_callbacks(callbacks)`

Configure lifecycle callbacks.

```python
from hypernodes import Pipeline
from hypernodes.telemetry import ProgressCallback, TelemetryCallback

pipeline = Pipeline(nodes=[...]).with_callbacks([
    ProgressCallback(),
    TelemetryCallback()
])
```

## Method Chaining

All methods return `self`, enabling fluent chaining:

```python
from hypernodes import Pipeline, DiskCache
from hypernodes.telemetry import ProgressCallback
import modal

# Complex configuration in readable, chainable style
pipeline = (
    Pipeline(nodes=[preprocess, encode, transform, aggregate])
    .with_backend(ModalBackend(
        image=modal.Image.debian_slim().pip_install("torch"),
        gpu="A100",
        memory="32GB",
        timeout=3600
    ))
    .with_cache(DiskCache(path="./remote-cache"))
    .with_callbacks([ProgressCallback()])
)

result = pipeline.run(inputs={...})
```

## Comparison with Traditional API

### Traditional (Constructor Arguments)

```python
pipeline = Pipeline(
    nodes=[node1, node2, node3],
    backend=ModalBackend(image=img, gpu="A100"),
    cache=DiskCache(path="./cache"),
    callbacks=[ProgressCallback()]
)
```

### Fluent (Method Chaining)

```python
pipeline = (
    Pipeline(nodes=[node1, node2, node3])
    .with_backend(ModalBackend(image=img, gpu="A100"))
    .with_cache(DiskCache(path="./cache"))
    .with_callbacks([ProgressCallback()])
)
```

Both approaches work identically. Use whichever style you prefer!

## Conditional Configuration

The fluent API makes conditional configuration easy:

```python
import os
from hypernodes import Pipeline, DiskCache

# Start with basic pipeline
pipeline = Pipeline(nodes=[...])

# Conditionally add features
if os.getenv("USE_CACHE") == "true":
    pipeline = pipeline.with_cache(DiskCache(path="./cache"))

if os.getenv("ENABLE_PROGRESS") == "true":
    from hypernodes.telemetry import ProgressCallback
    pipeline = pipeline.with_callbacks([ProgressCallback()])

# Choose backend based on environment
if os.getenv("ENV") == "production":
    import modal
    pipeline = pipeline.with_backend(
        ModalBackend(image=prod_image, gpu="A100")
    )
else:
    from hypernodes import LocalBackend
    pipeline = pipeline.with_backend(LocalBackend())

result = pipeline.run(inputs={...})
```

## When to Use

**Use Fluent API when:**
- Building complex configurations
- Conditionally adding features
- Prefer method chaining style
- Want clear, self-documenting code

**Use Traditional API when:**
- Simple pipelines with few options
- All configuration is static
- Prefer compact constructor calls

Both approaches have identical functionality and performance.
