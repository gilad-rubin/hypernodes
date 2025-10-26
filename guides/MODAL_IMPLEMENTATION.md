# Modal Backend Implementation - Option 4

## Overview

Implemented a clean separation between **placement** (where code runs) and **execution engine** (how code runs), allowing Modal to execute pipelines with identical semantics to `LocalBackend`.

## Architecture

### PipelineExecutionEngine
- **Purpose**: Reusable execution strategy encapsulation
- **Location**: `src/hypernodes/backend.py`
- **Key Feature**: Delegates to `LocalBackend` internally, ensuring single code path

```python
class PipelineExecutionEngine:
    """Reusable execution engine for pipelines."""
    
    def __init__(
        self,
        node_execution: Literal["sequential", "async", "threaded", "parallel"],
        map_execution: Literal["sequential", "async", "threaded", "parallel"],
        max_workers: Optional[int] = None,
        executor: Optional[Executor] = None,
    ):
        # Lazy-construct LocalBackend with these settings
        self._local_backend = None
    
    def run(self, pipeline, inputs, ctx):
        return self._ensure_local().run(pipeline, inputs, ctx)
```

### ModalBackend (Single-Container Mode)
- **Purpose**: Execute entire pipeline in one Modal container
- **Execution**: Serializes pipeline + engine config, reconstructs `LocalBackend` remotely
- **Benefits**: 
  - Avoids connection timeouts from many small `.remote()` calls
  - Reuses all `LocalBackend` logic (caching, callbacks, nested pipelines)
  - Identical semantics to local execution

```python
class ModalBackend(Backend):
    def __init__(
        self,
        image: modal.Image,
        # Resource config
        gpu: Optional[str] = None,
        memory: Optional[str] = None,
        # Execution engine config (reused remotely)
        node_execution: Literal["sequential", "async", "threaded", "parallel"] = "sequential",
        map_execution: Literal["sequential", "async", "threaded", "parallel"] = "sequential",
        max_workers: Optional[int] = None,
        ...
    ):
        # Store engine config to send to remote
        self._engine_config = {
            "node_execution": node_execution,
            "map_execution": map_execution,
            "max_workers": max_workers,
        }
```

### Remote Execution Flow

1. **Serialize**:
   - Pipeline (with `backend=None` to avoid recursion)
   - Inputs
   - Context data
   - Engine configuration

2. **Submit to Modal**:
   - Single `.remote()` call with serialized payload
   - Modal spins up container with configured resources

3. **Execute Remotely**:
   - Deserialize payload
   - Create `PipelineExecutionEngine` with same config
   - Assign engine's `LocalBackend` to pipeline
   - Run using familiar `LocalBackend` logic

4. **Return**:
   - Serialize results
   - Return to caller
   - Deserialize

## Usage

### Basic Usage
```python
import modal
from hypernodes import Pipeline, node
from hypernodes.backend import ModalBackend

@node(output_name="result")
def process(x: int) -> int:
    return x * 2

# Create Modal image
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("cloudpickle")

# Create backend
backend = ModalBackend(image=image, gpu="A10G")

# Run pipeline
pipeline = Pipeline(nodes=[process]).with_backend(backend)
result = pipeline.run(inputs={"x": 21})
# result == {"result": 42}
```

### Hebrew Pipeline Pattern
```python
# Single-item encoding
@node(output_name="encoded")
def encode_item(item: Document, encoder: Encoder) -> EncodedDoc:
    return EncodedDoc(embedding=encoder.encode(item.text))

# Inner pipeline
encode_single = Pipeline(nodes=[encode_item])

# Wrap for batch processing
encode_all = encode_single.as_node(
    input_mapping={"items": "item"},
    output_mapping={"encoded": "all_encoded"},
    map_over="items",
    name="encode_all"
)

# Full pipeline
pipeline = Pipeline(
    nodes=[load_items, encode_all, aggregate],
    backend=ModalBackend(image=image, gpu="A10G"),
    cache=DiskCache(path=".cache"),
)

# Run remotely with caching and progress
results = pipeline.run(inputs={"num_items": 1000})
```

### Execution Configuration
```python
backend = ModalBackend(
    image=image,
    gpu="A10G",
    # Execution strategy (used inside Modal container)
    node_execution="sequential",  # How nodes in a run execute
    map_execution="threaded",     # How map items are processed
    max_workers=8,                # Workers for parallel/threaded
)
```

## Key Benefits

1. **Single Code Path**: `LocalBackend` logic used everywhere (local and remote)
2. **No Connection Issues**: One remote call per `pipeline.run()`, not per map item
3. **Identical Semantics**: Caching, callbacks, nested pipelines work the same
4. **Clean Architecture**: Separation of placement and execution concerns
5. **Future-Proof**: Easy to add distributed map mode later

## Testing

### Quick Smoke Test
```bash
uv run python scripts/test_modal_smoke.py
```

### Full Test Suite
```bash
uv run pytest tests/test_modal_backend.py -v
```

### Hebrew-Style Minimal
```bash
uv run python scripts/test_modal_hebrew_minimal.py
```

See `tests/README_MODAL.md` for detailed testing guide.

## Implementation Files

- `src/hypernodes/backend.py`: Core implementation
  - `Backend` ABC (lines 69-115)
  - `PipelineExecutionEngine` (lines 118-169)
  - `LocalBackend` (unchanged, lines 172-1570)
  - `ModalBackend` (updated, lines 1573-1802)

- `tests/test_modal_backend.py`: Comprehensive pytest suite
  - 10 progressive tests from simple to complex
  - Tests all patterns used in Hebrew pipeline

- `scripts/test_modal_smoke.py`: Quick verification
  - 5 smoke tests
  - Fast feedback for development

- `scripts/test_modal_hebrew_minimal.py`: Minimal Hebrew pipeline
  - Mock encoder/index
  - Tests full retrieval pattern
  - No external dependencies

## Design Decisions

### Why Option 4?
- **Reusability**: Same execution logic everywhere
- **Extensibility**: Easy to add new placement backends (Ray, SageMaker, etc.)
- **Simplicity**: Clean mental model (placement + engine)

### Why Single-Container First?
- **Solves immediate problem**: Jupyter connection timeouts
- **Simplest to implement**: No chunking/distribution logic
- **Sufficient for most workloads**: Modal containers can be quite large

### Why PipelineExecutionEngine?
- **Avoids duplication**: Don't reimplement LocalBackend logic
- **Ensures consistency**: Identical behavior local and remote
- **Flexible**: Can swap backends without changing pipeline code

## Future Enhancements (Not Implemented)

### Distributed Map Mode
- Split large maps across multiple Modal containers
- Use Modal's `.map()` for parallel distribution
- Configurable via `mode="distributed"` parameter

### Chunked Execution
- Batch map items into chunks
- Process chunks in parallel
- Configurable chunk size and max concurrent

### Per-Node Configuration
- Different resources per node type
- Selective GPU allocation
- Mixed local/remote execution

## Migration Guide

### From LocalBackend
```python
# Before
pipeline = Pipeline(nodes=[...], backend=LocalBackend())

# After (runs on Modal, same semantics)
pipeline = Pipeline(nodes=[...], backend=ModalBackend(image=image))
```

### From Manual Modal Integration
```python
# Before (many .remote() calls, connection issues)
@app.function()
def process_item(item):
    ...
results = [process_item.remote(item) for item in items]

# After (single .remote() call, uses LocalBackend internally)
pipeline = Pipeline(nodes=[process_item]).with_backend(modal_backend)
results = pipeline.map(inputs={"item": items}, map_over="item")
```

## Performance Characteristics

- **Cold Start**: ~10-30s for image build + container start
- **Warm Runs**: ~1-3s overhead per run
- **Throughput**: Same as LocalBackend (runs inside single container)
- **Scalability**: Limited to single container resources (for now)

## Summary

Option 4 implementation provides:
- ✅ Clean separation of placement and execution
- ✅ Code reuse (single LocalBackend logic path)
- ✅ Single-container Modal execution (solves connection issues)
- ✅ Identical semantics (caching, callbacks, nested pipelines)
- ✅ Comprehensive test suite
- ✅ Ready for Hebrew pipeline use case
- ⏳ Distributed map mode (future enhancement)
