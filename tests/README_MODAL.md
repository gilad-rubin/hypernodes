# Modal Backend Testing

This directory contains comprehensive tests for the Modal backend integration with HyperNodes.

## Architecture

The Modal backend implements **Option 4** from the design:
- **Separation of Concerns**: Placement (local vs Modal) is separate from execution engine (sequential/async/threaded/parallel)
- **Code Reuse**: `PipelineExecutionEngine` delegates to `LocalBackend`, ensuring identical semantics locally and remotely
- **Single-Container by Default**: The entire pipeline runs in one Modal container using `LocalBackend` internally

## Quick Start

### 1. Smoke Tests (Fastest)

Run the quickest verification that Modal is working:

```bash
uv run python scripts/test_modal_smoke.py
```

This runs 5 progressive tests:
1. Simplest single node
2. Map operation
3. `.as_node()` with `map_over` (Hebrew pipeline pattern)
4. Pydantic models
5. Execution configuration

### 2. Pytest Suite (Comprehensive)

Run the full pytest suite:

```bash
# Run all tests
uv run pytest tests/test_modal_backend.py -v

# Run specific test
uv run pytest tests/test_modal_backend.py::test_modal_single_node_simple -v

# Run with detailed output
uv run pytest tests/test_modal_backend.py -v -s

# Run integration test only
uv run pytest tests/test_modal_backend.py -v -m integration
```

### 3. Hebrew-Style Minimal Example

Run a minimal version of the Hebrew retrieval pattern:

```bash
uv run python scripts/test_modal_hebrew_minimal.py
```

This tests:
- Data loading
- Single-item encoding nodes
- `.as_node()` with `map_over` for batch encoding
- Index building
- Query retrieval
- Result aggregation

## Test Progression

The tests increase in complexity:

### Level 1: Basic Execution
- Single node with single input
- Multiple dependent nodes
- Error handling

### Level 2: Map Operations
- Simple `pipeline.map()`
- `.as_node()` with `map_over`
- Execution engine configuration

### Level 3: Pydantic Models
- Frozen models
- Models with `arbitrary_types_allowed`
- Model serialization/deserialization

### Level 4: Hebrew Pipeline Patterns
- Nested pipelines with `.as_node()`
- Map-over-list pattern
- Multi-stage processing
- Result aggregation

### Level 5: Integration
- Full end-to-end pipeline
- Cache integration
- Execution engine modes
- Larger datasets

## Key Features Tested

### 1. Execution Engine Configuration

```python
backend = ModalBackend(
    image=image,
    node_execution="sequential",  # How nodes execute within a run
    map_execution="threaded",      # How map items are processed
    max_workers=4,                 # Worker count for parallel/threaded
    timeout=300,
)
```

### 2. Hebrew Pipeline Pattern

```python
# Single-item pipeline
encode_single = Pipeline(nodes=[encode_item])

# Wrap for batch processing
encode_all = encode_single.as_node(
    input_mapping={"items": "item"},
    output_mapping={"encoded": "all_encoded"},
    map_over="items",
    name="encode_all"
)

# Use in outer pipeline
pipeline = Pipeline(
    nodes=[load_items, encode_all, aggregate]
).with_backend(modal_backend)
```

### 3. Pydantic Model Serialization

```python
class Document(BaseModel):
    id: str
    embedding: Any  # numpy arrays, etc.
    model_config = {"frozen": True, "arbitrary_types_allowed": True}
```

### 4. Cache Integration

```python
pipeline = pipeline.with_cache(DiskCache(path=".cache"))
```

## Troubleshooting

### Cold Start Times
First run will be slower due to Modal image building and cold start. Subsequent runs are faster.

### Authentication
Ensure you're authenticated with Modal:
```bash
modal token new
```

### Image Building
If you need to rebuild the image:
```bash
modal image build
```

### Debugging
Run tests with `-s` flag to see print output:
```bash
uv run pytest tests/test_modal_backend.py -v -s
```

## Common Patterns

### Pattern 1: Simple Remote Execution
```python
pipeline = Pipeline(nodes=[...]).with_backend(
    ModalBackend(image=image)
)
result = pipeline.run(inputs={...})
```

### Pattern 2: Batch Processing
```python
encode_single = Pipeline(nodes=[encode_item])
encode_batch = encode_single.as_node(
    input_mapping={"items": "item"},
    output_mapping={"encoded": "encoded_items"},
    map_over="items",
)
pipeline = Pipeline(nodes=[load, encode_batch])
```

### Pattern 3: Execution Configuration
```python
backend = ModalBackend(
    image=image,
    map_execution="threaded",  # Parallel map items
    max_workers=8,
)
```

## Performance Notes

- **Single Container**: All processing happens in one Modal container by default
- **LocalBackend Inside**: Uses the same `LocalBackend` logic remotely for consistency
- **Caching**: Works across runs if using persistent Modal volumes
- **Execution Modes**: `sequential`, `async`, `threaded`, `parallel` work identically locally and on Modal

## Next Steps

1. **Run smoke tests** to verify basic functionality
2. **Run pytest suite** for comprehensive coverage
3. **Test with your actual Hebrew pipeline** using patterns from `test_modal_hebrew_minimal.py`
4. **Configure execution modes** for your workload

## Future Enhancements

- **Distributed Map**: Multiple Modal containers for large-scale parallel processing (not yet implemented)
- **Streaming Results**: Progressive result streaming for long-running jobs
- **Resource Optimization**: Per-node resource configuration
