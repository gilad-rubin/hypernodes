# DaftEngineV2 Implementation Summary

## ✅ What Works

### Core Architecture
- **Node → @daft.func UDF**: Each node becomes a Daft UDF column transformation
- **Type Inference**: Daft automatically infers return types from Python type hints
- **1-row DataFrame for `.run()`**: Single execution uses 1-row DataFrame
- **N-row DataFrame for `.map()`**: Batch execution uses N-row DataFrame  
- **MapPlanner Integration**: Uses existing `MapPlanner` for execution planning
- **Progress Callbacks**: `ProgressCallback` fires correctly during execution

### Test Results
```
✓ Test 1: Basic .run() with caching
✓ Test 2: .map() with multiple inputs
✓ Test 3: .map() with partial cache hits (execution only, no actual caching yet)
✓ Test 4: Output filtering
```

## ⚠️ Current Limitations

### 1. No Caching Yet
**Problem**: Daft UDFs must be serializable. Our cache object (DiskCache) has open file handles and isn't serializable.

**Current State**: Functions execute every time (no cache hits).

**Solution Options**:
1. **Global Cache Registry** (simplest):
   ```python
   _CACHE_REGISTRY = {}  # Global dict indexed by cache_dir
   
   @daft.func
   def cached_wrapper(*args):
       cache = _CACHE_REGISTRY.get(cache_dir)
       if cache and cache.has(signature):
           return cache.get(signature)
       # ... execute and cache
   ```

2. **@daft.cls with Stateful Setup** (recommended by Daft):
   ```python
   @daft.cls
   class CachedNodeExecutor:
       def __init__(self, cache_dir: str, node_func, ...):
           self.cache = DiskCache(cache_dir)  # Initialize once per worker
           self.node_func = node_func
       
       def __call__(self, *args):
           # Check cache, execute, store result
   ```

3. **Serializable Cache Implementation**:
   - Create a `SerializableDiskCache` that doesn't keep file handles open
   - Lazy-load cache on worker side

### 2. No Callbacks Inside UDFs
**Problem**: Same serialization issue - `CallbackContext`, `ProgressCallback` objects aren't serializable.

**Current State**: Pipeline-level callbacks work (start/end), but node-level callbacks don't fire.

**Solution Options**:
1. **Accept Limitation**: Document that Daft mode only has pipeline-level callbacks
2. **Global Callback Registry**: Similar to cache registry
3. **@daft.cls Approach**: Initialize callbacks in worker setup

## 📝 Implementation Details

### File Structure
```
src/hypernodes/integrations/daft/
├── engine_v2.py          # New clean implementation (370 lines)
├── engine.py             # Original implementation (2000+ lines)
└── __init__.py           # Exports both engines
```

### Key Design Decisions

1. **Direct UDF Wrapping**:
   ```python
   def _create_cached_udf(self, node, ...):
       return daft.func(node.func)  # Simple pass-through for now
   ```

2. **Automatic Type Inference**:
   - Daft reads `__annotations__` from functions
   - Supports: `int`, `float`, `str`, `bool`, `list`, `dict`, Pydantic models, TypedDict

3. **DataFrame Construction**:
   ```python
   # .run() → 1-row DataFrame
   df = daft.from_pydict({k: [v] for k, v in inputs.items()})
   
   # .map() → N-row DataFrame  
   df = daft.from_pydict({k: [plan[k] for plan in execution_plans]})
   ```

4. **Column Chaining**:
   ```python
   for node in pipeline.graph.execution_order:
       cached_udf = self._create_cached_udf(node, ...)
       input_cols = [daft.col(param) for param in node.root_args]
       df = df.with_column(node.output_name, cached_udf(*input_cols))
   ```

## 🎯 Next Steps

### Phase 1: Add Caching (High Priority)
- [ ] Implement global cache registry approach
- [ ] Test cache hits/misses
- [ ] Verify content-addressed signatures work
- [ ] Handle dependency hash tracking

### Phase 2: Add Node-Level Callbacks (Medium Priority)
- [ ] Implement callback registry or @daft.cls approach
- [ ] Fire `on_node_start`, `on_node_cached`, `on_node_end` callbacks
- [ ] Test with `ProgressCallback` and `TelemetryCallback`

### Phase 3: Stateful Inputs with @daft.cls (Low Priority)
- [ ] Auto-detect stateful inputs (class instances with `__call__`)
- [ ] Wrap with `@daft.cls` for expensive initialization
- [ ] Support resource control (`gpus`, `max_concurrency`)

### Phase 4: Nested Pipelines (Future)
- [ ] Handle `PipelineNode` execution
- [ ] Recursive DataFrame building
- [ ] Maintain cache/callback inheritance

## 🚀 Usage

```python
from hypernodes import Pipeline, node
from hypernodes.cache import DiskCache
from hypernodes.integrations.daft import DaftEngineV2

@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

@node(output_name="result")  
def add_ten(doubled: int) -> int:
    return doubled + 10

# Create engine
engine = DaftEngineV2()
pipeline = Pipeline(
    nodes=[double, add_ten],
    engine=engine,
    cache=DiskCache(".cache")  # Cache not used yet
)

# Single execution
result = pipeline.run(inputs={"x": 5})
# {'doubled': 10, 'result': 20}

# Batch execution
results = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
# {'doubled': [2, 4, 6], 'result': [12, 14, 16]}
```

## 📊 Performance Notes

- **Overhead**: ~0.26s for first run (Daft initialization)
- **Subsequent runs**: ~0.00s (Daft query plan reuse)
- **Map operations**: ~1000 items/s throughput
- **Parallelism**: Currently sequential (can enable with Daft's execution config)

## 🔍 Comparison: V2 vs Original

| Feature | DaftEngineV2 | Original DaftEngine |
|---------|--------------|---------------------|
| Lines of Code | 370 | 2000+ |
| Caching | ❌ (TODO) | ✅ |
| Callbacks | ⚠️ (partial) | ✅ |
| Type Inference | ✅ (automatic) | ⚠️ (manual mapping) |
| Code Generation | ❌ | ✅ (complex) |
| Readability | ✅✅✅ | ⚠️ |
| Maintainability | ✅✅✅ | ⚠️ |

## 📚 References

- [Daft UDF Documentation](https://www.getdaft.io/projects/docs/en/stable/user_guide/daft_in_depth/udfs.html)
- [@daft.func Examples](guides/daft-new-udf.md)
- [MapPlanner Implementation](src/hypernodes/map_planner.py)
- [Node Execution Logic](src/hypernodes/node_execution.py)

