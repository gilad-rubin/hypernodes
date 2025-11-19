# API Migration Guide: Engine-Centric Architecture

## Overview
HyperNodes has been refactored to separate execution concerns (`cache`, `callbacks`) from pipeline definition. This guide helps you migrate to the new API.

## What Changed

### Old API ❌ (Deprecated)
```python
from hypernodes import Pipeline, DiskCache
from hypernodes.telemetry import ProgressCallback

# Cache and callbacks on Pipeline
pipeline = Pipeline(
    nodes=[node1, node2],
    cache=DiskCache(path=".cache"),
    callbacks=[ProgressCallback()]
)
result = pipeline.run(inputs={"x": 5})
```

### New API ✅ (Current)
```python
from hypernodes import Pipeline, SeqEngine, DiskCache
from hypernodes.telemetry import ProgressCallback

# Cache and callbacks on Engine
engine = SeqEngine(
    cache=DiskCache(path=".cache"),
    callbacks=[ProgressCallback()]
)
pipeline = Pipeline(nodes=[node1, node2], engine=engine)
result = pipeline.run(inputs={"x": 5})
```

## Migration Steps

### 1. Update Imports
```python
# Add SeqEngine import
from hypernodes import Pipeline, SeqEngine, DiskCache
```

### 2. Move Configuration to Engine
**Before:**
```python
pipeline = Pipeline(
    nodes=[...],
    cache=DiskCache(),
    callbacks=[ProgressCallback()]
)
```

**After:**
```python
engine = SeqEngine(
    cache=DiskCache(),
    callbacks=[ProgressCallback()]
)
pipeline = Pipeline(nodes=[...], engine=engine)
```

### 3. Update DaftEngine Usage
**Before:**
```python
from hypernodes.engines import DaftEngine

pipeline = Pipeline(
    nodes=[...],
    engine=DaftEngine(),
    cache=DiskCache(),  # ❌ Not supported
    callbacks=[...]      # ❌ Not supported
)
```

**After:**
```python
from hypernodes.engines import DaftEngine

engine = DaftEngine(
    cache=DiskCache(),
    callbacks=[...],
    use_batch_udf=True
)
pipeline = Pipeline(nodes=[...], engine=engine)
```

## Benefits

1. **Separation of Concerns**: Pipeline defines "what", Engine defines "how"
2. **Consistency**: All engines use shared orchestration logic
3. **Extensibility**: Easy to add new engines without duplicating logic
4. **Type Safety**: Callback engine compatibility checking
5. **Flexibility**: Easy to swap engines per execution

## Common Patterns

### Basic Pipeline (No Changes)
```python
from hypernodes import Pipeline, node

@node(output_name="result")
def process(x: int) -> int:
    return x * 2

# Default engine is SeqEngine with no cache/callbacks
pipeline = Pipeline(nodes=[process])
result = pipeline.run(inputs={"x": 5})
```

### With Caching
```python
from hypernodes import Pipeline, SeqEngine, DiskCache

engine = SeqEngine(cache=DiskCache(path=".cache"))
pipeline = Pipeline(nodes=[...], engine=engine)
```

### With Callbacks
```python
from hypernodes import Pipeline, SeqEngine
from hypernodes.telemetry import ProgressCallback

engine = SeqEngine(callbacks=[ProgressCallback()])
pipeline = Pipeline(nodes=[...], engine=engine)
```

### With Both
```python
engine = SeqEngine(
    cache=DiskCache(path=".cache"),
    callbacks=[ProgressCallback()]
)
pipeline = Pipeline(nodes=[...], engine=engine)
```

### Engine Override Per Execution
```python
# Default engine
pipeline = Pipeline(nodes=[...], engine=SeqEngine())

# Override for specific execution
daft_engine = DaftEngine(use_batch_udf=True)
result = pipeline.run(inputs={...}, engine=daft_engine)
```

## Callback Engine Compatibility

Callbacks can now declare which engines they support:

```python
from hypernodes import PipelineCallback

class DaftOnlyCallback(PipelineCallback):
    @property
    def supported_engines(self):
        return ["DaftEngine"]
```

This fails early if used with incompatible engines:
```python
# ❌ Raises ValueError at execution time
engine = SeqEngine(callbacks=[DaftOnlyCallback()])
pipeline = Pipeline(nodes=[...], engine=engine)
pipeline.run(inputs={})  # ValueError: Callback not compatible with SeqEngine
```

## Removed Methods

The following Pipeline methods have been removed:
- `Pipeline.with_cache()` - use `engine` parameter
- `Pipeline.with_callbacks()` - use `engine` parameter

Use engine configuration instead:
```python
# Old: pipeline.with_cache(DiskCache()).with_callbacks([Progress()])
# New: 
engine = SeqEngine(cache=DiskCache(), callbacks=[Progress()])
pipeline = Pipeline(nodes=[...], engine=engine)
```

## Architecture Benefits

### Before: Coupled Design
```
Pipeline
  ├── cache (DiskCache instance)
  ├── callbacks ([ProgressCallback])
  ├── nodes
  └── engine (execution strategy)
```
- Duplication: Each engine reimplements cache/callback logic
- Testing: Hard to test engines independently
- Consistency: Different engines had different behavior

### After: Separated Design
```
Pipeline (Pure Definition)
  ├── nodes (DAG)
  └── engine → Engine (Runtime)
                  ├── cache
                  ├── callbacks
                  └── ExecutionOrchestrator (shared)
```
- Reusability: Shared orchestration across all engines
- Testing: Easy to test engines in isolation
- Consistency: All engines use same lifecycle management

## See Also

- `REFACTOR_SUMMARY.md` - Implementation details
- `.ruler/2-code_structure.md` - Updated architecture documentation
