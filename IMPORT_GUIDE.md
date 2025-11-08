# HyperNodes Import Guide

Quick reference for importing engines and executors.

## 🚀 Recommended Imports (User-Facing)

### Engines

```python
# Import engines
from hypernodes.engines import Engine, HypernodesEngine, DaftEngine

# Create engine instances
engine = HypernodesEngine(node_executor="threaded")
daft = DaftEngine(collect=True)

# Use with pipelines
pipeline = Pipeline(nodes=[...], backend=engine)
```

### Basic Usage

```python
from hypernodes import Pipeline, node
from hypernodes.engines import HypernodesEngine

@node(output_name="result")
def process(x: int) -> int:
    return x * 2

# Create engine and pipeline
engine = HypernodesEngine(node_executor="async")
pipeline = Pipeline(nodes=[process], backend=engine)

# Run
result = pipeline.run(inputs={"x": 5})
```

## 🔧 Advanced Imports (Framework Development)

### Executor Strategies

```python
# For custom executor implementations
from hypernodes.executor_strategies import SequentialExecutor, AsyncExecutor, DEFAULT_WORKERS

# Create custom executor
executor = AsyncExecutor(max_workers=100)
engine = HypernodesEngine(node_executor=executor)
```

### Direct Integration Access

```python
# Direct access to integration modules
from hypernodes.integrations.daft import DaftEngine

# Configure Daft engine
engine = DaftEngine(
    collect=True,
    show_plan=True,
    debug=False
)
```

## 🔄 Backward Compatibility

Old imports still work (for migration period):

```python
# Old style (still supported)
from hypernodes.executors import HyperNodesEngine, DaftEngine, Engine

# New style (recommended)
from hypernodes.engines import HypernodesEngine, DaftEngine, Engine
```

## 📦 Import Hierarchy

```
hypernodes.engines              # ⭐ Recommended for users
├── Engine                      # Abstract base class
├── HypernodesEngine           # Node-by-node execution
└── DaftEngine                 # Distributed execution (optional)

hypernodes.executor_strategies  # 🔧 For framework developers
├── SequentialExecutor
├── AsyncExecutor
└── DEFAULT_WORKERS

hypernodes.executors            # 🔄 Backward compatibility
├── Engine                      # Re-exported
├── HypernodesEngine           # Re-exported
├── HyperNodesEngine           # Alias (capital N)
└── DaftEngine                 # Re-exported (if available)

hypernodes.integrations         # 🔌 Optional integrations
└── daft/
    └── DaftEngine             # Direct access
```

## 📋 Quick Reference

| What you want | Import statement |
|---------------|------------------|
| Create basic engine | `from hypernodes.engines import HypernodesEngine` |
| Create Daft engine | `from hypernodes.engines import DaftEngine` |
| Engine base class | `from hypernodes.engines import Engine` |
| Custom executors | `from hypernodes.executor_strategies import SequentialExecutor` |
| Direct Daft access | `from hypernodes.integrations.daft import DaftEngine` |

## 🎯 Best Practices

1. **For application code**: Use `from hypernodes.engines import ...`
   ```python
   from hypernodes.engines import HypernodesEngine
   ```

2. **For testing**: Use specific imports
   ```python
   from hypernodes.executor_strategies import SequentialExecutor
   from hypernodes.engines import HypernodesEngine
   ```

3. **For integration development**: Use direct integration imports
   ```python
   from hypernodes.integrations.daft import DaftEngine
   ```

## ⚠️ Common Mistakes

### ❌ Don't mix file and package imports
```python
# Wrong - creates confusion
from hypernodes.executors import SequentialExecutor  # Won't work!
```

### ✅ Use correct import path
```python
# Correct
from hypernodes.executor_strategies import SequentialExecutor
```

### ❌ Don't import from internal modules
```python
# Wrong - internal implementation
from hypernodes.engine import HypernodesEngine
```

### ✅ Use public API
```python
# Correct - public API
from hypernodes.engines import HypernodesEngine
```

## 🔍 Finding the Right Import

**Question**: "How do I import [thing]?"

1. **Engines** (HypernodesEngine, DaftEngine, Engine) → `hypernodes.engines`
2. **Executor strategies** (SequentialExecutor, AsyncExecutor) → `hypernodes.executor_strategies`
3. **Everything else** → Check main package (`hypernodes`) or docs

## 📝 Examples

### Sequential Execution
```python
from hypernodes import Pipeline, node
from hypernodes.engines import HypernodesEngine

engine = HypernodesEngine(node_executor="sequential")
```

### Threaded Execution
```python
from hypernodes.engines import HypernodesEngine

engine = HypernodesEngine(node_executor="threaded", max_workers=4)
```

### Async Execution
```python
from hypernodes.engines import HypernodesEngine

engine = HypernodesEngine(node_executor="async")
```

### Distributed (Daft)
```python
from hypernodes.engines import DaftEngine

engine = DaftEngine(collect=True)
```

### Custom Executor
```python
from concurrent.futures import ThreadPoolExecutor
from hypernodes.engines import HypernodesEngine

custom_executor = ThreadPoolExecutor(max_workers=10)
engine = HypernodesEngine(node_executor=custom_executor)
```
