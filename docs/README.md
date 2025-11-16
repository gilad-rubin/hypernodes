# HyperNodes Documentation

Build once, cache intelligently, run anywhere.

HyperNodes is a hierarchical, modular pipeline system with intelligent caching for ML/AI workflows. This documentation follows a GitBook-style structure.

## Quick Navigation

- **[Getting Started](getting-started/README.md)** - Install and run your first pipeline
  - [Fluent API](getting-started/fluent-api.md) - Clean typed interface (Recommended!)
  - [Typed Interfaces](getting-started/typed-interfaces.md) - Advanced type generation
- **[In-Depth](in-depth/README.md)** - Core concepts, caching, visualization, and telemetry
- **[Advanced](advanced/README.md)** - Execution engines, async patterns, and distributed execution
- **[Philosophy](philosophy/README.md)** - Design principles and architecture decisions

## Key API Entrypoints

```python
from hypernodes import (
    node,                    # Decorator to create nodes
    Pipeline,                # Main pipeline class
    HypernodesEngine,        # Default execution engine
    DiskCache,               # Persistent caching
)
from hypernodes.telemetry import (
    ProgressCallback,        # Progress tracking
    TelemetryCallback,       # Distributed tracing
)
from hypernodes.engines import (
    DaftEngine,              # Distributed DataFrame execution (optional)
)
```

## Core Concepts in 30 Seconds

```python
from hypernodes import Pipeline, node, DiskCache, HypernodesEngine

# 1. Functions → Nodes (dependencies via parameter names)
@node(output_name="cleaned")
def clean(text: str) -> str:
    return text.strip().lower()

@node(output_name="words")
def tokenize(cleaned: str) -> list:  # Depends on clean()
    return cleaned.split()

# 2. Nodes → Pipeline (automatic DAG resolution)
pipeline = Pipeline(
    nodes=[clean, tokenize],
    cache=DiskCache(path=".cache"),
    engine=HypernodesEngine(node_executor="async")  # I/O-optimized
)

# 3. Single execution
result = pipeline(text="  Hello World  ")
# {'cleaned': 'hello world', 'words': ['hello', 'world']}

# 4. Batch execution with per-item caching
results = pipeline.map(
    inputs={"text": ["  Foo  ", "  Bar  ", "  Baz  "]},
    map_over="text"
)
```
