<div align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/dark_background_logo.png">
  <img alt="hypernodes" src="assets/light_background_logo.png" width="700">
</picture></div>

<p align="center">
  <a href="#installation">[Installation]</a> |
  <a href="#quick-start">[Quick Start]</a> |
  <a href="#license">[License]</a>
</p>

# HyperNodes

**Build once, cache intelligently, run anywhere.**

HyperNodes is a hierarchical, modular pipeline system with intelligent caching designed for ML/AI development workflows. It treats caching as a first-class citizen, enabling developers to iterate rapidly without re-running expensive computations.

## ✨ Key Features

**🧪 Test with One, Scale to Many**

Build and test your pipeline with a single input, then run it over thousands of inputs without changing a line of code. Keep your code simple, unit-testable, and debuggable while enabling production-scale batch processing with intelligent caching.

**💾 Development-First Caching**

During development, we run pipelines dozens of times with minor tweaks. HyperNodes automatically caches at node and example granularity and only re-runs what changed. When you scale to multiple inputs, each item benefits from the cache independently.

**🪆 Hierarchical Modularity**

Functions are nodes. Pipelines are made out of nodes, and Pipelines are nodes themselves. Build complex workflows from simple, reusable pieces.

**⚡ Flexible Execution**

Run pipelines with different execution strategies: sequential for debugging, async for I/O-bound workloads, threaded for mixed workloads, or parallel for CPU-intensive tasks. Choose the right executor for your use case.

**📊 Observable by Default**

Every node execution is tracked, visualized, and measurable. Progress bars, logs, and metrics are built-in, not bolted on.

**🪝 Intelligent Callback System**

Powerful hooks into the execution lifecycle for observability, progress tracking, distributed tracing, and custom instrumentation. Callbacks are composable and don't require modifying pipeline code.

---

## 🚀 Quick Start

### Installation

```bash
pip install hypernodes
```

### Basic Example

```python
from hypernodes import Pipeline, node

# Define functions as nodes
@node(output_name="cleaned_text")
def clean_text(passage: str) -> str:
    return passage.strip().lower()

@node(output_name="word_count")
def count_words(cleaned_text: str) -> int:
    return len(cleaned_text.split())

# Build pipeline - dependencies are automatically resolved
pipeline = Pipeline(nodes=[clean_text, count_words])

# Test with single input
result = pipeline.run(inputs={"passage": "Hello World"})
print(result)  # {'cleaned_text': 'hello world', 'word_count': 2}

# Scale to many inputs - each item cached independently
results = pipeline.map(
    inputs={"passage": ["Hello", "World", "Foo", "Bar"]},
    map_over="passage",
)
```

### With Caching

```python
from hypernodes import Pipeline, DiskCache

# Enable caching
pipeline = Pipeline(
    nodes=[clean_text, count_words],
    cache=DiskCache(path=".cache")
)

# First run: executes all nodes
result1 = pipeline.run(inputs={"passage": "Hello World"})

# Second run: instant cache hit
result2 = pipeline.run(inputs={"passage": "Hello World"})  # Cached!
```

### With Stateful Objects (Models, DB Connections)

```python
from hypernodes import stateful, node, Pipeline

# Mark expensive-to-initialize classes as stateful
@stateful
class ExpensiveModel:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)  # Lazy init - only on first use
    
    def predict(self, text: str) -> str:
        return self.model(text)

@node(output_name="prediction")
def predict(text: str, model: ExpensiveModel) -> str:
    return model.predict(text)

# Create model (doesn't load yet - lazy!)
model = ExpensiveModel("./model.pkl")

pipeline = Pipeline(nodes=[predict])

# Model loads on first item, reused for all 1000 items
results = pipeline.map(
    inputs={"text": texts_1000, "model": model},
    map_over="text"
)
```

### Nested Pipelines

```python
# Inner pipeline for text processing
text_pipeline = Pipeline(nodes=[clean_text, tokenize, normalize])

# Outer pipeline using nested pipeline
main_pipeline = Pipeline(
    nodes=[load_data, text_pipeline, train_model],
)

result = main_pipeline.run(inputs={"data_path": "corpus.txt"})
```

### Parallel Execution with Dask

```python
from hypernodes import Pipeline
from hypernodes.engines import DaskEngine

# Parallel map for CPU/IO-bound workloads
parallel_pipeline = Pipeline(
    nodes=[process_data, transform_results],
    engine=DaskEngine(scheduler="threads"),  # or "processes" for CPU-bound
)

# Regular run (sequential, no overhead)
result = parallel_pipeline.run(inputs={"data": [1, 2, 3]})

# Map operation (parallel via Dask Bag)
results = parallel_pipeline.map(inputs={"data": [1, 2, 3, 4, 5]}, map_over="data")
```

---

## 📚 Core Concepts

### Functions → Nodes

Each function declares its dependencies implicitly through parameter names:

```python
@node(output_name="cleaned_text")
def clean_text(passage: str) -> str:
    return passage.strip().lower()

@node(output_name="word_count")
def count_words(cleaned_text: str) -> int:  # ← depends on cleaned_text
    return len(cleaned_text.split())
```

### Nodes → Pipelines

Pipelines are directed acyclic graphs (DAGs) of nodes:

```python
pipeline = Pipeline(nodes=[clean_text, count_words])

# Visualize the DAG
pipeline.visualize()

# Run with inputs
result = pipeline.run(inputs={"passage": "Hello World"})
```

### Pipelines → Nodes

Pipelines can contain other pipelines, enabling hierarchical composition:

```python
inner_pipeline = Pipeline(nodes=[step1, step2, step3])
outer_pipeline = Pipeline(nodes=[load, inner_pipeline, save])
```

---

## 💾 Intelligent Caching

HyperNodes uses **computation signatures** for content-addressed caching:

```python
sig(node) = hash(
    code_hash      # Function source code
    + env_hash     # Environment (library versions, config)
    + inputs_hash  # Direct input values  
    + deps_hash    # Signatures of upstream nodes (recursive)
)
```

**Core guarantee**: If a node's code, direct inputs, and upstream dependencies haven't changed, its output is guaranteed to be identical—so we skip execution and reuse the cached result.

### Fine-Grained Invalidation

```python
pipeline = Pipeline(nodes=[load_data, preprocess, train_model])

# First run: all nodes execute
result1 = pipeline.run(inputs={"data_path": "data.csv", "learning_rate": 0.01})

# Change only learning_rate
result2 = pipeline.run(inputs={"data_path": "data.csv", "learning_rate": 0.001})
# ✅ load_data: CACHED (unchanged)
# ✅ preprocess: CACHED (unchanged)
# ❌ train_model: RE-RUN (learning_rate changed)
```

### Per-Item Caching with `.map()`

```python
# First run: process 100 items
results1 = pipeline.map(
    inputs={"passage": passages_100},
    map_over="passage",
)

# Add 50 new items
results2 = pipeline.map(
    inputs={"passage": passages_150},
    map_over="passage",
)
# ✅ First 100 items: CACHED
# ❌ 50 new items: EXECUTE
```

---

## 🖥️ Execution Engines

Engines determine **how** (execution strategy) and **where** (infrastructure) nodes execute.

### SequentialEngine (Default)

The default engine for simple, predictable execution:

```python
from hypernodes import Pipeline, SequentialEngine

# Sequential execution (default - no need to specify)
pipeline = Pipeline(nodes=[...])

# Or explicitly:
pipeline = Pipeline(
    nodes=[...],
    engine=SequentialEngine()
)
```

**Features:**
- Simple topological execution
- No parallelism overhead
- Easy debugging
- Best for development and testing

### DaskEngine (Parallel Map Operations)

For parallel execution using Dask Bag:

```python
from hypernodes import Pipeline
from hypernodes.engines import DaskEngine

# Auto-optimized for your workload
engine = DaskEngine()
pipeline = Pipeline(nodes=[...], engine=engine)

# Regular run (sequential, no overhead)
result = pipeline.run(inputs={"x": 5})

# Map operation (parallel via Dask Bag)
results = pipeline.map(
    inputs={"x": [1, 2, 3, 4, 5]},
    map_over="x"
)

# Custom configuration for CPU-bound workload
engine = DaskEngine(
    scheduler="processes",  # or "threads" (default)
    workload_type="cpu",    # or "io", "mixed" (default)
    num_workers=8           # defaults to CPU count
)
```

**Features:**
- Automatic parallelism for map operations
- Configurable scheduler (threads, processes)
- Auto-optimized partitioning
- Zero overhead for non-map operations

### DaftEngine (Distributed DataFrames)

For distributed DataFrame-based execution:

```python
from hypernodes import Pipeline
from hypernodes.engines import DaftEngine

# Requires: pip install getdaft
engine = DaftEngine(use_batch_udf=True)
pipeline = Pipeline(nodes=[...], engine=engine)

# All operations are lazy (builds computation graph)
result = pipeline.run(inputs={"x": 5})

# Map operations leverage Daft's distributed execution
results = pipeline.map(
    inputs={"x": [1, 2, 3, 4, 5]},
    map_over="x"
)
```

**Features:**
- Lazy DataFrame execution
- Batch UDF optimization
- Auto-tuned parallelism
- Best for large-scale distributed workloads

---

## 📊 Observability

### Progress Tracking

```python
from hypernodes import Pipeline
from hypernodes.telemetry import ProgressCallback

pipeline = Pipeline(
    nodes=[...],
    callbacks=[ProgressCallback()],
)

result = pipeline.run(inputs={"data": "..."})
```

**Output:**

```
Processing Pipeline ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:42
  ├─ clean_text ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:05
  ├─ extract_features ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:20
  └─ train_model ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:17
```

### Distributed Tracing

```python
from hypernodes import Pipeline
from hypernodes.telemetry import TelemetryCallback

pipeline = Pipeline(
    nodes=[...],
    callbacks=[TelemetryCallback()],
)

# Traces are automatically sent to OpenTelemetry-compatible systems
# (Jaeger, Zipkin, Logfire, etc.)
```

### Pipeline Visualization

```python
# Visualize the DAG
pipeline.visualize()

# Save to file
pipeline.visualize(filename="pipeline.svg")

# Control nested pipeline expansion
pipeline.visualize(depth=2)  # Show one level of nesting
```

---

## 🪆 Advanced: Nested Pipelines

### Using `.as_node()` for Interface Adaptation

```python
# Inner pipeline processes ONE item
@node(output_name="embedding")
def encode_text(passage: str) -> Vector:
    return model.encode(passage)

single_encode = Pipeline(nodes=[clean_text, encode_text])

# Adapt interface: map over corpus, rename inputs/outputs
encode_corpus = single_encode.as_node(
    input_mapping={"corpus": "passage"},  # outer → inner
    output_mapping={"embedding": "encoded_corpus"},  # inner → outer
    map_over="corpus",  # Map over corpus list
)

# Use in outer pipeline
index_pipeline = Pipeline(nodes=[encode_corpus, build_index])

# From outer perspective: encode_corpus takes List[str], returns List[Vector]
result = index_pipeline.run(inputs={"corpus": ["Hello", "World", "Foo"]})
```

### Hierarchical Configuration

Nested pipelines inherit configuration from parents but can override:

```python
# Parent defines defaults
from hypernodes import Pipeline, DiskCache
from hypernodes.engines import DaskEngine
from hypernodes.telemetry import ProgressCallback

parent = Pipeline(
    nodes=[preprocess, child_pipeline, postprocess],
    engine=DaskEngine(scheduler="threads"),
    cache=DiskCache(path=".cache"),
    callbacks=[ProgressCallback()],
)

# Child inherits all configuration
child_pipeline = Pipeline(
    nodes=[step1, step2]
    # Inherits: DaskEngine, DiskCache, ProgressCallback
)

# Grandchild overrides engine only (e.g., to process-based parallelism)
grandchild_pipeline = Pipeline(
    nodes=[cpu_intensive_step],
    engine=DaskEngine(scheduler="processes"),  # Override for CPU-bound tasks
    # Inherits: DiskCache, ProgressCallback
)
```

---

## 🧪 Testing

HyperNodes is designed to be easily testable:

```python
import pytest
from hypernodes import Pipeline, node

@node(output_name="result")
def my_function(input: str) -> str:
    return input.upper()

def test_single_node():
    pipeline = Pipeline(nodes=[my_function])
    result = pipeline.run(inputs={"input": "hello"})
    assert result["result"] == "HELLO"

def test_with_cache():
    cache = DiskCache(path="/tmp/test_cache")
    pipeline = Pipeline(nodes=[my_function], cache=cache)
    
    # First run
    result1 = pipeline.run(inputs={"input": "hello"})
    
    # Second run should hit cache
    result2 = pipeline.run(inputs={"input": "hello"})
    
    assert result1 == result2
```

---

## 🎯 Design Principles

1. **Simple by default, powerful when needed** - Start with basic pipelines, scale to complex workflows
2. **Cache-first** - Treat caching as core functionality, not an afterthought
3. **Test with one, scale to many** - Same code for single items and batch processing
4. **Hierarchical everything** - Composition through nesting at all levels
5. **Flexible execution** - Choose the right execution strategy for your workload
6. **Observable** - Visibility into execution is built-in

---

## 📄 License

MIT License - see LICENSE file for details.
