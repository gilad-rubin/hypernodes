<div align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/dark_background_logo.png">
  <img alt="hypernodes" src="assets/light_background_logo.png" width="700">
</picture></div>

<p align="center">
  <a href="#installation">[Installation]</a> |
  <a href="docs/00-introduction.md">[Documentation]</a> |
  <a href="#license">[License]</a>
</p>

# HyperNodes

**Build once, cache intelligently, run anywhere.**

HyperNodes is a hierarchical, modular pipeline system with intelligent caching designed for ML/AI development workflows. It treats caching as a first-class citizen, enabling developers to iterate rapidly without re-running expensive computations.

## ✨ Key Features

**🧪 Test with One, Scale to Many**

Build and test your pipeline with a single input, then run it over thousands of inputs without changing a line of code. Keep your code simple, unit-testable, and debuggable while enabling production-scale batch processing.

**💾 Intelligent Caching**

During development, we run pipelines dozens of times with minor tweaks. HyperNodes automatically caches at node and example granularity and only re-runs what changed.

**🪆 Hierarchical Modularity**

Functions are nodes. Pipelines are made out of nodes, and Pipelines are nodes themselves. Build complex workflows from simple, reusable pieces.

**⚡ Flexible Execution**

Run pipelines with different execution strategies: sequential for debugging, async for I/O-bound workloads, or **distributed parallel execution with [Daft](https://www.getdaft.io/)** for high-performance data processing.

---

## 📚 Documentation

The full documentation is available in the `docs/` directory:

- **[Introduction & Quick Start](docs/00-introduction.md)**
- **Core Concepts**
    - [Nodes](docs/01-core-concepts/01-nodes.md)
    - [Pipelines](docs/01-core-concepts/02-pipelines.md)
    - [Execution](docs/01-core-concepts/03-execution.md)
- **Data Processing**
    - [Mapping (Parallel Processing)](docs/02-data-processing/01-mapping.md)
    - [Nesting Pipelines](docs/02-data-processing/02-nesting.md)
    - [Caching](docs/02-data-processing/03-caching.md)
- **Scaling & Optimization**
    - [Engines Overview](docs/03-scaling/01-engines.md)
    - [Distributed Daft Engine](docs/03-scaling/02-daft-engine.md)
    - [Stateful Parameters](docs/03-scaling/03-stateful.md)
- **Observability**
    - [Visualization](docs/04-observability/01-visualization.md)
    - [Telemetry & Progress](docs/04-observability/02-telemetry.md)

---

## 🚀 Quick Start

### Installation

```bash
pip install hypernodes
# or with uv
uv add hypernodes
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

### High-Performance Execution with Daft

```python
from hypernodes import Pipeline
from hypernodes.engines import DaftEngine

# Distributed execution using Daft
# Requires: pip install getdaft
engine = DaftEngine(use_batch_udf=True)
pipeline = Pipeline(nodes=[clean_text, count_words], engine=engine)

# Auto-batches and executes in parallel
# Each item is cached independently
results = pipeline.map(
    inputs={"passage": ["Hello", "World"] * 1000},
    map_over="passage"
)
```

---

## 📄 License

MIT License - see LICENSE file for details.
