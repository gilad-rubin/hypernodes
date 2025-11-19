<div align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/dark_background_logo.png">
  <img alt="hypernodes" src="assets/light_background_logo.png" width="700">
</picture></div>

<p align="center">
  <a href="#installation">[Installation]</a> |
  <a href="docs/introduction.mdx">[Documentation]</a> |
  <a href="#license">[License]</a>
</p>

# HyperNodes

**Hierarchical, Modular Data Pipelines for AI/ML**

HyperNodes is designed to make building complex data pipelines intuitive and scalable. It starts with a simple premise: **define your logic on single items**, then compose and scale.

## ✨ Key Features

**Simplicity First: Think Singular**

Define your functions and pipelines as if they are processing a single item (e.g., one PDF, one audio file, one text string). This makes your code:
- **Easy to grasp**: No complex loops or batch logic cluttering your business logic.
- **Testable**: Write standard unit tests for individual functions.
- **Debuggable**: Step through your code easily.

**🪆 Hierarchical Composition**

Once you have your building blocks, compose them into progressively more complex pipelines.
- **Nesting**: Pipelines are nodes. You can use a pipeline as a node inside another pipeline.
- **Mapping**: Seamlessly apply a pipeline over a list of inputs using `.map()`.
- **Visual**: Always keep one level of understanding. Look at the high-level flow, then dive deeper into specific sub-pipelines as needed.

**Structured Data & Protocols**

HyperNodes integrates deeply with **Pydantic** and **dataclasses**. Define reusable protocols and data structures to ensure type safety and clarity throughout your pipeline. This leads to a codebase that is easy to maintain and extend.

**🚀 Powered by Daft: Performance & Caching**

When you're ready to scale, HyperNodes leverages **[Daft](https://www.getdaft.io/)** as its computation engine.
- **Distributed Execution**: Run on your laptop or a cluster without changing code.
- **Intelligent Caching**: Caching is a first-class citizen. Daft handles distributed caching, ensuring you only recompute what's necessary.

---

## 💡 Inspiration

HyperNodes stands on the shoulders of giants. It was inspired by and grew from working with:
- **[Pipefunc](https://github.com/pipefunc/pipefunc)**: For its elegant approach to function composition.
- **[Apache Hamilton](https://github.com/dagworks-inc/hamilton)**: For its paradigm of defining dataflows using standard Python functions.

HyperNodes aims to bring native support for hierarchical pipelines and advanced caching to this ecosystem.

---

## 📚 Documentation

The full documentation is available in the `docs/` directory:

- **[Introduction & Quick Start](docs/introduction.mdx)**
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

# 1. Define functions on a single item
@node(output_name="cleaned_text")
def clean_text(passage: str) -> str:
    return passage.strip().lower()

@node(output_name="word_count")
def count_words(cleaned_text: str) -> int:
    return len(cleaned_text.split())

# 2. Build pipeline
pipeline = Pipeline(nodes=[clean_text, count_words])

# 3. Test with single input
result = pipeline.run(inputs={"passage": "Hello World"})
print(result)  # {'cleaned_text': 'hello world', 'word_count': 2}

# 4. Scale with .map()
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
