# Quick Start

Get started with HyperNodes in 5 minutes!

## Installation

```bash
pip install hypernodes
```

## Basic Example

```python
from hypernodes import Pipeline, node

# 1. Define functions as nodes
@node(output_name="cleaned_text")
def clean_text(passage: str) -> str:
    return passage.strip().lower()

@node(output_name="word_count")
def count_words(cleaned_text: str) -> int:
    return len(cleaned_text.split())

# 2. Build pipeline (dependencies auto-resolved from parameter names)
pipeline = Pipeline(nodes=[clean_text, count_words])

# 3. Run with single input (two equivalent syntaxes)
result = pipeline(passage="  Hello World  ")
print(result)
# {'cleaned_text': 'hello world', 'word_count': 2}

result = pipeline.run(inputs={"passage": "  Hello World  "})
print(result)
# {'cleaned_text': 'hello world', 'word_count': 2}
```

## Map Over Multiple Inputs

Scale from one to many without changing code:

```python
# Process 100 texts
results = pipeline.map(
    inputs={"passage": ["  Hello  ", "  World  ", "  Foo  ", "  Bar  "]},
    map_over="passage"
)
print(results)
# [
#   {'cleaned_text': 'hello', 'word_count': 1},
#   {'cleaned_text': 'world', 'word_count': 1},
#   {'cleaned_text': 'foo', 'word_count': 1},
#   {'cleaned_text': 'bar', 'word_count': 1}
# ]
```

## Add Caching

Enable intelligent caching for instant re-runs:

```python
from hypernodes import SequentialEngine, DiskCache

engine = SequentialEngine(cache=DiskCache(path=".cache"))
pipeline = Pipeline(nodes=[clean_text, count_words], engine=engine)

# First run: executes
result1 = pipeline(passage="  Hello World  ")

# Second run: instant cache hit!
result2 = pipeline(passage="  Hello World  ")  # Cached!

# Different input: executes
result3 = pipeline(passage="  Goodbye  ")
```

## Async Execution

Speed up I/O-bound workloads with async execution:

```python
from hypernodes import HypernodesEngine

@node(output_name="data")
def fetch_api(url: str) -> dict:
    """Sync function - auto-wrapped for async execution"""
    import requests
    return requests.get(url).json()

@node(output_name="count")
def count_items(data: dict) -> int:
    return len(data.get("items", []))

# Async executor for I/O-bound work
pipeline = Pipeline(
    nodes=[fetch_api, count_items],
    engine=HypernodesEngine(map_executor="async")
)

# Process 10 URLs concurrently!
urls = [f"https://api.example.com/data/{i}" for i in range(10)]
results = pipeline.map(inputs={"url": urls}, map_over="url")
```

## Parallel Execution

Speed up CPU-bound workloads with parallel execution:

```python
from hypernodes import HypernodesEngine

@node(output_name="result")
def cpu_intensive(n: int) -> int:
    """CPU-bound computation"""
    def fib(x):
        return x if x < 2 else fib(x-1) + fib(x-2)
    return fib(n)

# Parallel executor for CPU-bound work
pipeline = Pipeline(
    nodes=[cpu_intensive],
    engine=HypernodesEngine(map_executor="parallel")
)

# Compute 4 fibonacci numbers in parallel across CPU cores
results = pipeline.map(
    inputs={"n": [35, 36, 37, 38]},
    map_over="n"
)
```

## Progress Tracking

Monitor execution with built-in progress bars:

```python
from hypernodes import SequentialEngine
from hypernodes.telemetry import ProgressCallback

engine = SequentialEngine(
    cache=DiskCache(path=".cache"),
    callbacks=[ProgressCallback()]
)
pipeline = Pipeline(nodes=[clean_text, count_words], engine=engine)

results = pipeline.map(
    inputs={"passage": ["text1", "text2", "text3", ...]},
    map_over="passage"
)

# Output:
# Processing Pipeline ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:12
#   ├─ clean_text ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:05
#   └─ count_words ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:07
```

## Nested Pipelines

Compose pipelines hierarchically:

```python
# Inner pipeline for text processing
@node(output_name="tokens")
def tokenize(cleaned_text: str) -> list:
    return cleaned_text.split()

text_pipeline = Pipeline(nodes=[clean_text, tokenize])

# Outer pipeline using nested pipeline
@node(output_name="data")
def load_data(file_path: str) -> str:
    with open(file_path) as f:
        return f.read()

@node(output_name="summary")
def summarize(tokens: list) -> dict:
    return {"token_count": len(tokens), "unique": len(set(tokens))}

main_pipeline = Pipeline(
    nodes=[load_data, text_pipeline, summarize]
)

result = main_pipeline(file_path="data.txt")
# {
#   'data': '...',
#   'cleaned_text': '...',
#   'tokens': [...],
#   'summary': {'token_count': 100, 'unique': 75}
# }
```

## What's Next?

- **[Core Concepts](../in-depth/core-concepts.md)** - Deep dive into nodes, pipelines, and dependencies
- **[Caching](../in-depth/caching.md)** - Learn how intelligent caching works
- **[Execution Engines](../advanced/execution-engines.md)** - Optimize performance with async, threaded, and parallel execution
- **[Nested Pipelines](../in-depth/nested-pipelines.md)** - Build complex workflows from simple pieces

## Common Patterns

### Pattern 1: ETL Pipeline

```python
@node(output_name="raw_data")
def extract(source: str) -> list:
    return load_from_source(source)

@node(output_name="clean_data")
def transform(raw_data: list) -> list:
    return [clean(item) for item in raw_data]

@node(output_name="success")
def load(clean_data: list, destination: str) -> bool:
    save_to_destination(clean_data, destination)
    return True

etl = Pipeline(nodes=[extract, transform, load])
result = etl(source="database", destination="warehouse")
```

### Pattern 2: ML Pipeline

```python
@node(output_name="features")
def extract_features(text: str) -> np.ndarray:
    return vectorizer.transform([text])

@node(output_name="prediction")
def predict(features: np.ndarray) -> str:
    return model.predict(features)[0]

@node(output_name="confidence")
def get_confidence(features: np.ndarray) -> float:
    return model.predict_proba(features).max()

ml_pipeline = Pipeline(nodes=[extract_features, predict, get_confidence])

# Batch predictions with caching
results = ml_pipeline.map(
    inputs={"text": texts},
    map_over="text"
)
```

### Pattern 3: Data Processing with Checkpoints

```python
@node(output_name="preprocessed", cache=True)
def expensive_preprocessing(data: list) -> list:
    """Cached - won't re-run"""
    return [expensive_operation(x) for x in data]

@node(output_name="result", cache=False)
def experiment(preprocessed: list, param: float) -> dict:
    """Not cached - re-runs every time"""
    return run_experiment(preprocessed, param)

engine = SequentialEngine(cache=DiskCache(path=".cache"))
pipeline = Pipeline(nodes=[expensive_preprocessing, experiment], engine=engine)

# Iterate on experiment without re-running preprocessing
for param in [0.1, 0.2, 0.3]:
    result = pipeline(data=my_data, param=param)
    # Only experiment() re-runs, preprocessing is cached!
```
