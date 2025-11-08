# Core Concepts

HyperNodes is built on three core concepts that compose hierarchically:

1. **Functions → Nodes** (via `@node` decorator)
2. **Nodes → Pipelines** (DAG composition)
3. **Pipelines → Nodes** (hierarchical nesting)

## Functions → Nodes

Every node is a decorated function with an explicit output name:

```python
from hypernodes import node

@node(output_name="cleaned_text")
def clean(passage: str) -> str:
    return passage.strip().lower()

@node(output_name="word_count")
def count_words(cleaned_text: str) -> int:
    return len(cleaned_text.split())
```

### Implicit Dependencies

Dependencies are declared **implicitly** through parameter names:

```python
@node(output_name="a")
def make_a(x: int) -> int:
    return x + 1

@node(output_name="b")
def make_b(a: int) -> int:  # ← depends on make_a via parameter name
    return a * 2
```

When you use `cleaned_text` as a parameter name in `count_words`, HyperNodes knows it depends on the node that outputs `cleaned_text`.

### Why This Works

This design has several benefits:

- **Refactoring-friendly**: Rename a variable, and the dependency updates automatically
- **Self-documenting**: Just read the function signature to see dependencies
- **Type-checkable**: Standard Python type hints work naturally
- **No boilerplate**: No need to manually declare edges or connections

## Nodes → Pipelines

A Pipeline is a Directed Acyclic Graph (DAG) of nodes:

```python
from hypernodes import Pipeline

pipeline = Pipeline(nodes=[clean, count_words])
```

### Automatic DAG Construction

The pipeline automatically:

1. Builds a dependency graph by matching parameter names to output names
2. Computes topological ordering (nodes execute in the right order)
3. Detects circular dependencies and raises errors
4. Identifies root arguments (parameters with no matching outputs)

### Execution Modes

```python
# Single execution - two equivalent syntaxes
result = pipeline(passage="Hello World")
result = pipeline.run(inputs={"passage": "Hello World"})
# → {'cleaned_text': 'hello world', 'word_count': 2}

# Batch execution - map over multiple inputs
results = pipeline.map(
    inputs={"passage": ["Hello", "World", "Foo"]},
    map_over="passage"
)
# → [
#      {'cleaned_text': 'hello', 'word_count': 1},
#      {'cleaned_text': 'world', 'word_count': 1},
#      {'cleaned_text': 'foo', 'word_count': 1}
#    ]
```

### Selective Execution

You can request only specific outputs:

```python
# Only compute what's needed for word_count
result = pipeline.run(
    inputs={"passage": "Hello World"},
    output_name="word_count"
)
# → {'word_count': 2}
```

The pipeline will only execute nodes in the dependency path to `word_count`.

## Pipelines → Nodes

Pipelines can contain other pipelines, enabling hierarchical composition:

```python
# Inner pipeline
text_pipeline = Pipeline(nodes=[clean, tokenize, normalize])

# Outer pipeline uses inner pipeline as a node
main_pipeline = Pipeline(
    nodes=[load_data, text_pipeline, train_model]
)
```

### Interface Adaptation with `.as_node()`

Use `.as_node()` to adapt a pipeline's interface when nesting:

```python
@node(output_name="embedding")
def encode(text: str) -> list:
    return model.encode(text)

# Pipeline that processes ONE text
single_encoder = Pipeline(nodes=[clean, encode])

# Adapt to process MANY texts
batch_encoder = single_encoder.as_node(
    input_mapping={"texts": "text"},          # outer → inner
    output_mapping={"embedding": "embeddings"},  # inner → outer
    map_over="texts"  # Map over this parameter
)

# Use in outer pipeline
index_pipeline = Pipeline(nodes=[batch_encoder, build_index])

# From outer perspective: processes list of texts
result = index_pipeline.run(inputs={"texts": ["Hello", "World", "Foo"]})
# → {'embeddings': [[0.1, ...], [0.2, ...], [0.3, ...]], 'index': {...}}
```

## Dependency Resolution

Let's walk through how HyperNodes resolves dependencies:

```python
@node(output_name="a")
def make_a(x: int) -> int:
    return x + 1

@node(output_name="b")
def make_b(y: int) -> int:
    return y * 2

@node(output_name="c")
def make_c(a: int, b: int) -> int:
    return a + b

pipeline = Pipeline(nodes=[make_a, make_b, make_c])
```

**Step 1: Extract metadata**
- `make_a`: parameters=`(x,)`, output=`a`
- `make_b`: parameters=`(y,)`, output=`b`
- `make_c`: parameters=`(a, b)`, output=`c`

**Step 2: Build dependency graph**
- `make_a` depends on nothing (root)
- `make_b` depends on nothing (root)
- `make_c` depends on `a` and `b`

**Step 3: Topological sort**
- Execution order: `[make_a, make_b, make_c]` (or `[make_b, make_a, make_c]`)

**Step 4: Identify root arguments**
- Root args: `x`, `y` (no nodes produce these)

**Step 5: Execute**
```python
result = pipeline.run(inputs={"x": 5, "y": 3})
# make_a(x=5) → a=6
# make_b(y=3) → b=6
# make_c(a=6, b=6) → c=12
# → {'a': 6, 'b': 6, 'c': 12}
```

## Advanced: Multiple Outputs

A node can return multiple outputs using tuples:

```python
@node(output_name=("min_val", "max_val"))
def compute_range(numbers: list) -> tuple:
    return (min(numbers), max(numbers))

@node(output_name="span")
def compute_span(min_val: int, max_val: int) -> int:
    return max_val - min_val

pipeline = Pipeline(nodes=[compute_range, compute_span])

result = pipeline.run(inputs={"numbers": [1, 5, 3, 9, 2]})
# → {'min_val': 1, 'max_val': 9, 'span': 8}
```

## Advanced: Caching Control

Control caching at the node level:

```python
@node(output_name="timestamp", cache=False)
def get_timestamp() -> str:
    """Never cache - always return current time"""
    import datetime
    return datetime.datetime.now().isoformat()

@node(output_name="processed")
def process(data: str, timestamp: str) -> dict:
    return {"data": data, "timestamp": timestamp}
```

See [Caching](caching.md) for details.

## See Also

- [Caching](caching.md) - Intelligent caching system
- [Nested Pipelines](nested-pipelines.md) - Hierarchical composition patterns
- [Execution Engines](../advanced/execution-engines.md) - Parallel and async execution
