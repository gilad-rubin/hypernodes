# Caching

HyperNodes treats caching as a first-class citizen, not an afterthought. The caching system is content-addressed, meaning results are indexed by **what was computed**, not **when** or **where** it was computed.

## Quick Start

```python
from hypernodes import Pipeline, node, SequentialEngine, DiskCache

@node(output_name="cleaned")
def clean(text: str) -> str:
    return text.strip().lower()

@node(output_name="tokens")
def tokenize(cleaned: str) -> list:
    return cleaned.split()

# Enable caching at engine level
engine = SequentialEngine(cache=DiskCache(path=".cache"))
pipeline = Pipeline(nodes=[clean, tokenize], engine=engine)

# First run: executes all nodes
result1 = pipeline(text="  Hello World  ")

# Second run: instant cache hit!
result2 = pipeline(text="  Hello World  ")

# Different input: cache miss, executes
result3 = pipeline(text="  Goodbye  ")
```

## How It Works

### Computation Signatures

Every node execution has a unique **signature** computed as:

```
signature = hash(
    code_hash     # Function source code + closures
    + inputs_hash # Direct input values
    + deps_hash   # Signatures of upstream nodes (recursive!)
    + env_hash    # Environment (library versions, config)
)
```

**Key guarantee**: If the signature is the same, the output is guaranteed to be identical.

### Example: Signature Computation

```python
@node(output_name="a")
def make_a(x: int) -> int:
    return x + 1

@node(output_name="b")
def make_b(a: int) -> int:
    return a * 2
```

For `make_b`:
- **code_hash**: Hash of `make_b` function source
- **inputs_hash**: Hash of the value of `a` (direct input to make_b)
- **deps_hash**: Signature of `make_a` (which produced `a`)
- **env_hash**: Environment configuration

If you change `make_a`'s code, `make_b`'s signature changes too (via `deps_hash`), so it re-executes even though `make_b`'s code didn't change!

## Per-Item Caching with `.map()`

When you map over multiple items, **each item is cached independently**:

```python
engine = SequentialEngine(cache=DiskCache(path=".cache"))
pipeline = Pipeline(nodes=[clean, tokenize], engine=engine)

# First run: process 100 items
results1 = pipeline.map(
    inputs={"text": texts_100},  # 100 texts
    map_over="text"
)

# Add 50 new items
results2 = pipeline.map(
    inputs={"text": texts_150},  # 150 texts (100 old + 50 new)
    map_over="text"
)
# ✅ First 100 items: CACHED (instant)
# ❌ 50 new items: EXECUTE
```

This is why it's called **"Development-First Caching"** - you can iterate on new examples without re-running old ones.

## Fine-Grained Invalidation

The cache automatically invalidates when anything changes:

### Code Changes

```python
@node(output_name="processed")
def process(text: str) -> str:
    return text.upper()  # Version 1

engine = SequentialEngine(cache=DiskCache(path=".cache"))
pipeline = Pipeline(nodes=[process], engine=engine)

# First run
result1 = pipeline(text="hello")  # EXECUTE
# → {'processed': 'HELLO'}

# Second run (same code)
result2 = pipeline(text="hello")  # CACHED

# Now change the function
@node(output_name="processed")
def process(text: str) -> str:
    return text.upper() + "!"  # Version 2 (added !)

engine = SequentialEngine(cache=DiskCache(path=".cache"))
pipeline = Pipeline(nodes=[process], engine=engine)

# Run again - cache invalidated automatically!
result3 = pipeline(text="hello")  # EXECUTE (code changed)
# → {'processed': 'HELLO!'}
```

### Input Changes

```python
result1 = pipeline(text="hello")   # EXECUTE
result2 = pipeline(text="hello")   # CACHED
result3 = pipeline(text="goodbye") # EXECUTE (different input)
```

### Upstream Changes

```python
@node(output_name="a")
def make_a(x: int) -> int:
    return x + 1

@node(output_name="b")
def make_b(a: int) -> int:
    return a * 2

engine = SequentialEngine(cache=DiskCache(path=".cache"))
pipeline = Pipeline(nodes=[make_a, make_b], engine=engine)

# First run
result1 = pipeline(x=5)  # EXECUTE both

# Second run
result2 = pipeline(x=5)  # CACHED both

# Change make_a
@node(output_name="a")
def make_a(x: int) -> int:
    return x + 10  # Changed!

engine = SequentialEngine(cache=DiskCache(path=".cache"))
pipeline = Pipeline(nodes=[make_a, make_b], engine=engine)

# Run again
result3 = pipeline(x=5)
# ❌ make_a: EXECUTE (code changed)
# ❌ make_b: EXECUTE (upstream dependency changed)
```

## Selective Caching

Control caching at the node level:

### Disable Caching for Specific Nodes

```python
@node(output_name="timestamp", cache=False)
def get_timestamp() -> str:
    """Always execute - never cache"""
    import datetime
    return datetime.datetime.now().isoformat()

@node(output_name="data")
def fetch_data() -> dict:
    """This WILL be cached"""
    import requests
    return requests.get("https://api.example.com/data").json()

engine = SequentialEngine(cache=DiskCache(path=".cache"))
pipeline = Pipeline(nodes=[get_timestamp, fetch_data], engine=engine)

# Every run: timestamp re-executes, data cached
result = pipeline(inputs={})
```

## DiskCache Configuration

```python
from hypernodes import DiskCache

# Basic usage
cache = DiskCache(path=".cache")

# Custom path
cache = DiskCache(path="/tmp/my_pipeline_cache")

# Disable expiry (cache forever)
cache = DiskCache(path=".cache", expire=None)
```

**Storage format**: DiskCache uses pickle for serialization. Cached files are stored in:
```
.cache/
├── node_signatures/
│   ├── make_a_abc123.pkl
│   └── make_b_def456.pkl
└── ...
```

## Advanced: Custom Cache Keys

For objects that aren't easily hashable, implement `__cache_key__()`:

```python
class CustomModel:
    def __init__(self, weights):
        self.weights = weights

    def __cache_key__(self):
        """Custom cache key for this object"""
        import hashlib
        # Hash only the weights, ignore other attributes
        return hashlib.md5(str(self.weights).encode()).hexdigest()

@node(output_name="prediction")
def predict(model: CustomModel, input: str) -> float:
    return model.predict(input)

# Cache will use model.__cache_key__() to determine if model changed
```

## Caching Strategies

### Strategy 1: Development Iteration

```python
# Enable caching during development
engine = SequentialEngine(cache=DiskCache(path=".dev_cache"))
pipeline = Pipeline(
    nodes=[expensive_preprocessing, train_model, evaluate],
    engine=engine
)

# Iterate on evaluate() without re-running preprocessing
# Change evaluate() → only it re-executes
```

### Strategy 2: Production Incremental Processing

```python
# Process new data without reprocessing old data
engine = SequentialEngine(cache=DiskCache(path="/prod/cache"))
pipeline = Pipeline(
    nodes=[clean, extract_features, classify],
    engine=engine
)

# Day 1: 1000 items
results = pipeline.map(inputs={"data": items_1000}, map_over="data")

# Day 2: 500 new items
results = pipeline.map(inputs={"data": items_1500}, map_over="data")
# Only 500 new items execute!
```

### Strategy 3: Selective Invalidation

```python
# Cache everything except non-deterministic nodes
@node(output_name="random_seed", cache=False)
def generate_seed() -> int:
    import random
    return random.randint(0, 1000000)

@node(output_name="model")
def train(data: list, random_seed: int):
    """This IS cached (deterministic given seed)"""
    return train_model(data, seed=random_seed)
```

## Debugging Cache Behavior

Check if a result was cached:

```python
from hypernodes import Pipeline, SequentialEngine, DiskCache
from hypernodes.telemetry import ProgressCallback

# Use ProgressCallback to see cache hits
engine = SequentialEngine(
    cache=DiskCache(path=".cache"),
    callbacks=[ProgressCallback()]
)
pipeline = Pipeline(nodes=[...], engine=engine)

# Progress bar will show: "✓" for cache hit, "..." for execution
result = pipeline(inputs={})
```

## Limitations

1. **Picklability**: DiskCache uses pickle, so objects must be picklable
2. **Disk Space**: Cache can grow large - monitor `.cache/` directory size
3. **Cross-Machine**: Cache is local to the machine (not distributed)

## Best Practices

✅ **Do**:
- Enable caching during development for faster iteration
- Use per-item caching for incremental batch processing
- Disable caching for non-deterministic nodes (timestamps, random seeds)

❌ **Don't**:
- Cache nodes with side effects (writing to databases, sending emails)
- Assume cache is distributed (it's local to the machine)
- Forget to clear cache when making breaking changes

## See Also

- [Core Concepts](core-concepts.md) - Node and pipeline basics
- [Execution Engines](../advanced/execution-engines.md) - Parallel execution with caching
- [Callbacks](callbacks.md) - Monitor cache hits and misses
