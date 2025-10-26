# Phase 5: Nested Pipelines Tests

# Overview

Verify nested pipeline functionality with proper output propagation and independent caching.

---

## Test 5.1: Simple Nested Pipeline

**Goal:** Verify pipeline used as node in another pipeline.

```python
@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

@node(output_name="incremented")
def add_one(doubled: int) -> int:
    return doubled + 1

inner_pipeline = Pipeline(nodes=[double, add_one])

@node(output_name="result")
def square(incremented: int) -> int:
    return incremented ** 2

outer_pipeline = Pipeline(nodes=[inner_pipeline, square])

result = outer_[pipeline.run](http://pipeline.run)(inputs={"x": 5})
assert result == {"doubled": 10, "incremented": 11, "result": 121}
```

**Validates:**

- Pipeline used as node
- Outputs from nested pipeline available to outer pipeline
- Dependencies resolved across pipeline boundaries

---

## Test 5.2: Nested Pipeline with Map

**Goal:** Verify nested pipeline in map operation.

```python
@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

@node(output_name="result")
def add_one(doubled: int) -> int:
    return doubled + 1

inner_pipeline = Pipeline(nodes=[double, add_one])

outer_pipeline = Pipeline(nodes=[inner_pipeline])

results = outer_[pipeline.map](http://pipeline.map)(inputs={"x": [1, 2, 3]}, map_over=["x"])
assert results == {"doubled": [2, 4, 6], "result": [3, 5, 7]}
```

**Validates:**

- Nested pipelines work in map operations
- Each item processed through full nested pipeline

---

## Test 5.3: Two-Level Nesting

**Goal:** Verify deeper nesting (3 levels total).

```python
@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

inner_inner = Pipeline(nodes=[double])

@node(output_name="incremented")
def add_one(doubled: int) -> int:
    return doubled + 1

inner = Pipeline(nodes=[inner_inner, add_one])

@node(output_name="result")
def square(incremented: int) -> int:
    return incremented ** 2

outer = Pipeline(nodes=[inner, square])

result = [outer.run](http://outer.run)(inputs={"x": 5})
assert result == {"doubled": 10, "incremented": 11, "result": 121}
```

**Validates:**

- Multiple levels of nesting work
- Outputs propagate up through levels

---

## Test 5.4: Nested Pipeline with Independent Caching

**Goal:** Verify each pipeline level has independent cache.

```python
from pipeline_system import CachingCallback
import logging

logging.basicConfig(level=[logging.INFO](http://logging.INFO))
logger = logging.getLogger(__name__)

class LoggingCachingCallback(CachingCallback):
    def __init__(self, name: str):
        super().__init__()
        [self.name](http://self.name) = name
        self.executions = []
        self.cache_hits = []
    
    def on_node_start(self, node_id: str, inputs: dict):
        self.executions.append(node_id)
        [logger.info](http://logger.info)(f"[{[self.name](http://self.name)}] Executing {node_id}")
    
    def on_cache_hit(self, node_id: str, signature: str):
        self.cache_hits.append(node_id)
        [logger.info](http://logger.info)(f"[{[self.name](http://self.name)}] Cache hit for {node_id}")

@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

@node(output_name="incremented")
def add_one(doubled: int) -> int:
    return doubled + 1

inner_callback = LoggingCachingCallback("inner")
inner = Pipeline(
    nodes=[double, add_one],
    callbacks=[inner_callback]
)

@node(output_name="result")
def square(incremented: int) -> int:
    return incremented ** 2

outer_callback = LoggingCachingCallback("outer")
outer = Pipeline(
    nodes=[inner, square],
    callbacks=[outer_callback]
)

# First run
[logger.info](http://logger.info)("=== First run ===")
inner_callback.executions = []
inner_callback.cache_hits = []
outer_callback.executions = []
outer_callback.cache_hits = []

result1 = [outer.run](http://outer.run)(inputs={"x": 5})
assert result1 == {"doubled": 10, "incremented": 11, "result": 121}
assert len(inner_callback.executions) == 2  # double, add_one
assert len(outer_callback.executions) == 1  # square
[logger.info](http://logger.info)(f"Inner: {len(inner_callback.executions)} executed, {len(inner_callback.cache_hits)} cached")
[logger.info](http://logger.info)(f"Outer: {len(outer_callback.executions)} executed, {len(outer_callback.cache_hits)} cached")

# Second run - all cached
[logger.info](http://logger.info)("=== Second run ===")
inner_callback.executions = []
inner_callback.cache_hits = []
outer_callback.executions = []
outer_callback.cache_hits = []

result2 = [outer.run](http://outer.run)(inputs={"x": 5})
assert result2 == {"doubled": 10, "incremented": 11, "result": 121}
assert len(inner_callback.executions) == 0
assert len(inner_callback.cache_hits) == 2  # double, add_one cached
assert len(outer_callback.executions) == 0
assert len(outer_callback.cache_hits) == 1  # square cached
[logger.info](http://logger.info)(f"Inner: {len(inner_callback.executions)} executed, {len(inner_callback.cache_hits)} cached")
[logger.info](http://logger.info)(f"Outer: {len(outer_callback.executions)} executed, {len(outer_callback.cache_hits)} cached")
```

**Expected output:**

```
=== First run ===
[inner] Executing double
[inner] Executing add_one
[outer] Executing square
Inner: 2 executed, 0 cached
Outer: 1 executed, 0 cached
=== Second run ===
[inner] Cache hit for double
[inner] Cache hit for add_one
[outer] Cache hit for square
Inner: 0 executed, 2 cached
Outer: 0 executed, 1 cached
```

**Validates:**

- Nested pipelines have independent caching
- Cache hits at all levels
- Callback tracking shows caching behavior at each level
- Callback tracking shows caching behavior at each level

---

## Test 5.5: Pipeline as Node with Input Renaming

**Goal:** Verify `.as_node()` with `input_mapping` to adapt parameter names.

```python
@node(output_name="cleaned")
def clean_text(passage: str) -> str:
    return passage.strip().lower()

inner = Pipeline(nodes=[clean_text])

# Outer pipeline uses "document" instead of "passage"
adapted = [inner.as](http://inner.as)_node(
    input_mapping={"document": "passage"}  # outer → inner
)

outer = Pipeline(nodes=[adapted])

result = [outer.run](http://outer.run)(inputs={"document": "  Hello World  "})
assert result["cleaned"] == "hello world"
```

**Validates:**

- Input mapping works correctly
- Direction is {outer: inner}
- Inner pipeline receives correctly renamed parameter

---

## Test 5.6: Pipeline as Node with Output Renaming

**Goal:** Verify `.as_node()` with `output_mapping` to rename outputs.

```python
@node(output_name="result")
def process(data: str) -> str:
    return data.upper()

inner = Pipeline(nodes=[process])

# Outer pipeline wants the output named "processed_data"
adapted = [inner.as](http://inner.as)_node(
    output_mapping={"result": "processed_data"}  # inner → outer
)

outer = Pipeline(nodes=[adapted])

result = [outer.run](http://outer.run)(inputs={"data": "hello"})
assert "processed_data" in result
assert "result" not in result  # Original name not visible
assert result["processed_data"] == "HELLO"
```

**Validates:**

- Output mapping works correctly
- Direction is {inner: outer}
- Original output name is hidden from outer pipeline

---

## Test 5.7: Pipeline as Node with Combined Renaming

**Goal:** Verify `.as_node()` with both input and output mapping.

```python
@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

inner = Pipeline(nodes=[double])

adapted = [inner.as](http://inner.as)_node(
    input_mapping={"value": "x"},
    output_mapping={"doubled": "result"}
)

outer = Pipeline(nodes=[adapted])

result = [outer.run](http://outer.run)(inputs={"value": 5})
assert result["result"] == 10
assert "doubled" not in result
```

**Validates:**

- Both mappings work simultaneously
- Inner and outer pipelines have completely different interfaces

---

## Test 5.8: Internal Mapping with Renaming (Encapsulated Map)

**Goal:** Verify `.as_node()` with `map_over` to encapsulate internal mapping.

```python
from typing import List, NamedTuple

class Item(NamedTuple):
    id: int
    value: str

@node(output_name="processed")
def process_item(item: Item) -> str:
    return f"{[item.id](http://item.id)}: {item.value.upper()}"

# Inner pipeline processes ONE item
single_process = Pipeline(nodes=[process_item])

# Adapt to process a LIST with renamed interface
batch_process = single_[process.as](http://process.as)_node(
    map_over="items",  # Outer provides "items" as list
    input_mapping={"items": "item"},  # Each list element becomes "item"
    output_mapping={"processed": "results"}  # Collect as "results"
)

outer = Pipeline(nodes=[batch_process])

items = [Item(id=1, value="hello"), Item(id=2, value="world")]
result = [outer.run](http://outer.run)(inputs={"items": items})

assert result["results"] == ["1: HELLO", "2: WORLD"]
assert "processed" not in result
```

**Validates:**

- `map_over` parameter works with renaming
- Inner pipeline executes once per item
- Outer pipeline sees list input → list output
- Mapping is completely encapsulated

---

## Test 5.9: Internal Mapping with Caching

**Goal:** Verify independent caching when using `.as_node()` with `map_over`.

```python
@node(output_name="doubled")
def double(x: int) -> int:
    print(f"Computing double({x})")
    return x * 2

single = Pipeline(nodes=[double])

batch = [single.as](http://single.as)_node(
    map_over="numbers",
    input_mapping={"numbers": "x"},
    output_mapping={"doubled": "results"}
)

outer = Pipeline(nodes=[batch])

# First run
result1 = [outer.run](http://outer.run)(inputs={"numbers": [1, 2, 3]})
assert result1["results"] == [2, 4, 6]
# Should print: "Computing double(1/2/3)" three times

# Second run with overlapping inputs
result2 = [outer.run](http://outer.run)(inputs={"numbers": [2, 3, 4]})
assert result2["results"] == [4, 6, 8]
# Should print: "Computing double(4)" only once (2 and 3 are cached)
```

**Validates:**

- Each item in the mapped execution is cached independently
- Cache hits work across different batch runs
- New items execute, cached items are retrieved

---

## Test 5.10: Namespace Collision Avoidance

**Goal:** Verify `output_mapping` prevents naming collisions between pipelines.

```python
@node(output_name="result")
def process_a(input: int) -> int:
    return input * 2

@node(output_name="result")
def process_b(input: int) -> int:
    return input * 3

pipeline_a = Pipeline(nodes=[process_a]).as_node(
    output_mapping={"result": "result_a"}
)

pipeline_b = Pipeline(nodes=[process_b]).as_node(
    output_mapping={"result": "result_b"}
)

@node(output_name="combined")
def combine(result_a: int, result_b: int) -> int:
    return result_a + result_b

outer = Pipeline(nodes=[pipeline_a, pipeline_b, combine])

result = [outer.run](http://outer.run)(inputs={"input": 5})
assert result["result_a"] == 10
assert result["result_b"] == 15
assert result["combined"] == 25
```

**Validates:**

- Multiple pipelines with same output name can coexist
- Output mapping creates separate namespaces
- Downstream functions can depend on renamed outputs

---

## Test 5.11: Complex Nested Mapping (Real-World Example)

**Goal:** Verify the encode corpus → build index pattern from documentation.

```python
from typing import List, NamedTuple, Sequence

class Passage(NamedTuple):
    pid: str
    text: str

class Vector(NamedTuple):
    values: List[float]

class EncodedPassage(NamedTuple):
    pid: str
    embedding: Vector

class Encoder:
    def encode(self, text: str) -> Vector:
        # Dummy encoder
        return Vector(values=[float(ord(c)) for c in text[:3]])

class Indexer:
    def index(self, passages: Sequence[EncodedPassage]) -> dict:
        return {"count": len(passages), "ids": [[p.pid](http://p.pid) for p in passages]}

@node(output_name="cleaned_text")
def clean_text(passage: Passage) -> str:
    return passage.text.strip().lower()

@node(output_name="embedding")
def encode_text(encoder: Encoder, cleaned_text: str) -> Vector:
    return encoder.encode(cleaned_text)

@node(output_name="encoded_passage")
def pack_encoded(passage: Passage, embedding: Vector) -> EncodedPassage:
    return EncodedPassage(pid=[passage.pid](http://passage.pid), embedding=embedding)

# Inner pipeline: processes ONE passage
single_encode = Pipeline(nodes=[clean_text, encode_text, pack_encoded])

# Adapt to process a CORPUS (list) with renamed interface
encode_corpus = single_[encode.as](http://encode.as)_node(
    map_over="corpus",
    input_mapping={"corpus": "passage"},
    output_mapping={"encoded_passage": "encoded_corpus"}
)

@node(output_name="index")
def build_index(indexer: Indexer, encoded_corpus: Sequence[EncodedPassage]) -> dict:
    return indexer.index(encoded_corpus)

# Outer pipeline: corpus → index
encode_and_index = Pipeline(nodes=[encode_corpus, build_index])

# Execute
corpus = [
    Passage(pid="p1", text="Hello World"),
    Passage(pid="p2", text="The Quick Brown Fox"),
]
encoder = Encoder()
indexer = Indexer()

outputs = encode_and_[index.run](http://index.run)(
    inputs={
        "corpus": corpus,
        "encoder": encoder,
        "indexer": indexer,
    }
)

index = outputs["index"]
assert index["count"] == 2
assert index["ids"] == ["p1", "p2"]
assert "encoded_passage" not in outputs  # Inner name hidden
assert "encoded_corpus" in outputs  # Outer name visible
```

**Validates:**

- Complete real-world pattern works end-to-end
- Input/output renaming with mapping
- Multiple pipeline levels
- Proper namespace isolation
- Downstream functions receive correctly named outputs
- Downstream functions receive correctly named outputs

---

## Test 5.12: Configuration Inheritance - Backend Only

**Goal:** Verify backend configuration inherits from parent when not specified.

```python
@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

@node(output_name="result")
def add_one(doubled: int) -> int:
    return doubled + 1

# Child has no backend specified
inner = Pipeline(nodes=[double, add_one])

# Parent defines backend
outer = Pipeline(
    nodes=[inner],
    backend=LocalBackend()
)

result = [outer.run](http://outer.run)(inputs={"x": 5})
assert result["result"] == 11

# Verify inner inherited backend from outer
assert inner.effective_backend == outer.backend
```

**Validates:**

- Nested pipeline inherits parent backend when not specified
- Execution works correctly with inherited backend

---

## Test 5.13: Configuration Inheritance - Selective Override

**Goal:** Verify selective override of backend while inheriting other configuration.

```python
from pipeline_system import DiskCache, ProgressCallback

@node(output_name="result")
def process(x: int) -> int:
    return x * 2

# Parent with full configuration
parent = Pipeline(
    nodes=[load_data, process_data, save_results],
    backend=LocalBackend(),
    cache=DiskCache(path="/tmp/cache"),
    callbacks=[ProgressCallback()]
)

# Child overrides only backend
child = Pipeline(
    nodes=[process],
    backend=ModalBackend(gpu="A100")  # Override
    # Should inherit: cache and callbacks from parent
)

# Verify inheritance
assert child.effective_backend == ModalBackend(gpu="A100")  # Overridden
assert child.effective_cache == parent.cache  # Inherited
assert child.effective_callbacks == parent.callbacks  # Inherited
```

**Validates:**

- Selective override of one configuration aspect
- Other aspects inherited from parent
- Override does not affect parent's configuration

---

## Test 5.14: Configuration Inheritance - Recursive Chain

**Goal:** Verify configuration inherits through multiple nesting levels.

```python
@node(output_name="result")
def process(x: int) -> int:
    return x * 2

# Level 1: Define all configuration
level_1 = Pipeline(
    nodes=[process],
    backend=LocalBackend(),
    cache=RedisCache(host="[localhost](http://localhost)"),
    timeout=300
)

# Level 2: Override backend only
level_2 = Pipeline(
    nodes=[process],
    backend=ModalBackend(gpu="A100")  # Override
    # Inherits: cache, timeout from level_1
)

# Level 3: Override cache only
level_3 = Pipeline(
    nodes=[process],
    cache=None  # Override: disable cache
    # Inherits: backend from level_2 (Modal), timeout from level_1
)

# Verify final configuration for level_3
assert level_3.effective_backend == ModalBackend(gpu="A100")  # From level_2
assert level_3.effective_cache is None  # Overridden at level_3
assert level_3.effective_timeout == 300  # From level_1
```

**Validates:**

- Configuration inherits through full chain
- Each level can override different aspects
- Overrides propagate down (level_2's backend override affects level_3)
- Original values from level_1 still accessible at level_3 if not overridden

---

## Test 5.15: Configuration Inheritance - Disable Caching

**Goal:** Verify explicit cache disabling overrides parent cache.

```python
from pipeline_system import DiskCache

@node(output_name="result")
def expensive_operation(x: int) -> int:
    return x ** 2

# Parent with caching enabled
outer = Pipeline(
    nodes=[load, inner, save],
    cache=DiskCache(path=".cache")
)

# Inner explicitly disables caching
inner = Pipeline(
    nodes=[expensive_operation],
    cache=None  # Explicit override: no caching
)

# First run
result1 = [outer.run](http://outer.run)(inputs={"x": 5})

# Second run - inner should re-execute (no cache)
result2 = [outer.run](http://outer.run)(inputs={"x": 5})

# Verify inner always executes (no cache hits)
assert inner.cache_hit_count == 0  # Never cached
```

**Validates:**

- `cache=None` explicitly disables caching
- Parent cache does not affect explicitly disabled child
- Child can opt out of parent's caching strategy

---

## Test 5.16: Configuration Inheritance - Callback Inheritance

**Goal:** Verify callback inheritance and override behavior.

```python
from pipeline_system import ProgressCallback, LoggingCallback, MetricsCallback

@node(output_name="result")
def process(x: int) -> int:
    return x * 2

# Parent with multiple callbacks
parent = Pipeline(
    nodes=[...],
    callbacks=[ProgressCallback(), LoggingCallback(), MetricsCallback()]
)

# Child inherits all callbacks (no override)
child = Pipeline(
    nodes=[process]
    # No callbacks specified → inherits all three from parent
)

# Grandchild overrides with different callbacks
grandchild = Pipeline(
    nodes=[process],
    callbacks=[TelemetryCallback()]  # Override: only telemetry
)

# Verify inheritance
assert len(child.effective_callbacks) == 3  # Inherited all three
assert len(grandchild.effective_callbacks) == 1  # Overridden with one
assert isinstance(grandchild.effective_callbacks[0], TelemetryCallback)
```

**Validates:**

- Callbacks inherit from parent when not specified
- Explicit callback list overrides parent completely
- Override affects only that level, not siblings

---

## Test 5.17: Configuration Inheritance - Full Inheritance

**Goal:** Verify complete inheritance when child specifies no configuration.

```python
@node(output_name="result")
def process(x: int) -> int:
    return x * 2

# Outer with full configuration
outer = Pipeline(
    nodes=[preprocess, inner, postprocess],
    backend=LocalBackend(),
    cache=DiskCache(path="/tmp/cache"),
    callbacks=[ProgressCallback()],
    timeout=60
)

# Inner has NO configuration
inner = Pipeline(
    nodes=[process]
    # No configuration at all
)

# Verify complete inheritance
assert inner.effective_backend == outer.backend
assert inner.effective_cache == outer.cache
assert inner.effective_callbacks == outer.callbacks
assert inner.effective_timeout == outer.timeout
```

**Validates:**

- Nested pipeline with no configuration inherits everything
- All configuration aspects inherited simultaneously
- Child behaves as if configuration was explicitly copied