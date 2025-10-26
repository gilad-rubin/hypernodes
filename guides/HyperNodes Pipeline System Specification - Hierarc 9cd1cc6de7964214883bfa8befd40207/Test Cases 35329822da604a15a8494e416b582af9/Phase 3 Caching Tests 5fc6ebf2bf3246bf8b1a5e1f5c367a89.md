# Overview

Verify caching functionality with computation signatures and selective re-execution.

---

## Test 3.1: Single Node Cache Hit

**Goal:** Verify basic caching functionality.

```python
from pipeline_system import Pipeline, DiskCache, LoggingCallback
import logging

logging.basicConfig(level=logging.DEBUG)

@node(output_name="result")
def add_one(x: int) -> int:
    print("add_one function body executed")
    return x + 1

# Configure pipeline with cache
pipeline = Pipeline(
    nodes=[add_one],
    cache=DiskCache(path=".cache"),
    callbacks=[LoggingCallback()]
)

# First run
print("=== First run ===")
result1 = [pipeline.run](http://pipeline.run)(x=5)

# Second run (should hit cache)
print("=== Second run ===")
result2 = [pipeline.run](http://pipeline.run)(x=5)

# Third run with different input
print("=== Third run with different input ===")
result3 = [pipeline.run](http://pipeline.run)(x=10)
```

**Expected output:**

```jsx
=== First run ===
Executing add_one
add_one function body executed
=== Second run ===
Cache hit for add_one (signature: abc12345...)
=== Third run with different input ===
Executing add_one
add_one function body executed
```

**Validates:**

- Cache configured via `cache=DiskCache(...)` parameter
- Cached results returned without re-execution (no "function body executed" log)
- Cache hit logs show which nodes were cached
- Computation signature works

---

## Test 3.2: Partial Cache Hit in Chain

**Goal:** Verify selective re-execution when some nodes cached.

```python
from pipeline_system import Pipeline, DiskCache, LoggingCallback
import logging

logging.basicConfig(level=logging.DEBUG)

@node(output_name="doubled")
def double(x: int) -> int:
    print("double function body executed")
    return x * 2

@node(output_name="result")
def add_one(doubled: int) -> int:
    print("add_one function body executed")
    return doubled + 1

# Configure pipeline with cache
pipeline = Pipeline(
    nodes=[double, add_one],
    cache=DiskCache(path=".cache"),
    callbacks=[LoggingCallback()]
)

# First run
print("=== First run ===")
result1 = [pipeline.run](http://pipeline.run)(x=5)

# Second run (full cache hit)
print("=== Second run ===")
result2 = [pipeline.run](http://pipeline.run)(x=5)

# Third run with different input (cache miss)
print("=== Third run with different input ===")
result3 = [pipeline.run](http://pipeline.run)(x=10)
```

**Expected output:**

```jsx
=== First run ===
Executing double
double function body executed
Executing add_one
add_one function body executed
=== Second run ===
Cache hit for double
Cache hit for add_one
=== Third run with different input ===
Executing double
double function body executed
Executing add_one
add_one function body executed
```

**Validates:**

- Only changed nodes re-execute
- Upstream cached results reused
- Cache invalidation works correctly
- Logging clearly shows which nodes executed vs cached

---

## Test 3.3: Map with Independent Item Caching

**Goal:** Verify each map item cached independently.

```python
from pipeline_system import Pipeline, DiskCache, LoggingCallback
import logging

logging.basicConfig(level=logging.DEBUG)

@node(output_name="result")
def add_one(x: int) -> int:
    print(f"add_one function body executed with x={x}")
    return x + 1

# Configure pipeline with cache
pipeline = Pipeline(
    nodes=[add_one],
    cache=DiskCache(path=".cache"),
    callbacks=[LoggingCallback()]
)

# First run with [1, 2, 3]
print("=== First run with [1, 2, 3] ===")
results1 = [pipeline.run](http://pipeline.run)(x=[1, 2, 3], map_over=["x"])

# Second run with same inputs (should hit cache for all)
print("=== Second run with [1, 2, 3] ===")
results2 = [pipeline.run](http://pipeline.run)(x=[1, 2, 3], map_over=["x"])

# Third run with [2, 3, 4] (cache hit for 2 and 3, miss for 4)
print("=== Third run with [2, 3, 4] ===")
results3 = [pipeline.run](http://pipeline.run)(x=[2, 3, 4], map_over=["x"])
```

**Expected output:**

```jsx
=== First run with [1, 2, 3] ===
Executing add_one with {'x': 1}
Executing add_one with {'x': 2}
Executing add_one with {'x': 3}
Executed 3 times, 0 cache hits
=== Second run with [1, 2, 3] ===
Cache hit for add_one
Cache hit for add_one
Cache hit for add_one
Executed 0 times, 3 cache hits
=== Third run with [2, 3, 4] ===
Cache hit for add_one
Cache hit for add_one
Executing add_one with {'x': 4}
Executed 1 times, 2 cache hits
```

**Validates:**

- Map items cached independently
- Cache hits across map calls
- Only new items re-execute
- Callback tracking provides clear execution vs cache metrics

---

## Test 3.5: Caching with Class Instances (Deterministic)

**Goal:** Verify caching works correctly when class instances are passed as inputs.

```python
from pipeline_system import Pipeline, DiskCache, LoggingCallback
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.DEBUG)

# Config class with explicit fields
@dataclass
class EncoderConfig:
    dim: int
    model_name: str
    
class DeterministicEncoder:
    def __init__(self, config: EncoderConfig):
        self.config = config
        self.dim = config.dim
        self._internal_cache = {}  # Private, excluded from hash
        
    def encode(self, text: str) -> list[float]:
        # Deterministic: same text -> same output
        return [float(ord(c)) for c in text[:self.dim]]

@node(output_name="embedding")
def encode_text(encoder: DeterministicEncoder, text: str) -> list[float]:
    print(f"encode_text executed with text='{text}'")
    return encoder.encode(text)

pipeline = Pipeline(
    nodes=[encode_text],
    cache=DiskCache(path=".cache"),
    callbacks=[LoggingCallback()]
)

# First run with encoder instance
config1 = EncoderConfig(dim=4, model_name="test-v1")
encoder1 = DeterministicEncoder(config1)

print("=== First run with encoder1 ===")
result1 = [pipeline.run](http://pipeline.run)(inputs={"encoder": encoder1, "text": "hello"})

# Second run with SAME config but NEW instance
config2 = EncoderConfig(dim=4, model_name="test-v1")
encoder2 = DeterministicEncoder(config2)

print("=== Second run with encoder2 (same config) ===")
result2 = [pipeline.run](http://pipeline.run)(inputs={"encoder": encoder2, "text": "hello"})

# Third run with DIFFERENT config
config3 = EncoderConfig(dim=4, model_name="test-v2")
encoder3 = DeterministicEncoder(config3)

print("=== Third run with encoder3 (different config) ===")
result3 = [pipeline.run](http://pipeline.run)(inputs={"encoder": encoder3, "text": "hello"})
```

**Expected output:**

```jsx
=== First run with encoder1 ===
Executing encode_text
encode_text executed with text='hello'

=== Second run with encoder2 (same config) ===
Cache hit for encode_text
(no execution log)

=== Third run with encoder3 (different config) ===
Executing encode_text
encode_text executed with text='hello'
```

**Validates:**

- Class instances with identical configuration produce cache hits
- Different class instances with same config share cache
- Private attributes (starting with `_`) are excluded from cache key
- Changing configuration invalidates cache

---

## Test 3.6: Caching with Custom `__cache_key__()` Method

**Goal:** Verify custom cache key implementation for complex objects.

```python
from pipeline_system import Pipeline, DiskCache, LoggingCallback
import logging
import json

logging.basicConfig(level=logging.DEBUG)

class ModelWithCustomKey:
    def __init__(self, model_name: str, temperature: float, api_key: str):
        self.model_name = model_name
        self.temperature = temperature
        self._api_key = api_key  # Secret, should not affect cache
        self._call_count = 0
        
    def __cache_key__(self) -> str:
        # Only model_name and temperature affect caching
        return f"{self.__class__.__name__}::{json.dumps({
            'model': self.model_name,
            'temp': self.temperature
        }, sort_keys=True)}"
        
    def generate(self, prompt: str) -> str:
        self._call_count += 1
        # Deterministic for testing
        return f"[{self.model_name}] {prompt.upper()}"

@node(output_name="result")
def generate_text(model: ModelWithCustomKey, prompt: str) -> str:
    print(f"generate_text executed with prompt='{prompt}'")
    return model.generate(prompt)

pipeline = Pipeline(
    nodes=[generate_text],
    cache=DiskCache(path=".cache"),
    callbacks=[LoggingCallback()]
)

# First run
model1 = ModelWithCustomKey("gpt-4", 0.7, "secret-key-123")
print("=== First run ===")
result1 = [pipeline.run](http://pipeline.run)(inputs={"model": model1, "prompt": "hello"})

# Second run: different API key, but same model config
model2 = ModelWithCustomKey("gpt-4", 0.7, "different-key-456")
print("=== Second run (different API key) ===")
result2 = [pipeline.run](http://pipeline.run)(inputs={"model": model2, "prompt": "hello"})

# Third run: different temperature
model3 = ModelWithCustomKey("gpt-4", 0.9, "secret-key-123")
print("=== Third run (different temperature) ===")
result3 = [pipeline.run](http://pipeline.run)(inputs={"model": model3, "prompt": "hello"})
```

**Expected output:**

```jsx
=== First run ===
Executing generate_text
generate_text executed with prompt='hello'

=== Second run (different API key) ===
Cache hit for generate_text
(no execution log)

=== Third run (different temperature) ===
Executing generate_text
generate_text executed with prompt='hello'
```

**Validates:**

- Custom `__cache_key__()` method controls what affects caching
- Private/secret data can be excluded from cache key
- Internal state changes (`_call_count`) don't invalidate cache
- Only explicitly included fields affect cache key

---

## Test 3.7: Caching with Nested Class Instances

**Goal:** Verify caching with complex nested object hierarchies.

```python
from pipeline_system import Pipeline, DiskCache, LoggingCallback
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.DEBUG)

@dataclass
class VectorConfig:
    dim: int
    normalize: bool

@dataclass  
class EncoderConfig:
    vector_config: VectorConfig
    model_version: str

class NestedEncoder:
    def __init__(self, config: EncoderConfig):
        self.config = config
        
    def encode(self, text: str) -> list[float]:
        result = [float(ord(c)) for c in text[:self.config.vector_config.dim]]
        if self.config.vector_config.normalize:
            total = sum(result) or 1.0
            result = [x / total for x in result]
        return result

@node(output_name="embedding")
def encode_with_nested(encoder: NestedEncoder, text: str) -> list[float]:
    print(f"encode_with_nested executed with text='{text}'")
    return encoder.encode(text)

pipeline = Pipeline(
    nodes=[encode_with_nested],
    cache=DiskCache(path=".cache"),
    callbacks=[LoggingCallback()]
)

# First run
vec_cfg1 = VectorConfig(dim=4, normalize=True)
enc_cfg1 = EncoderConfig(vector_config=vec_cfg1, model_version="v1")
encoder1 = NestedEncoder(enc_cfg1)

print("=== First run ===")
result1 = [pipeline.run](http://pipeline.run)(inputs={"encoder": encoder1, "text": "hello"})

# Second run: same nested config
vec_cfg2 = VectorConfig(dim=4, normalize=True)
enc_cfg2 = EncoderConfig(vector_config=vec_cfg2, model_version="v1")
encoder2 = NestedEncoder(enc_cfg2)

print("=== Second run (same nested config) ===")
result2 = [pipeline.run](http://pipeline.run)(inputs={"encoder": encoder2, "text": "hello"})

# Third run: change nested field
vec_cfg3 = VectorConfig(dim=4, normalize=False)  # Changed normalize
enc_cfg3 = EncoderConfig(vector_config=vec_cfg3, model_version="v1")
encoder3 = NestedEncoder(enc_cfg3)

print("=== Third run (changed nested field) ===")
result3 = [pipeline.run](http://pipeline.run)(inputs={"encoder": encoder3, "text": "hello"})
```

**Expected output:**

```jsx
=== First run ===
Executing encode_with_nested
encode_with_nested executed with text='hello'

=== Second run (same nested config) ===
Cache hit for encode_with_nested
(no execution log)

=== Third run (changed nested field) ===
Executing encode_with_nested
encode_with_nested executed with text='hello'
```

**Validates:**

- Nested dataclass attributes are recursively serialized
- Changes to deeply nested fields correctly invalidate cache
- Serialization depth configuration controls how deep to recurse
- Complex object hierarchies work correctly with caching

---

## Test 3.8: Non-Deterministic Classes Require Special Handling

**Goal:** Demonstrate that non-deterministic classes need explicit cache control.

```python
from pipeline_system import Pipeline, DiskCache, LoggingCallback
import logging
import random

logging.basicConfig(level=logging.DEBUG)

class NonDeterministicEncoder:
    """This encoder has non-deterministic behavior"""
    def __init__(self, dim: int):
        self.dim = dim
        self._rng = random.Random()  # Internal random state
        
    def encode(self, text: str) -> list[float]:
        # Non-deterministic: returns different values each time
        return [self._rng.random() for _ in range(self.dim)]

class DeterministicEncoder:
    """This encoder has deterministic behavior with fixed seed"""
    def __init__(self, dim: int, seed: int = 42):
        self.dim = dim
        self.seed = seed
        self._rng = random.Random(seed)  # Private, but seed is public
        
    def encode(self, text: str) -> list[float]:
        # Deterministic: same seed + text -> same output
        local_rng = random.Random(self.seed)
        return [local_rng.random() for _ in range(self.dim)]

@node(output_name="embedding", cache=False)  # Explicitly disable cache
def encode_non_deterministic(encoder: NonDeterministicEncoder, text: str) -> list[float]:
    print(f"encode_non_deterministic executed")
    return encoder.encode(text)

@node(output_name="embedding")  # Cache enabled
def encode_deterministic(encoder: DeterministicEncoder, text: str) -> list[float]:
    print(f"encode_deterministic executed")
    return encoder.encode(text)

pipeline_non_det = Pipeline(
    nodes=[encode_non_deterministic],
    cache=DiskCache(path=".cache"),
    callbacks=[LoggingCallback()]
)

pipeline_det = Pipeline(
    nodes=[encode_deterministic],
    cache=DiskCache(path=".cache"),
    callbacks=[LoggingCallback()]
)

# Test non-deterministic (always executes)
encoder_non_det = NonDeterministicEncoder(dim=4)
print("=== Non-deterministic: First run ===")
result1 = pipeline_non_[det.run](http://det.run)(inputs={"encoder": encoder_non_det, "text": "hello"})
print("=== Non-deterministic: Second run ===")
result2 = pipeline_non_[det.run](http://det.run)(inputs={"encoder": encoder_non_det, "text": "hello"})

# Test deterministic (caches correctly)
encoder_det1 = DeterministicEncoder(dim=4, seed=42)
print("=== Deterministic: First run ===")
result3 = pipeline_[det.run](http://det.run)(inputs={"encoder": encoder_det1, "text": "hello"})

encoder_det2 = DeterministicEncoder(dim=4, seed=42)  # Same seed
print("=== Deterministic: Second run (same seed) ===")
result4 = pipeline_[det.run](http://det.run)(inputs={"encoder": encoder_det2, "text": "hello"})
```

**Expected output:**

```jsx
=== Non-deterministic: First run ===
Executing encode_non_deterministic
encode_non_deterministic executed

=== Non-deterministic: Second run ===
Executing encode_non_deterministic
encode_non_deterministic executed
(always executes because cache=False)

=== Deterministic: First run ===
Executing encode_deterministic
encode_deterministic executed

=== Deterministic: Second run (same seed) ===
Cache hit for encode_deterministic
(no execution log)
```

**Validates:**

- Non-deterministic operations should use `cache=False` on the node
- Deterministic operations with seeds can be cached if seed is in cache key
- Private attributes (`_rng`) don't affect cache key
- Public configuration (dim, seed) correctly affects cache key

---

## Test 3.4: Nested Pipeline with Map and Caching

**Goal:** Verify caching works correctly with nested pipelines that use map operations.

```python
from pipeline_system import Pipeline, DiskCache, LoggingCallback
import logging

logging.basicConfig(level=logging.DEBUG)

# Inner pipeline nodes
@node(output_name="doubled")
def double(x: int) -> int:
    print(f"[inner] double function body executed with x={x}")
    return x * 2

@node(output_name="result")
def add_one(doubled: int) -> int:
    print(f"[inner] add_one function body executed")
    return doubled + 1

# Inner pipeline (will be mapped over items)
inner = Pipeline(
    nodes=[double, add_one],
    callbacks=[LoggingCallback(prefix="[inner]")]
)

# Outer pipeline node
@node(output_name="summary")
def summarize(results: list[int]) -> dict:
    print(f"[outer] summarize function body executed")
    return {"total": sum(results), "count": len(results)}

# Outer pipeline with cache
outer = Pipeline(
    nodes=[[inner.as](http://inner.as)_node(map_over=["x"]), summarize],
    cache=DiskCache(path=".cache"),
    callbacks=[LoggingCallback(prefix="[outer]")]
)

# First run with [1, 2, 3]
print("=== First run with [1, 2, 3] ===")
result1 = [outer.run](http://outer.run)(x=[1, 2, 3])
print(f"Inner: 6 executed, 0 cached")
print(f"Outer: 1 executed, 0 cached")

# Second run with same inputs (full cache hit)
print("=== Second run with [1, 2, 3] ===")
result2 = [outer.run](http://outer.run)(x=[1, 2, 3])
print(f"Inner: 0 executed, 6 cached")
print(f"Outer: 0 executed, 1 cached")

# Third run with [2, 3, 4] (partial cache hit)
print("=== Third run with [2, 3, 4] ===")
result3 = [outer.run](http://outer.run)(x=[2, 3, 4])
print(f"Inner: 2 executed, 4 cached")
print(f"Outer: 1 executed, 0 cached")
```

**Expected output:**

```jsx
=== First run with [1, 2, 3] ===
[inner] Executing double with x=1
[inner] Executing add_one with x=N/A
[inner] Executing double with x=2
[inner] Executing add_one with x=N/A
[inner] Executing double with x=3
[inner] Executing add_one with x=N/A
[outer] Executing summarize with x=N/A
Inner: 6 executed, 0 cached
Outer: 1 executed, 0 cached
=== Second run with [1, 2, 3] ===
[inner] Cache hit for double
[inner] Cache hit for add_one
[inner] Cache hit for double
[inner] Cache hit for add_one
[inner] Cache hit for double
[inner] Cache hit for add_one
[outer] Cache hit for summarize
Inner: 0 executed, 6 cached
Outer: 0 executed, 1 cached
=== Third run with [2, 3, 4] ===
[inner] Cache hit for double
[inner] Cache hit for add_one
[inner] Cache hit for double
[inner] Cache hit for add_one
[inner] Executing double with x=4
[inner] Executing add_one with x=N/A
[outer] Executing summarize with x=N/A
Inner: 2 executed, 4 cached
Outer: 1 executed, 0 cached
```

**Validates:**

- Nested pipelines with map operations cache correctly
- Each item in the map is cached independently within the inner pipeline
- Outer pipeline nodes (summarize) also benefit from caching
- Cache hits at both inner and outer pipeline levels work together
- Partial cache hits correctly re-execute only new items
- Callback tracking shows caching behavior at both nested levels