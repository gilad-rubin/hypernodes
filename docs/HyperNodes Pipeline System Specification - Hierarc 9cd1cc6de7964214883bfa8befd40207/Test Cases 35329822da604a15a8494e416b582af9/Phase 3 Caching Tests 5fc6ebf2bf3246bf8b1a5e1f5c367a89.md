# Phase 3: Caching Tests

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