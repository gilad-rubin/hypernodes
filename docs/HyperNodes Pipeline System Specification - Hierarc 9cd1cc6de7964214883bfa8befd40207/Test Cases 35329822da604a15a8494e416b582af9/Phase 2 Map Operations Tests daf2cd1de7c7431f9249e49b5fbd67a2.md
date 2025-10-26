# Phase 2: Map Operations Tests

# Overview

Verify map operations over collections with various parameter configurations and execution patterns.

---

## Test 2.1: Map Over Single Parameter

**Goal:** Verify [`pipeline.map`](http://pipeline.map)`()` with single parameter.

```python
@node(output_name="result")
def add_one(x: int) -> int:
    return x + 1

pipeline = Pipeline(nodes=[add_one])

results = [pipeline.map](http://pipeline.map)(inputs={"x": [1, 2, 3]}, map_over="x")
assert results == {"result": [2, 3, 4]}
```

**Validates:**

- [`pipeline.map`](http://pipeline.map)`()` executes over collection
- `inputs` is a dictionary with list values for varying parameters
- `map_over` accepts single string for single parameter (also accepts `["x"]`)
- Results returned as lists
- Order preserved

---

## Test 2.2: Map Over Two Sequential Nodes

**Goal:** Verify map with dependency chain.

```python
@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

@node(output_name="result")
def add_one(doubled: int) -> int:
    return doubled + 1

pipeline = Pipeline(nodes=[double, add_one])

results = [pipeline.map](http://pipeline.map)(inputs={"x": [1, 2, 3]}, map_over="x")
assert results == {"doubled": [2, 4, 6], "result": [3, 5, 7]}
```

**Validates:**

- Map works with multi-node pipelines
- All intermediate results returned as lists
- Dependencies resolved per item

---

## Test 2.3: Map with Diamond Pattern

**Goal:** Verify map with parallel-capable pattern.

```python
@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

@node(output_name="tripled")
def triple(x: int) -> int:
    return x * 3

@node(output_name="result")
def add(doubled: int, tripled: int) -> int:
    return doubled + tripled

pipeline = Pipeline(nodes=[double, triple, add])

results = [pipeline.map](http://pipeline.map)(inputs={"x": [1, 2, 3]}, map_over="x")
assert results == {"doubled": [2, 4, 6], "tripled": [3, 6, 9], "result": [5, 10, 15]}
```

**Validates:**

- Complex DAG patterns work with map
- Each item executed independently

---

## Test 2.4: Map with Fixed and Varying Parameters

**Goal:** Verify map with some parameters fixed and others varying.

```python
@node(output_name="result")
def multiply(x: int, factor: int) -> int:
    return x * factor

pipeline = Pipeline(nodes=[multiply])

results = [pipeline.map](http://pipeline.map)(inputs={"x": [1, 2, 3], "factor": 10}, map_over="x")
assert results == {"result": [10, 20, 30]}
```

**Validates:**

- Fixed parameters used for all items
- Only parameters in `map_over` vary
- Correct behavior with mixed varying/fixed inputs

---

## Test 2.5: Empty Collection

**Goal:** Verify map handles empty input gracefully.

```python
@node(output_name="result")
def add_one(x: int) -> int:
    return x + 1

pipeline = Pipeline(nodes=[add_one])

results = [pipeline.map](http://pipeline.map)(inputs={"x": []}, map_over="x")
assert results == {"result": []}
```

**Validates:**

- Empty collection returns empty results
- No errors raised

---

## Test 2.6: Map with Multiple Parameters (Zip Mode - Default)

**Goal:** Verify map with multiple `map_over` parameters using zip mode (default).

```python
@node(output_name="result")
def process(id: int, text: str) -> str:
    return f"{id}: {text.upper()}"

pipeline = Pipeline(nodes=[process])

results = [pipeline.map](http://pipeline.map)(
    inputs={"id": [1, 2, 3], "text": ["hello", "world", "test"]},
    map_over=["id", "text"]
)
assert results == {"result": ["1: HELLO", "2: WORLD", "3: TEST"]}
```

**Validates:**

- Multiple `map_over` parameters work
- Zip mode processes corresponding items together
- Default `map_mode` is "zip"
- Order preserved

---

## Test 2.7: Map with Zip Mode - Mismatched Lengths Error

**Goal:** Verify that zip mode raises error when list lengths don't match.

```python
@node(output_name="result")
def process(id: int, text: str) -> str:
    return f"{id}: {text.upper()}"

pipeline = Pipeline(nodes=[process])

# Should raise error: lists have different lengths
try:
    results = [pipeline.map](http://pipeline.map)(
        inputs={"id": [1, 2, 3], "text": ["hello", "world"]},  # Mismatched: 3 vs 2
        map_over=["id", "text"]
    )
    assert False, "Should have raised error"
except ValueError as e:
    assert "length" in str(e).lower()
```

**Validates:**

- Zip mode validates list lengths
- Clear error message when lengths don't match
- Prevents silent bugs from length mismatches

---

## Test 2.8: Map with Multiple Parameters (Product Mode)

**Goal:** Verify map with multiple `map_over` parameters using product mode.

```python
@node(output_name="result")
def multiply(x: int, y: int) -> int:
    return x * y

pipeline = Pipeline(nodes=[multiply])

results = [pipeline.map](http://pipeline.map)(
    inputs={"x": [1, 2], "y": [10, 20, 30]},
    map_over=["x", "y"],
    map_mode="product"
)
assert results == {"result": [10, 20, 30, 20, 40, 60]}  # (1,10), (1,20), (1,30), (2,10), (2,20), (2,30)
```

**Validates:**

- Product mode creates all combinations
- Lists can be of different lengths
- Correct number of executions (6 = 2 × 3)
- Results in correct order

---

## Test 2.9: Map with Zip Mode and Fixed Parameter

**Goal:** Verify zip mode with some parameters fixed and others varying.

```python
@node(output_name="result")
def format_message(id: int, text: str, prefix: str) -> str:
    return f"{prefix}{id}: {text}"

pipeline = Pipeline(nodes=[format_message])

results = [pipeline.map](http://pipeline.map)(
    inputs={"id": [1, 2], "text": ["hello", "world"], "prefix": "MSG-"},
    map_over=["id", "text"]
    map_mode="zip" #default
)
assert results == {"result": ["MSG-1: hello", "MSG-2: world"]}
```

**Validates:**

- Zip mode with subset of parameters varying
- Fixed parameters used in all executions
- Correct behavior with mixed varying/fixed inputs