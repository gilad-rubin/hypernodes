# Fluent API

The fluent API provides a clean, typed interface for running pipelines with **IDE autocomplete** and **type validation**.

## Why Use the Fluent API?

**Traditional approach (dict-based):**
```python
# ❌ No autocomplete, easy to make typos
result = pipeline.run(inputs={"x": 5, "y": 10})
results = pipeline.map(inputs={"x": [1,2,3], "y": 10}, map_over="x")
```

**Fluent API:**
```python
# ✅ IDE autocompletes parameter names!
result = pipeline.with_inputs(x=5, y=10).run()
results = pipeline.with_map_inputs(x=[1,2,3], y=10).map_over("x")
```

## Quick Start

### Single Execution

```python
from hypernodes import Pipeline, node

@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

@node(output_name="sum")
def add(doubled: int, y: int) -> int:
    return doubled + y

pipeline = Pipeline(nodes=[double, add])

# Execute once
result = pipeline.with_inputs(x=5, y=10).run()
# {'doubled': 10, 'sum': 20}
```

### Map Execution

```python
# Map over x (list), broadcast y (scalar)
results = pipeline.with_map_inputs(x=[1, 2, 3], y=10).map_over("x")
# [{'doubled': 2, 'sum': 12}, {'doubled': 4, 'sum': 14}, {'doubled': 6, 'sum': 16}]
```

## API Reference

### `pipeline.with_inputs(**inputs)`

Create a typed runner for **single execution** (scalars only).

**Parameters:**
- `**inputs`: Input values (must be scalars, lists not allowed)

**Returns:** `TypedRunner` for single execution

**Example:**
```python
result = pipeline.with_inputs(x=5, y=10).run()
```

**Type Validation:**
```python
# ❌ Raises TypeError: lists not allowed in with_inputs
pipeline.with_inputs(x=[1,2,3])

# ✅ OK: type IS List[int]
@node(output_name="total")
def sum_items(items: List[int]) -> int:
    return sum(items)

pipeline.with_inputs(items=[1,2,3]).run()  # OK!
```

### `pipeline.with_map_inputs(**inputs)`

Create a typed runner for **map execution** (allows lists and scalars).

**Parameters:**
- `**inputs**: Input values (scalars for broadcasting, lists for mapping)

**Returns:** `TypedRunner` for map execution

**Example:**
```python
results = pipeline.with_map_inputs(x=[1,2,3], y=10).map_over("x")
```

**Must call `.map_over()`:**
```python
# ❌ Raises RuntimeError
pipeline.with_map_inputs(x=[1,2,3]).run()

# ✅ Must call map_over()
pipeline.with_map_inputs(x=[1,2,3]).map_over("x")
```

### `.map_over(*params, mode="zip")`

Specify which parameters to map over and **execute immediately**.

**Parameters:**
- `*params`: Parameter names to map over (must be lists)
- `mode`: `"zip"` (parallel iteration) or `"product"` (all combinations)

**Returns:** `List[Dict[str, Any]]` - Results for each iteration

**Example:**
```python
# Map single parameter
results = pipeline.with_map_inputs(x=[1,2,3], y=10).map_over("x")

# Zip mode (parallel iteration)
results = pipeline.with_map_inputs(
    x=[1,2,3],
    y=[10,20,30]
).map_over("x", "y")  # Pairs: (1,10), (2,20), (3,30)

# Product mode (all combinations)
results = pipeline.with_map_inputs(
    x=[1,2],
    y=[10,20]
).map_over("x", "y", mode="product")  # (1,10), (1,20), (2,10), (2,20)
```

**Type Validation:**
```python
# ❌ Mapped param must be a list
pipeline.with_map_inputs(x=5, y=10).map_over("x")  # TypeError!

# ❌ Non-mapped params can't be lists
pipeline.with_map_inputs(x=[1,2,3], y=[10,20]).map_over("x")  # TypeError!

# ✅ Include y in map_over() or make it scalar
pipeline.with_map_inputs(x=[1,2,3], y=[10,20,30]).map_over("x", "y")  # OK
pipeline.with_map_inputs(x=[1,2,3], y=10).map_over("x")  # OK
```

### `.select(*output_names)`

Select specific outputs to compute (optimization).

**Parameters:**
- `*output_names`: Names of outputs to compute

**Returns:** `TypedRunner` for chaining

**Example:**
```python
# Single execution
result = pipeline.with_inputs(x=5).select("output1", "output2").run()

# Map execution
results = pipeline.with_map_inputs(x=[1,2,3]).select("output1").map_over("x")
```

### `.run()`

Execute the pipeline (single execution only).

**Returns:** `Dict[str, Any]` - Results dictionary

**Example:**
```python
result = pipeline.with_inputs(x=5, y=10).run()
```

**Not allowed on `with_map_inputs()`:**
```python
# ❌ Raises RuntimeError
pipeline.with_map_inputs(x=[1,2,3]).run()
```

## Usage Patterns

### Pattern 1: Simple Single Execution

```python
result = pipeline.with_inputs(x=5, y=10).run()
```

### Pattern 2: Single Execution with Output Selection

```python
result = pipeline.with_inputs(x=5, y=10).select("sum").run()
```

### Pattern 3: Map Over Single Parameter

```python
# x is mapped (list), y is broadcast (scalar)
results = pipeline.with_map_inputs(
    x=[1, 2, 3],
    y=10
).map_over("x")
```

### Pattern 4: Map Over Multiple Parameters (Zip)

```python
# Parallel iteration: (1,10), (2,20), (3,30)
results = pipeline.with_map_inputs(
    x=[1, 2, 3],
    y=[10, 20, 30]
).map_over("x", "y")
```

### Pattern 5: Map Over Multiple Parameters (Product)

```python
# All combinations: (1,10), (1,20), (2,10), (2,20)
results = pipeline.with_map_inputs(
    x=[1, 2],
    y=[10, 20]
).map_over("x", "y", mode="product")
```

### Pattern 6: Map with Output Selection

```python
results = pipeline.with_map_inputs(
    x=[1, 2, 3],
    y=10
).select("sum").map_over("x")
```

### Pattern 7: Chaining

```python
# Chain select() before map_over()
results = (
    pipeline
    .with_map_inputs(x=[1,2,3], y=10)
    .select("sum")
    .map_over("x")
)
```

## Type Safety

### Scalar vs List Validation

The fluent API enforces type safety:

```python
@node(output_name="doubled")
def double(x: int) -> int:  # Type is int, not List[int]
    return x * 2

pipeline = Pipeline(nodes=[double])

# ✅ OK: scalar for single execution
pipeline.with_inputs(x=5).run()

# ❌ TypeError: x is int, not List
pipeline.with_inputs(x=[1,2,3]).run()

# ✅ OK: list allowed in with_map_inputs
pipeline.with_map_inputs(x=[1,2,3]).map_over("x")
```

### When Parameter Type IS List

```python
@node(output_name="total")
def sum_items(items: List[int]) -> int:  # Type IS List[int]
    return sum(items)

pipeline = Pipeline(nodes=[sum_items])

# ✅ OK: type IS list, so list value is OK
pipeline.with_inputs(items=[1,2,3,4,5]).run()

# ✅ OK: each iteration gets a List[int]
pipeline.with_map_inputs(
    items=[[1,2], [3,4], [5,6]]  # List of lists
).map_over("items")
```

### Mapped vs Broadcast Parameters

```python
pipeline.with_map_inputs(x=[1,2,3], y=10).map_over("x")
# ✅ x is mapped (must be list)
# ✅ y is broadcast (must be scalar)

pipeline.with_map_inputs(x=[1,2,3], y=[10,20]).map_over("x")
# ❌ TypeError: y is a list but not in map_over()

pipeline.with_map_inputs(x=[1,2,3], y=[10,20,30]).map_over("x", "y")
# ✅ Both mapped (both must be lists)
```

## Comparison with Dict-Based API

| Feature | Dict-Based | Fluent API |
|---------|------------|------------|
| **IDE Autocomplete** | ❌ No | ✅ Yes |
| **Type Validation** | ⚠️ Runtime only | ✅ Runtime + IDE |
| **Readability** | ⚠️ Verbose | ✅ Clean |
| **Type Safety** | ❌ None | ✅ Enforced |
| **Map vs Run** | ⚠️ Manual | ✅ Separated |

**Dict-Based:**
```python
# No autocomplete, no validation
result = pipeline.run(inputs={"x": 5, "y": 10})
results = pipeline.map(
    inputs={"x": [1,2,3], "y": 10},
    map_over="x"
)
```

**Fluent API:**
```python
# ✅ Autocomplete, type validation
result = pipeline.with_inputs(x=5, y=10).run()
results = pipeline.with_map_inputs(x=[1,2,3], y=10).map_over("x")
```

## Error Messages

The fluent API provides clear, actionable error messages:

### List in `with_inputs()`
```python
pipeline.with_inputs(x=[1,2,3])
# TypeError: with_inputs() received a list for parameter 'x',
# but its type is <class 'int'>, not List.
# Use with_map_inputs() to map over lists instead.
```

### `.run()` on `with_map_inputs()`
```python
pipeline.with_map_inputs(x=[1,2,3]).run()
# RuntimeError: Cannot call .run() on with_map_inputs().
# Use .map_over() to execute: pipeline.with_map_inputs(...).map_over('x')
```

### Non-mapped param is list
```python
pipeline.with_map_inputs(x=[1,2,3], y=[10,20]).map_over("x")
# TypeError: Parameters ['y'] are lists but not in map_over().
# Either include them in map_over() or provide scalar values for broadcasting.
```

### Mapped param is not list
```python
pipeline.with_map_inputs(x=5, y=10).map_over("x")
# TypeError: Parameter 'x' specified in map_over() but value is not a list.
# Got: int
```

## Best Practices

### 1. Use Fluent API by Default

```python
# ✅ Recommended: fluent API
result = pipeline.with_inputs(x=5, y=10).run()

# ⚠️ Legacy: dict-based (still works)
result = pipeline.run(inputs={"x": 5, "y": 10})
```

### 2. Always Add Type Hints

```python
# ✅ Type hints enable validation
@node(output_name="result")
def process(x: int, y: str) -> float:
    return float(x) * len(y)

# ⚠️ No type hints = no validation
@node(output_name="result")
def process(x, y):  # No autocomplete!
    return float(x) * len(y)
```

### 3. Use `with_map_inputs()` for Clarity

```python
# ✅ Clear intent: this is a map operation
results = pipeline.with_map_inputs(x=[1,2,3], y=10).map_over("x")

# ⚠️ Less clear (still works)
results = pipeline.map(inputs={"x": [1,2,3], "y": 10}, map_over="x")
```

### 4. Chain for Readability

```python
# ✅ Readable chain
results = (
    pipeline
    .with_map_inputs(x=[1,2,3], y=10)
    .select("sum")
    .map_over("x")
)
```

## See Also

- [Typed Interfaces](typed-interfaces.md) - TypedDict generation for outputs
- [Core Concepts](../in-depth/core-concepts.md) - Understanding pipelines
- [Map Operations](../in-depth/map-operations.md) - Deep dive into mapping

