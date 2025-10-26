# Phase 1: Core Execution Tests

# Overview

Verify basic function decoration and single-node pipeline execution through progressively complex dependency patterns.

---

## Test 1.1: Single Node with Simple Inputs

**Goal:** Verify basic function decoration and single-node pipeline execution.

```python
from pipeline_system import node, Pipeline

# Note: LocalBackend with sequential execution is the default

@node(output_name="result")
def add_one(x: int) -> int:
    return x + 1

pipeline = Pipeline(nodes=[add_one])

result = 
```

**Validates:**

- `@node` decorator with `output_name` parameter works
- Pipeline construction with single function
- Default backend (LocalBackend with sequential execution) is used
- [`pipeline.run](http://pipeline.run)()` executes and returns correct output
- Output is returned as a dictionary with correct key

---

## Test 1.2: Two Sequential Nodes

**Goal:** Verify dependency resolution and sequential execution.

```python
@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

@node(output_name="result")
def add_one(doubled: int) -> int:
    return doubled + 1

pipeline = Pipeline(nodes=[double, add_one])

result = 
```

**Validates:**

- Dependency resolution based on parameter names
- Sequential execution order (double → add_one)
- Multiple outputs returned in result dictionary
- Intermediate results accessible

---

## Test 1.3: Three Nodes with Linear Dependencies

**Goal:** Verify execution order with longer chains.

```python
@node(output_name="doubled")
```

**Validates:**

- Longer dependency chains work correctly
- Each intermediate result is computed and available

---

## Test 1.4: Diamond Dependency Pattern

**Goal:** Verify parallel-capable execution with converging dependencies.

```python
@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

@node(output_name="tripled")
def triple(x: int) -> int:
    return x * 3

@node(output_name="result")
```

**Validates:**

- Multiple nodes can depend on same input
- Node with multiple dependencies waits for all
- Correct execution with diamond pattern

---

## Test 1.5: Multiple Independent Inputs

**Goal:** Verify handling of multiple input parameters.

```python
@node(output_name="sum")
def add(x: int, y: int) -> int:
    return x + y

@node(output_name="product")
def multiply(x: int, y: int) -> int:
    return x * y

@node(output_name="result")
```

**Validates:**

- Multiple input parameters work
- Multiple nodes can use same inputs
- All outputs computed correctly

---

## Test 1.6: Simple Nested Pipeline

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
```

**Validates:**

- Pipeline used as node
- Outputs from nested pipeline available to outer pipeline
- Dependencies resolved across pipeline boundaries

```python
@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

@node(output_name="incremented")
def add_one(doubled: int) -> int:
    return doubled + 1

@node(output_name="result")
def square(incremented: int) -> int:
    return incremented ** 2

pipeline = Pipeline(nodes=[double, add_one, square])

result = 
```