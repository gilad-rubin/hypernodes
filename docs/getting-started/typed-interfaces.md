# Typed Interfaces for IDE Autocomplete

HyperNodes provides TypedDict generation for pipeline inputs and outputs, giving you **IDE autocomplete** and **static type checking** without any runtime overhead.

## Why Use Typed Interfaces?

By default, pipeline inputs and outputs are dictionaries:

```python
# Default usage - no type hints
result = pipeline.run(inputs={"x": 5, "y": 10})
print(result["sum"])  # ⚠️ No autocomplete, typos possible
```

With typed interfaces, your IDE knows what keys exist:

```python
# With typed interfaces - IDE autocomplete! ✅
InputType = pipeline.get_input_type()
OutputType = pipeline.get_output_type()

inputs: InputType = {"x": 5, "y": 10}  # ✅ IDE suggests "x" and "y"
result: OutputType = pipeline.run(inputs=inputs)
print(result["sum"])  # ✅ IDE autocompletes "sum"
```

## Quick Start

### 1. Define Your Pipeline

```python
from hypernodes import Pipeline, node

@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

@node(output_name="sum")
def add(doubled: int, y: int) -> int:
    return doubled + y

pipeline = Pipeline(nodes=[double, add], name="MathPipeline")
```

### 2. Get Autocomplete Helpers

```python
# For inputs: Use constructor function (best IDE support)
make_input = pipeline.get_input_constructor()

# For outputs: Use TypedDict (works great for reading)
OutputType = pipeline.get_output_type()
```

### 3. Use Them

```python
# ✅ Input constructor provides autocomplete
inputs = make_input(
    x=5,    # ✅ IDE suggests "x"
    y=10    # ✅ IDE suggests "y"
)

# ✅ Output TypedDict provides autocomplete for reading
result: OutputType = pipeline.run(inputs=inputs)
print(result["sum"])      # ✅ IDE autocompletes "sum"
print(result["doubled"])  # ✅ IDE autocompletes "doubled"
```

> **Note on TypedDict Limitations**: TypedDict works great for **reading** dictionaries (outputs) but has limited IDE support for **constructing** dictionaries (inputs). That's why we provide `get_input_constructor()` for inputs and `get_output_type()` for outputs.

## What You Get

### IDE Autocomplete

When you type `inputs: InputType = {`, your IDE will suggest the available keys:

![IDE Autocomplete Example](../assets/typed-autocomplete.png)

### Static Type Checking

Use `mypy` or your IDE's type checker to catch errors before runtime:

```python
# ❌ mypy error: Expected int, got str
inputs: InputType = {"x": "hello", "y": 10}

# ❌ mypy error: Missing required key "y"
inputs: InputType = {"x": 5}

# ❌ mypy error: Key "wrong" not in OutputType
result: OutputType = pipeline.run(inputs=inputs)
print(result["wrong"])
```

### Runtime Validation

TypedDict provides **static checking only** (no runtime validation). The pipeline will still validate:

- Required inputs are provided (via `pipeline._validate_inputs()`)
- Node functions validate their own types when executed

If you need runtime validation, consider using Pydantic models instead.

## Advanced Usage

### Complex Types

TypedDict preserves complex type hints:

```python
from typing import List, Dict

@node(output_name="items")
def get_items(count: int) -> List[int]:
    return list(range(count))

@node(output_name="summary")
def summarize(items: List[int]) -> Dict[str, int]:
    return {"total": sum(items), "count": len(items)}

pipeline = Pipeline(nodes=[get_items, summarize])

InputType = pipeline.get_input_type()
OutputType = pipeline.get_output_type()

# InputType has: {'count': int}
# OutputType has: {'items': List[int], 'summary': Dict[str, int]}
```

### Multiple Outputs

For nodes with multiple outputs (tuple return types), TypedDict currently defaults to `Any`:

```python
@node(output_name=("mean", "std"))
def stats(data: List[float]) -> tuple[float, float]:
    mean = sum(data) / len(data)
    std = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
    return mean, std

# Both outputs will be typed as 'Any' for now
# TODO: Parse Tuple[type1, type2] annotations
```

### Unnamed Pipelines

If your pipeline doesn't have a name, default names are used:

```python
pipeline = Pipeline(nodes=[process])  # No name

InputType = pipeline.get_input_type()   # Name: "PipelineInput"
OutputType = pipeline.get_output_type() # Name: "PipelineOutput"
```

For better IDE experience, always name your pipelines:

```python
pipeline = Pipeline(nodes=[process], name="DataProcessor")
# Name: "DataProcessorInput", "DataProcessorOutput"
```

## Usage Patterns

### Pattern 1: Input Constructor + Output TypedDict (Recommended ⭐)

```python
make_input = pipeline.get_input_constructor()
OutputType = pipeline.get_output_type()

inputs = make_input(x=5, y=10)  # ✅ Autocomplete!
result: OutputType = pipeline.run(inputs=inputs)
print(result["sum"])  # ✅ Autocomplete!
```

### Pattern 2: Output TypedDict Only

```python
OutputType = pipeline.get_output_type()

result = pipeline.run(inputs={"x": 5, "y": 10})
typed_result: OutputType = result
print(typed_result["sum"])  # ✅ Autocomplete works
```

### Pattern 3: Input TypedDict (Limited IDE Support)

```python
InputType = pipeline.get_input_type()

# ⚠️ TypedDict doesn't provide autocomplete for dict literals
inputs: InputType = {"x": 5, "y": 10}  # No autocomplete when typing
result = pipeline.run(inputs=inputs)
```

> **Note**: Use `get_input_constructor()` instead of `get_input_type()` for better IDE support.

### Pattern 4: Typed Wrapper Function

Create a typed wrapper with the input constructor:

```python
make_input = pipeline.get_input_constructor()
OutputType = pipeline.get_output_type()

def run_pipeline(x: int, y: int) -> OutputType:
    """Typed wrapper around pipeline."""
    inputs = make_input(x=x, y=y)
    return pipeline.run(inputs=inputs)

# Now you get full autocomplete!
result = run_pipeline(x=5, y=10)
print(result["sum"])
```

## Best Practices

### 1. Always Add Type Hints to Your Nodes

```python
# ✅ Good - full type hints
@node(output_name="result")
def process(x: int, name: str) -> str:
    return f"{name}: {x}"

# ⚠️ Bad - no type hints (will default to 'Any')
@node(output_name="result")
def process(x, name):
    return f"{name}: {x}"
```

### 2. Name Your Pipelines

```python
# ✅ Good
pipeline = Pipeline(nodes=[...], name="DataProcessor")
# Generates: DataProcessorInput, DataProcessorOutput

# ⚠️ OK but less descriptive
pipeline = Pipeline(nodes=[...])
# Generates: PipelineInput, PipelineOutput
```

### 3. Generate Types Once

```python
# ✅ Generate at module level
pipeline = Pipeline(nodes=[...])
InputType = pipeline.get_input_type()
OutputType = pipeline.get_output_type()

# Use throughout your code
def func1():
    inputs: InputType = {...}
    
def func2():
    result: OutputType = pipeline.run(...)
```

### 4. Use with Nested Pipelines

```python
inner = Pipeline(nodes=[clean_text], name="Cleaner")
outer = Pipeline(nodes=[inner.as_node(), analyze], name="Analyzer")

# Get types for the outer pipeline
InputType = outer.get_input_type()
OutputType = outer.get_output_type()
```

## Why Two Different Approaches?

### TypedDict Limitation for Dict Construction

Python's TypedDict has a fundamental limitation: **IDEs don't autocomplete dict literals during construction**.

```python
# ❌ This doesn't work well in IDEs:
InputType = pipeline.get_input_type()
inputs: InputType = {"x": ...}  # IDE can't suggest "x" here
```

This is because when you type `{`, the IDE doesn't yet know you're building an `InputType`. TypedDict works great for **reading** existing dicts (like outputs) but not **constructing** them.

### Input Constructor Solution

The input constructor solves this by using **function parameters** instead of dict keys:

```python
# ✅ This works perfectly:
make_input = pipeline.get_input_constructor()
inputs = make_input(x=...)  # IDE suggests "x" as a parameter!
```

Function parameters have full IDE support, so you get autocomplete and type checking exactly where you need it.

### Summary

- **For INPUTS** (construction): Use `get_input_constructor()` ✅
- **For OUTPUTS** (reading): Use `get_output_type()` ✅
- **For INPUTS** (TypedDict): Limited IDE support ⚠️

## Limitations

### No Runtime Validation

TypedDict is for **static type checking only**. It doesn't validate at runtime:

```python
# This will NOT raise an error, even though "x" is the wrong type
inputs: InputType = {"x": "hello", "y": 10}  # Type checker warning only
pipeline.run(inputs=inputs)  # Runtime error occurs in node
```

For runtime validation, consider:
- Using Pydantic models externally
- Adding validation in your node functions
- Using Python's `dataclasses` with validation

### Limited Tuple Parsing

Multiple outputs currently default to `Any`:

```python
@node(output_name=("a", "b"))
def multi_output(x: int) -> tuple[int, str]:
    return x, str(x)

# Both 'a' and 'b' will be typed as 'Any'
```

## See Also

- [Core Concepts](../in-depth/core-concepts.md) - Understanding HyperNodes fundamentals
- [Quick Start](quick-start.md) - Getting started with HyperNodes
- [Pipeline Visualization](../in-depth/pipeline-visualization.md) - Visualizing your pipelines

