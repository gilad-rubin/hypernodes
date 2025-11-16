"""Demo of typed interface for Pipeline inputs and outputs."""

from hypernodes import Pipeline, node


@node(output_name="doubled")
def double(x: int) -> int:
    """Double the input."""
    return x * 2


@node(output_name="sum")
def add(doubled: int, y: int) -> int:
    """Add doubled value to y."""
    return doubled + y


# Create pipeline
pipeline = Pipeline(nodes=[double, add], name="MathPipeline")

# ===== GENERATE TYPED INTERFACES =====
InputType = pipeline.get_input_type()
OutputType = pipeline.get_output_type()

print("Generated Types:")
print(f"  InputType: {InputType.__name__}")
print(f"    Fields: {list(InputType.__annotations__.keys())}")
print(f"    Types: {InputType.__annotations__}")
print()
print(f"  OutputType: {OutputType.__name__}")
print(f"    Fields: {list(OutputType.__annotations__.keys())}")
print(f"    Types: {OutputType.__annotations__}")
print()

# ===== USAGE PATTERN 1: Type your input dict =====
print("Pattern 1: Type the input dict")
inputs: InputType = {"x": 5, "y": 10}
result = pipeline.run(inputs=inputs)
print(f"  Result: {result}")
print()

# ===== USAGE PATTERN 2: Type the output =====
print("Pattern 2: Type the output")
result_raw = pipeline.run(inputs={"x": 3, "y": 7})
typed_result: OutputType = result_raw
print(f"  typed_result['doubled'] = {typed_result['doubled']}")
print(f"  typed_result['sum'] = {typed_result['sum']}")
print()

# ===== USAGE PATTERN 3: Both typed =====
print("Pattern 3: Both input and output typed")
my_inputs: InputType = {"x": 10, "y": 20}
my_result: OutputType = pipeline.run(inputs=my_inputs)
print(f"  Result: {my_result}")
print()

# ===== WHAT YOU GET IN YOUR IDE =====
print("IDE Benefits:")
print("  ✅ When typing 'inputs: InputType = {', IDE suggests 'x' and 'y'")
print("  ✅ When typing 'result[', IDE suggests 'doubled' and 'sum'")
print("  ✅ mypy catches typos: result['summ'] -> error!")
print("  ✅ mypy catches wrong types: {'x': 'hello'} -> error!")

