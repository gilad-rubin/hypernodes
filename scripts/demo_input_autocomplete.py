"""Demo showing different approaches to input autocomplete."""

from hypernodes import Pipeline, node


@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2


@node(output_name="sum")
def add(doubled: int, y: int) -> int:
    return doubled + y


pipeline = Pipeline(nodes=[double, add], name="Math")

print("=" * 60)
print("INPUT AUTOCOMPLETE SOLUTIONS")
print("=" * 60)
print()

# ========== PROBLEM: TypedDict doesn't help with construction ==========
print("❌ PROBLEM: TypedDict has limited IDE support for dict literals")
print("-" * 60)
InputType = pipeline.get_input_type()

# This does NOT give autocomplete in most IDEs:
inputs: InputType = {"x": 5, "y": 10}  # No suggestions when typing
print(f"inputs = {inputs}")
print("  Issue: IDE doesn't autocomplete dict literal construction")
print()

# ========== SOLUTION 1: Input Constructor (RECOMMENDED) ==========
print("✅ SOLUTION 1: Use Input Constructor (Recommended)")
print("-" * 60)
make_input = pipeline.get_input_constructor()

# This DOES give autocomplete!
inputs = make_input(
    x=5,  # ✅ IDE suggests 'x'
    y=10,  # ✅ IDE suggests 'y'
)
print(f"make_input(x=5, y=10) = {inputs}")
print("  ✅ IDE autocompletes parameter names")
print("  ✅ Type checker validates types")
print("  ✅ Returns dict compatible with pipeline.run()")

result = pipeline.run(inputs=inputs)
print(f"  Result: {result}")
print()

# ========== SOLUTION 2: Output TypedDict (This works!) ==========
print("✅ OUTPUT TypedDict (This works great!)")
print("-" * 60)
OutputType = pipeline.get_output_type()

result = pipeline.run(inputs={"x": 3, "y": 7})
typed_result: OutputType = result

# This DOES give autocomplete because we're READING, not WRITING
print(f"result['sum'] = {typed_result['sum']}")  # ✅ Autocompletes 'sum'
print(f"result['doubled'] = {typed_result['doubled']}")  # ✅ Autocompletes 'doubled'
print("  ✅ IDE autocompletes keys when reading")
print()

# ========== COMPARISON ==========
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print()
print("For INPUTS (construction):")
print("  ✅ Use: pipeline.get_input_constructor()")
print("  ❌ Avoid: InputType TypedDict (limited IDE support)")
print()
print("For OUTPUTS (reading):")
print("  ✅ Use: pipeline.get_output_type()")
print("  ✅ Works great for autocomplete when reading results")
print()
print("Full Example:")
print("-" * 60)
print("make_input = pipeline.get_input_constructor()")
print("OutputType = pipeline.get_output_type()")
print()
print("inputs = make_input(x=5, y=10)  # ✅ Autocomplete!")
print("result: OutputType = pipeline.run(inputs=inputs)")
print("print(result['sum'])  # ✅ Autocomplete!")
