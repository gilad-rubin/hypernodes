"""Demo of the fluent API with with_inputs and with_map_inputs."""

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

print("=" * 70)
print("FLUENT API DEMO")
print("=" * 70)
print()

# ========== SINGLE EXECUTION ==========
print("1. SINGLE EXECUTION with_inputs().run()")
print("-" * 70)
result = pipeline.with_inputs(x=5, y=10).run()
print(f"   pipeline.with_inputs(x=5, y=10).run()")
print(f"   Result: {result}")
print(f"   ✅ IDE autocompletes parameter names: x, y")
print()

# ========== SINGLE EXECUTION WITH OUTPUT SELECTION ==========
print("2. SINGLE EXECUTION with output selection")
print("-" * 70)
result = pipeline.with_inputs(x=5, y=10).select("sum").run()
print(f"   pipeline.with_inputs(x=5, y=10).select('sum').run()")
print(f"   Result: {result}")
print(f"   ✅ Only 'sum' is computed (doubled is skipped)")
print()

# ========== MAP EXECUTION (single parameter) ==========
print("3. MAP EXECUTION over single parameter")
print("-" * 70)
results = pipeline.with_map_inputs(x=[1, 2, 3], y=10).map(map_over="x")
print(f"   pipeline.with_map_inputs(x=[1,2,3], y=10).map_over('x')")
print(f"   Results:")
for i, result in enumerate(results):
    print(f"     [{i}]: {result}")
print(f"   ✅ x is mapped, y is broadcast (10 for all)")
print()

# ========== MAP EXECUTION (zip mode) ==========
print("4. MAP EXECUTION over multiple parameters (zip mode)")
print("-" * 70)
results = pipeline.with_map_inputs(x=[1, 2, 3], y=[10, 20, 30]).map_over("x", "y")
print(f"   pipeline.with_map_inputs(x=[1,2,3], y=[10,20,30]).map_over('x', 'y')")
print(f"   Results:")
for i, result in enumerate(results):
    print(f"     [{i}]: {result}")
print(f"   ✅ Zip mode: pairs (1,10), (2,20), (3,30)")
print()

# ========== MAP EXECUTION (product mode) ==========
print("5. MAP EXECUTION over multiple parameters (product mode)")
print("-" * 70)
results = pipeline.with_map_inputs(x=[2, 3], y=[10, 100]).map_over(
    "x", "y", mode="product"
)
print(f"   pipeline.with_map_inputs(x=[2,3], y=[10,100]).map_over('x', 'y', mode='product')")
print(f"   Results:")
for i, result in enumerate(results):
    print(f"     [{i}]: {result}")
print(f"   ✅ Product mode: all combinations (2,10), (2,100), (3,10), (3,100)")
print()

# ========== MAP EXECUTION with output selection ==========
print("6. MAP EXECUTION with output selection")
print("-" * 70)
results = pipeline.with_map_inputs(x=[1, 2, 3], y=10).select("sum").map_over("x")
print(f"   pipeline.with_map_inputs(x=[1,2,3], y=10).select('sum').map_over('x')")
print(f"   Results:")
for i, result in enumerate(results):
    print(f"     [{i}]: {result}")
print(f"   ✅ Only 'sum' is computed for each iteration")
print()

# ========== ERROR CASES ==========
print("7. ERROR CASES (validation)")
print("-" * 70)

# Error: list in with_inputs
print("   ❌ pipeline.with_inputs(x=[1,2,3])")
try:
    pipeline.with_inputs(x=[1, 2, 3])
except TypeError as e:
    print(f"   TypeError: {e}")
print()

# Error: calling .run() on with_map_inputs
print("   ❌ pipeline.with_map_inputs(x=[1,2,3]).run()")
try:
    pipeline.with_map_inputs(x=[1, 2, 3]).run()
except RuntimeError as e:
    print(f"   RuntimeError: {e}")
print()

# Error: non-mapped param is a list
print("   ❌ pipeline.with_map_inputs(x=[1,2,3], y=[10,20]).map_over('x')")
try:
    pipeline.with_map_inputs(x=[1, 2, 3], y=[10, 20]).map_over("x")
except TypeError as e:
    print(f"   TypeError: {e}")
print()

# ========== SUMMARY ==========
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("Single Execution:")
print("  pipeline.with_inputs(x=5, y=10).run()")
print("  pipeline.with_inputs(x=5).select('output1').run()")
print()
print("Map Execution:")
print("  pipeline.with_map_inputs(x=[1,2,3], y=10).map_over('x')")
print("  pipeline.with_map_inputs(x=[1,2], y=[10,20]).map_over('x', 'y')")
print("  pipeline.with_map_inputs(x=[1,2], y=[10,20]).map_over('x', 'y', mode='product')")
print()
print("Benefits:")
print("  ✅ IDE autocomplete for parameter names")
print("  ✅ Type validation (lists only allowed where appropriate)")
print("  ✅ Fluent, readable API")
print("  ✅ Clear separation: with_inputs vs with_map_inputs")

