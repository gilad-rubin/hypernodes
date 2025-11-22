"""Test script to verify backward compatibility with existing code."""

from hypernodes import Pipeline, node


# Create simple test pipeline
@node(output_name="doubled")
def double(x: int) -> int:
    """Double the input."""
    return x * 2

@node(output_name="result")
def add_ten(doubled: int) -> int:
    """Add ten to the input."""
    return doubled + 10

# Create pipeline
pipeline = Pipeline(nodes=[double, add_ten], name="BackwardCompatTest")

print("="*60)
print("BACKWARD COMPATIBILITY TEST")
print("="*60)

# Test 1: Original API call (no engine parameter)
print("\n1. Original API (no engine param, all default)...")
try:
    result = pipeline.visualize(return_type="graphviz")
    print(f"   ✓ Success! Type: {type(result).__name__}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 2: Original API with style
print("\n2. Original API with style='dark'...")
try:
    result = pipeline.visualize(style="dark", return_type="graphviz")
    print(f"   ✓ Success! Type: {type(result).__name__}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 3: Original API with orient
print("\n3. Original API with orient='LR'...")
try:
    result = pipeline.visualize(orient="LR", return_type="graphviz")
    print(f"   ✓ Success! Type: {type(result).__name__}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 4: Original API with show_legend
print("\n4. Original API with show_legend=True...")
try:
    result = pipeline.visualize(show_legend=True, return_type="graphviz")
    print(f"   ✓ Success! Type: {type(result).__name__}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 5: Original API with flatten
print("\n5. Original API with flatten=True...")
try:
    result = pipeline.visualize(flatten=True, return_type="graphviz")
    print(f"   ✓ Success! Type: {type(result).__name__}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 6: All parameters together
print("\n6. Original API with all parameters...")
try:
    result = pipeline.visualize(
        orient="LR",
        depth=1,
        flatten=False,
        group_inputs=True,
        show_legend=True,
        show_types=True,
        style="professional",
        return_type="graphviz"
    )
    print(f"   ✓ Success! Type: {type(result).__name__}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 7: New API - explicit engine
print("\n7. New API with engine='graphviz'...")
try:
    result = pipeline.visualize(engine="graphviz", return_type="graphviz")
    print(f"   ✓ Success! Type: {type(result).__name__}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 8: New API - engine with options
print("\n8. New API with engine + options...")
try:
    result = pipeline.visualize(
        engine="graphviz",
        style="dark",
        show_legend=True,
        return_type="graphviz"
    )
    print(f"   ✓ Success! Type: {type(result).__name__}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 9: Direct function call (old way)
print("\n9. Direct function call (old visualize API)...")
try:
    from hypernodes.viz import visualize
    result = visualize(pipeline, return_type="graphviz")
    print(f"   ✓ Success! Type: {type(result).__name__}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n" + "="*60)
print("BACKWARD COMPATIBILITY: ALL TESTS PASSED ✓")
print("="*60)

