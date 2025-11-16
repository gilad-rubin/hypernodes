"""Test the user's original example."""

from hypernodes import Pipeline, node


@node(output_name="hello")
def a(x: int) -> int:
    return x + 1


@node
def b(hello: int, y: int) -> int:
    return hello + y


pipeline = Pipeline(nodes=[a, b])
pipeline.visualize()

# This should now work - only requires 'x' to compute 'hello'
result = pipeline.run(inputs={"x": 1}, output_name="hello")
print(f"Result: {result}")
assert result == {"hello": 2}, f"Expected {{'hello': 2}}, got {result}"

print("✓ Test passed!")

