"""Demonstration of selective input validation based on output_name.

This feature allows you to provide only the inputs needed to compute
specific outputs, rather than all inputs for the entire pipeline.
"""

from hypernodes import Pipeline, node


@node(output_name="a")
def step_a(x: int) -> int:
    """First step - only needs x."""
    print(f"  Executing step_a with x={x}")
    return x + 1


@node(output_name="b")
def step_b(a: int) -> int:
    """Second step - depends on a."""
    print(f"  Executing step_b with a={a}")
    return a * 2


@node(output_name="c")
def step_c(b: int, y: int) -> int:
    """Third step - needs b and y."""
    print(f"  Executing step_c with b={b}, y={y}")
    return b + y


def main():
    pipeline = Pipeline(nodes=[step_a, step_b, step_c])

    print("=" * 60)
    print("Pipeline Structure:")
    print("  step_a(x) -> a")
    print("  step_b(a) -> b")
    print("  step_c(b, y) -> c")
    print("=" * 60)

    # Example 1: Get only 'a' - only needs x
    print("\n1. Getting 'a' (only needs x):")
    result = pipeline.run(inputs={"x": 5}, output_name="a")
    print(f"   Result: {result}")
    assert result == {"a": 6}

    # Example 2: Get only 'b' - needs x but not y
    print("\n2. Getting 'b' (needs x, not y):")
    result = pipeline.run(inputs={"x": 5}, output_name="b")
    print(f"   Result: {result}")
    assert result == {"b": 12}

    # Example 3: Get 'c' - needs both x and y
    print("\n3. Getting 'c' (needs both x and y):")
    result = pipeline.run(inputs={"x": 5, "y": 3}, output_name="c")
    print(f"   Result: {result}")
    assert result == {"c": 15}

    # Example 4: Without output_name - needs all inputs
    print("\n4. Getting all outputs (needs all inputs):")
    result = pipeline.run(inputs={"x": 5, "y": 3})
    print(f"   Result: {result}")
    assert result == {"a": 6, "b": 12, "c": 15}

    # Example 5: Multiple outputs
    print("\n5. Getting 'a' and 'b' (only needs x):")
    result = pipeline.run(inputs={"x": 5}, output_name=["a", "b"])
    print(f"   Result: {result}")
    assert result == {"a": 6, "b": 12}

    print("\n" + "=" * 60)
    print("✓ All examples passed!")
    print("=" * 60)

    # Example 6: Error case - missing required input
    print("\n6. Error case - trying to get 'c' without 'y':")
    try:
        pipeline.run(inputs={"x": 5}, output_name="c")
        print("   ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"   ✓ Correctly raised error: {e}")


if __name__ == "__main__":
    main()

