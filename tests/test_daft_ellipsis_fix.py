"""Test that DaftEngine code generation doesn't use ellipsis (...) in lists."""

from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine


def test_long_list_no_ellipsis():
    """Generated code should have full list representation, not truncated with ..."""

    @node(output_name="result")
    def process(values: list[int]) -> int:
        return sum(values)

    pipeline = Pipeline(nodes=[process], name="test")

    # Use a long list that reprlib would truncate
    long_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Generate Daft code
    code = pipeline.show_daft_code(inputs={"values": long_list}, output_name="result")

    print("Generated code:")
    print("=" * 80)
    print(code)
    print("=" * 80)

    # Verify no ellipsis in generated code
    assert "..." not in code, "Generated code should not contain ellipsis (...)"

    # Verify the full list is present
    assert str(long_list) in code, f"Full list {long_list} should be in generated code"

    # Verify it's valid Python
    compile(code, "<generated>", "exec")

    print("\n✓ Long list correctly formatted in generated code (no ellipsis)")


def test_very_long_list_no_ellipsis():
    """Test with a very long list like recall_k_list."""

    @node(output_name="result")
    def process(k_values: list[int]) -> int:
        return max(k_values)

    pipeline = Pipeline(nodes=[process], name="test")

    # Similar to the recall_k_list that caused the bug
    k_values = [20, 50, 100, 200, 300, 400, 500]

    code = pipeline.show_daft_code(inputs={"k_values": k_values}, output_name="result")

    print("\nGenerated code for k_values:")
    print("=" * 80)

    # Find the from_pydict section
    start_idx = code.find("from_pydict")
    if start_idx != -1:
        section = code[start_idx:start_idx + 200]
        print(section)

    print("=" * 80)

    # Verify no ellipsis
    assert "..." not in code, "Generated code should not contain ellipsis"

    # Verify the full list
    assert str(k_values) in code, f"Full list {k_values} should be in generated code"

    # Verify each value is present
    for k in k_values:
        assert str(k) in code, f"Value {k} should be in generated code"

    print(f"\n✓ All {len(k_values)} values present in generated code")


if __name__ == "__main__":
    test_long_list_no_ellipsis()
    test_very_long_list_no_ellipsis()
    print("\n✓✓ All ellipsis fix tests pass!")
