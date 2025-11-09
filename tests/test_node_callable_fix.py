"""Test that Node is callable with positional arguments."""

from hypernodes import node


def test_node_callable_with_positional_args():
    """Node should be callable with positional arguments."""

    @node(output_name="result")
    def add(a: int, b: int) -> int:
        return a + b

    # Test calling with positional args (like in generated Daft code)
    result = add(3, 5)
    assert result == 8

    # Also test with keyword args (existing behavior)
    result = add(a=10, b=20)
    assert result == 30

    # Mixed positional and keyword
    result = add(7, b=3)
    assert result == 10


def test_node_callable_for_daft_wrapper():
    """
    Test the pattern used in Daft code generation.

    When wrapping nodes in @daft.cls wrappers, we call them with positional args:

    @daft.cls
    class MyWrapper:
        def __call__(self, x):
            return my_node(x, self.config)  # ← positional args
    """

    @node(output_name="processed")
    def process(text: str, prefix: str) -> str:
        return f"{prefix}: {text}"

    # Simulate Daft wrapper pattern
    class Wrapper:
        def __init__(self, prefix: str):
            self.prefix = prefix

        def __call__(self, text: str):
            # This is how we call nodes in generated Daft code
            return process(text, self.prefix)

    wrapper = Wrapper("INFO")
    result = wrapper("hello")

    assert result == "INFO: hello"


if __name__ == "__main__":
    test_node_callable_with_positional_args()
    test_node_callable_for_daft_wrapper()
    print("✓ All Node callable tests pass!")
