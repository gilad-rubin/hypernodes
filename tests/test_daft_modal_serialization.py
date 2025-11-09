"""Test that DaftEngine handles Modal-style serialization correctly.

This test verifies that node functions defined inside other functions
(like Modal's run_pipeline) can be serialized and executed correctly.
"""

import pytest

try:
    import daft
    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False

from hypernodes import node, Pipeline

if DAFT_AVAILABLE:
    from hypernodes.engines import DaftEngine

pytestmark = pytest.mark.skipif(not DAFT_AVAILABLE, reason="Daft not installed")


def test_nodes_defined_in_function():
    """Nodes defined inside a function should serialize correctly."""

    def create_pipeline():
        """This simulates a Modal function that defines nodes inline."""

        @node(output_name="doubled")
        def double(x: int) -> int:
            return x * 2

        @node(output_name="result")
        def add_ten(doubled: int) -> int:
            return doubled + 10

        return Pipeline(nodes=[double, add_ten], engine=DaftEngine())

    # Create pipeline inside function (like Modal does)
    pipeline = create_pipeline()

    # Should work without ModuleNotFoundError
    result = pipeline.run(inputs={"x": 5})

    assert result["result"] == 20


def test_stateful_objects_in_function():
    """Stateful objects passed to nodes in a function should serialize correctly."""

    def create_pipeline_with_state():
        """Simulate Modal pattern with stateful objects."""

        class Multiplier:
            def __init__(self, factor: int):
                self.factor = factor

            def multiply(self, x: int) -> int:
                return x * self.factor

        @node(output_name="multiplied")
        def apply_multiplier(x: int, multiplier: Multiplier) -> int:
            return multiplier.multiply(x)

        @node(output_name="result")
        def add_ten(multiplied: int) -> int:
            return multiplied + 10

        multiplier = Multiplier(factor=3)

        return Pipeline(
            nodes=[apply_multiplier, add_ten],
            engine=DaftEngine()
        ), multiplier

    # Create pipeline and stateful object
    pipeline, multiplier = create_pipeline_with_state()

    # Should work without serialization errors
    result = pipeline.run(inputs={"x": 5, "multiplier": multiplier})

    assert result["result"] == 25  # (5 * 3) + 10


def test_pydantic_models_in_function():
    """Pydantic models used in nodes defined in function should work."""
    from pydantic import BaseModel
    from typing import List

    def create_pipeline_with_pydantic():
        """Simulate Modal pattern with Pydantic models."""

        class Document(BaseModel):
            id: str
            text: str
            model_config = {"frozen": True}

        @node(output_name="documents")
        def create_docs(count: int) -> List[Document]:
            return [
                Document(id=f"doc_{i}", text=f"Document {i}")
                for i in range(count)
            ]

        @node(output_name="first_text")
        def get_first(documents: List[Document]) -> str:
            return documents[0].text if documents else ""

        return Pipeline(nodes=[create_docs, get_first], engine=DaftEngine())

    # Create pipeline
    pipeline = create_pipeline_with_pydantic()

    # Should work without serialization errors
    result = pipeline.run(inputs={"count": 3})

    assert result["first_text"] == "Document 0"


def test_function_closure_captures():
    """Functions that capture closure variables should serialize correctly."""

    def create_pipeline_with_closure(base_value: int):
        """Simulate closure capture (common in Modal patterns)."""

        @node(output_name="added")
        def add_base(x: int) -> int:
            # Captures base_value from closure
            return x + base_value

        @node(output_name="result")
        def double(added: int) -> int:
            return added * 2

        return Pipeline(nodes=[add_base, double], engine=DaftEngine())

    # Create pipeline with closure
    pipeline = create_pipeline_with_closure(base_value=10)

    # Should work with closure capture
    result = pipeline.run(inputs={"x": 5})

    assert result["result"] == 30  # (5 + 10) * 2


if __name__ == "__main__":
    test_nodes_defined_in_function()
    test_stateful_objects_in_function()
    test_pydantic_models_in_function()
    test_function_closure_captures()

    print("\n" + "=" * 80)
    print("✓✓ All Modal serialization tests pass!")
    print("=" * 80)
    print("\nKey findings:")
    print("1. ✅ Nodes defined inside functions serialize correctly")
    print("2. ✅ Stateful objects work with inline node definitions")
    print("3. ✅ Pydantic models work with inline definitions")
    print("4. ✅ Closure captures are preserved during serialization")
    print("\nThis confirms DaftEngine works with Modal-style patterns!")
