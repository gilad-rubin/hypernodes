"""Test case: Same output name in mutually exclusive branches.

This test demonstrates the expected behavior when multiple nodes produce
the same output name but are in mutually exclusive branches.

STATUS:
- Simple two-way branch: WORKS ✓
- Three+ nodes across nested branches: FAILS ✗

The nested case fails with:
    DependencyError: Multiple nodes produce output 'response': 
    rejection and figure_response. This is only allowed when 
    producers are in mutually exclusive branches.

Expected behavior: All three are mutually exclusive because:
1. rejection is in is_valid=False branch
2. figure_response is in has_figures=True branch (nested under is_valid=True)
3. generation is in has_figures=False branch (nested under is_valid=True)
"""

import pytest
from hypernodes import Pipeline, branch, node


# === Test Setup ===


@node(output_name="validated")
def validate(value: int) -> dict:
    """Validate input value."""
    return {"value": value, "is_valid": value > 0}


@branch(when_true="handle_valid", when_false="handle_invalid")
def is_valid(validated: dict) -> bool:
    """Branch based on validation result."""
    return validated["is_valid"]


@node(output_name="response")  # Same output name!
def handle_valid(validated: dict) -> str:
    """Handle valid input - returns positive response."""
    return f"Valid: {validated['value']}"


@node(output_name="response")  # Same output name!
def handle_invalid(validated: dict) -> str:
    """Handle invalid input - returns error response."""
    return f"Invalid: {validated['value']} is not positive"


# === Tests ===


class TestBranchSameOutputName:
    """Tests for same output name in mutually exclusive branches."""

    def test_pipeline_creation_should_succeed(self):
        """Pipeline with same output name in exclusive branches should build."""
        # This currently FAILS - should PASS
        pipeline = Pipeline(
            nodes=[validate, is_valid, handle_valid, handle_invalid],
            name="branch_same_output",
        )
        assert pipeline is not None

    def test_valid_path_produces_response(self):
        """When condition is True, handle_valid produces 'response'."""
        pipeline = Pipeline(
            nodes=[validate, is_valid, handle_valid, handle_invalid],
            name="branch_same_output",
        )

        result = pipeline.run(inputs={"value": 5})

        assert "response" in result
        assert result["response"] == "Valid: 5"
        # handle_invalid should NOT have run
        assert "Invalid" not in result["response"]

    def test_invalid_path_produces_response(self):
        """When condition is False, handle_invalid produces 'response'."""
        pipeline = Pipeline(
            nodes=[validate, is_valid, handle_valid, handle_invalid],
            name="branch_same_output",
        )

        result = pipeline.run(inputs={"value": -3})

        assert "response" in result
        assert result["response"] == "Invalid: -3 is not positive"
        # handle_valid should NOT have run
        assert "Valid" not in result["response"]


class TestNestedBranchSameOutputName:
    """Tests for same output name in nested branches."""

    @pytest.fixture
    def nested_pipeline_nodes(self):
        """Create nodes for nested branch test."""

        @node(output_name="data")
        def get_data(x: int) -> dict:
            return {"value": x, "is_positive": x > 0, "is_large": x > 100}

        @branch(when_true="check_size", when_false="handle_negative")
        def check_positive(data: dict) -> bool:
            return data["is_positive"]

        @branch(when_true="handle_large", when_false="handle_small")
        def check_size(data: dict) -> bool:
            return data["is_large"]

        @node(output_name="result")  # Same output name
        def handle_negative(data: dict) -> str:
            return "negative"

        @node(output_name="result")  # Same output name
        def handle_small(data: dict) -> str:
            return "small_positive"

        @node(output_name="result")  # Same output name
        def handle_large(data: dict) -> str:
            return "large_positive"

        return [get_data, check_positive, check_size, handle_negative, handle_small, handle_large]

    def test_nested_branches_same_output(self, nested_pipeline_nodes):
        """All three handlers produce 'result' but are mutually exclusive."""
        pipeline = Pipeline(nodes=nested_pipeline_nodes, name="nested_branches")

        # Test negative path
        result = pipeline.run(inputs={"x": -5})
        assert result["result"] == "negative"

        # Test small positive path
        result = pipeline.run(inputs={"x": 50})
        assert result["result"] == "small_positive"

        # Test large positive path
        result = pipeline.run(inputs={"x": 200})
        assert result["result"] == "large_positive"


# === Workaround: Different Output Names (Currently Works) ===


class TestBranchDifferentOutputNames:
    """Workaround: Using different output names (currently works)."""

    def test_different_output_names_works(self):
        """Different output names work but require checking multiple keys."""

        @node(output_name="validated")
        def validate_v2(value: int) -> dict:
            return {"value": value, "is_valid": value > 0}

        @branch(when_true="handle_valid_v2", when_false="handle_invalid_v2")
        def is_valid_v2(validated: dict) -> bool:
            return validated["is_valid"]

        @node(output_name="valid_response")  # Different name
        def handle_valid_v2(validated: dict) -> str:
            return f"Valid: {validated['value']}"

        @node(output_name="invalid_response")  # Different name
        def handle_invalid_v2(validated: dict) -> str:
            return f"Invalid: {validated['value']}"

        pipeline = Pipeline(
            nodes=[validate_v2, is_valid_v2, handle_valid_v2, handle_invalid_v2],
            name="branch_different_output",
        )

        # Valid path
        result = pipeline.run(inputs={"value": 5})
        assert "valid_response" in result
        assert "invalid_response" not in result

        # Invalid path
        result = pipeline.run(inputs={"value": -3})
        assert "invalid_response" in result
        assert "valid_response" not in result


class TestThreeWayBranchSameOutput:
    """Test case that currently FAILS: Three nodes producing same output across nested branches.

    This is the real-world scenario from a RAG pipeline:
    - rejection: produced when query is invalid (is_valid=False)
    - figure_response: produced when document has figures (is_valid=True, has_figures=True)
    - generation: produced otherwise (is_valid=True, has_figures=False)

    All three are mutually exclusive but HyperNodes doesn't detect this.
    """

    def test_three_way_branch_same_output_should_work(self):
        """Three mutually exclusive paths should be able to produce same output."""

        @node(output_name="validation_result")
        def validate_query(query: str) -> dict:
            return {"is_valid": "medical" in query.lower(), "query": query}

        @branch(when_true="retrieve_docs", when_false="create_rejection")
        def check_valid(validation_result: dict) -> bool:
            return validation_result["is_valid"]

        @node(output_name="response")  # Output 1
        def create_rejection(validation_result: dict) -> str:
            return f"Rejected: {validation_result['query']}"

        @node(output_name="document")
        def retrieve_docs(query: str) -> dict:
            return {"title": "Medical Guide", "has_figures": "figure" in query.lower()}

        @branch(when_true="create_figure_response", when_false="generate_answer")
        def check_figures(document: dict) -> bool:
            return document["has_figures"]

        @node(output_name="response")  # Output 2 - same as rejection!
        def create_figure_response(document: dict) -> str:
            return f"See figures in: {document['title']}"

        @node(output_name="response")  # Output 3 - same as rejection and figure!
        def generate_answer(document: dict) -> str:
            return f"Answer based on: {document['title']}"

        # This currently FAILS - should PASS
        pipeline = Pipeline(
            nodes=[
                validate_query,
                check_valid,
                create_rejection,
                retrieve_docs,
                check_figures,
                create_figure_response,
                generate_answer,
            ],
            name="rag_pipeline",
        )

        # Test rejection path (invalid query)
        result = pipeline.run(inputs={"query": "What's the weather?"})
        assert result["response"] == "Rejected: What's the weather?"

        # Test figure path (valid + has figures)
        result = pipeline.run(inputs={"query": "Show medical figure"})
        assert "See figures" in result["response"]

        # Test generation path (valid + no figures)
        result = pipeline.run(inputs={"query": "Medical treatment"})
        assert "Answer based on" in result["response"]


if __name__ == "__main__":
    print("=" * 60)
    print("Testing branch with same output name")
    print("=" * 60)

    # Test 1: Simple two-way branch (WORKS)
    print("\n1. Simple two-way branch:")
    try:
        pipeline = Pipeline(
            nodes=[validate, is_valid, handle_valid, handle_invalid],
            name="test",
        )
        print("   ✓ Pipeline created successfully")

        result = pipeline.run(inputs={"value": 5})
        print(f"   ✓ Valid path: {result['response']}")

        result = pipeline.run(inputs={"value": -3})
        print(f"   ✓ Invalid path: {result['response']}")

    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Test 2: Three-way nested branch (FAILS)
    print("\n2. Three-way nested branch (the bug):")

    @node(output_name="validation_result")
    def validate_query(query: str) -> dict:
        return {"is_valid": "medical" in query.lower(), "query": query}

    @branch(when_true="retrieve_docs", when_false="create_rejection")
    def check_valid(validation_result: dict) -> bool:
        return validation_result["is_valid"]

    @node(output_name="response")
    def create_rejection(validation_result: dict) -> str:
        return f"Rejected: {validation_result['query']}"

    @node(output_name="document")
    def retrieve_docs(query: str) -> dict:
        return {"title": "Medical Guide", "has_figures": "figure" in query.lower()}

    @branch(when_true="create_figure_response", when_false="generate_answer")
    def check_figures(document: dict) -> bool:
        return document["has_figures"]

    @node(output_name="response")
    def create_figure_response(document: dict) -> str:
        return f"See figures in: {document['title']}"

    @node(output_name="response")
    def generate_answer(document: dict) -> str:
        return f"Answer based on: {document['title']}"

    try:
        pipeline = Pipeline(
            nodes=[
                validate_query,
                check_valid,
                create_rejection,
                retrieve_docs,
                check_figures,
                create_figure_response,
                generate_answer,
            ],
            name="rag_pipeline",
        )
        print("   ✓ Pipeline created successfully")

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        print("\n   This is the bug to fix!")
        print("   All three 'response' producers are mutually exclusive:")
        print("   - create_rejection: is_valid=False")
        print("   - create_figure_response: is_valid=True AND has_figures=True")
        print("   - generate_answer: is_valid=True AND has_figures=False")

