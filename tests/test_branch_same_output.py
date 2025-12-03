"""Test case: Same output name in mutually exclusive branches.

This test demonstrates the expected behavior when multiple nodes produce
the same output name but are in mutually exclusive branches.

Tests cover:
- Simple two-way branch with same output name
- Nested branches (three+ nodes) with same output name
- Real-world RAG pipeline pattern
"""

import pytest
from hypernodes import Pipeline, branch, node

# === Simple Two-Way Branch Tests ===


class TestBranchSameOutputName:
    """Tests for same output name in simple two-way branches."""

    @pytest.fixture
    def simple_branch_nodes(self):
        """Create nodes for simple branch test."""

        @node(output_name="validated")
        def validate(value: int) -> dict:
            """Validate input value."""
            return {"value": value, "is_valid": value > 0}

        @node(output_name="response")  # Same output name!
        def handle_valid(validated: dict) -> str:
            """Handle valid input - returns positive response."""
            return f"Valid: {validated['value']}"

        @node(output_name="response")  # Same output name!
        def handle_invalid(validated: dict) -> str:
            """Handle invalid input - returns error response."""
            return f"Invalid: {validated['value']} is not positive"

        @branch(when_true=handle_valid, when_false=handle_invalid)
        def is_valid(validated: dict) -> bool:
            """Branch based on validation result."""
            return validated["is_valid"]

        return [validate, is_valid, handle_valid, handle_invalid]

    def test_pipeline_creation_should_succeed(self, simple_branch_nodes):
        """Pipeline with same output name in exclusive branches should build."""
        pipeline = Pipeline(
            nodes=simple_branch_nodes,
            name="branch_same_output",
        )
        assert pipeline is not None

    def test_valid_path_produces_response(self, simple_branch_nodes):
        """When condition is True, handle_valid produces 'response'."""
        pipeline = Pipeline(
            nodes=simple_branch_nodes,
            name="branch_same_output",
        )
        result = pipeline.run(inputs={"value": 5})
        assert "response" in result
        assert result["response"] == "Valid: 5"
        # handle_invalid should NOT have run
        assert "Invalid" not in result["response"]

    def test_invalid_path_produces_response(self, simple_branch_nodes):
        """When condition is False, handle_invalid produces 'response'."""
        pipeline = Pipeline(
            nodes=simple_branch_nodes,
            name="branch_same_output",
        )
        result = pipeline.run(inputs={"value": -3})
        assert "response" in result
        assert result["response"] == "Invalid: -3 is not positive"
        # handle_valid should NOT have run
        assert "Valid" not in result["response"]


# === Nested Branch Tests ===


class TestNestedBranchSameOutputName:
    """Tests for same output name in nested branches."""

    @pytest.fixture
    def nested_pipeline_nodes(self):
        """Create nodes for nested branch test."""

        @node(output_name="data")
        def get_data(x: int) -> dict:
            return {"value": x, "is_positive": x > 0, "is_large": x > 100}

        @node(output_name="result")  # Same output name
        def handle_negative(data: dict) -> str:
            return "negative"

        @node(output_name="result")  # Same output name
        def handle_small(data: dict) -> str:
            return "small_positive"

        @node(output_name="result")  # Same output name
        def handle_large(data: dict) -> str:
            return "large_positive"

        @branch(when_true=handle_large, when_false=handle_small)
        def check_size(data: dict) -> bool:
            return data["is_large"]

        @branch(when_true=check_size, when_false=handle_negative)
        def check_positive(data: dict) -> bool:
            return data["is_positive"]

        return [
            get_data,
            check_positive,
            check_size,
            handle_negative,
            handle_small,
            handle_large,
        ]

    def test_nested_branches_pipeline_creation(self, nested_pipeline_nodes):
        """Pipeline with three handlers producing 'result' should build."""
        pipeline = Pipeline(nodes=nested_pipeline_nodes, name="nested_branches")
        assert pipeline is not None

    def test_negative_path(self, nested_pipeline_nodes):
        """When x < 0, handle_negative produces 'result'."""
        pipeline = Pipeline(nodes=nested_pipeline_nodes, name="nested_branches")
        result = pipeline.run(inputs={"x": -5})
        assert result["result"] == "negative"

    def test_small_positive_path(self, nested_pipeline_nodes):
        """When 0 < x <= 100, handle_small produces 'result'."""
        pipeline = Pipeline(nodes=nested_pipeline_nodes, name="nested_branches")
        result = pipeline.run(inputs={"x": 50})
        assert result["result"] == "small_positive"

    def test_large_positive_path(self, nested_pipeline_nodes):
        """When x > 100, handle_large produces 'result'."""
        pipeline = Pipeline(nodes=nested_pipeline_nodes, name="nested_branches")
        result = pipeline.run(inputs={"x": 200})
        assert result["result"] == "large_positive"


# === Real-World RAG Pipeline Pattern ===


class TestThreeWayBranchSameOutput:
    """Test case for real-world RAG pipeline pattern.

    This is the scenario from a RAG pipeline:
    - rejection: produced when query is invalid (is_valid=False)
    - figure_response: produced when document has figures (is_valid=True, has_figures=True)
    - generation: produced otherwise (is_valid=True, has_figures=False)

    All three are mutually exclusive through nested branches.
    """

    @pytest.fixture
    def rag_pipeline_nodes(self):
        """Create nodes for RAG pipeline test."""

        @node(output_name="validation_result")
        def validate_query(query: str) -> dict:
            return {"is_valid": "medical" in query.lower(), "query": query}

        @node(output_name="response")  # Output 1
        def create_rejection(validation_result: dict) -> str:
            return f"Rejected: {validation_result['query']}"

        @node(output_name="document")
        def retrieve_docs(query: str) -> dict:
            return {"title": "Medical Guide", "has_figures": "figure" in query.lower()}

        @node(output_name="response")  # Output 2 - same as rejection!
        def create_figure_response(document: dict) -> str:
            return f"See figures in: {document['title']}"

        @node(output_name="response")  # Output 3 - same as rejection and figure!
        def generate_answer(document: dict) -> str:
            return f"Answer based on: {document['title']}"

        @branch(when_true=create_figure_response, when_false=generate_answer)
        def check_figures(document: dict) -> bool:
            return document["has_figures"]

        @branch(when_true=retrieve_docs, when_false=create_rejection)
        def check_valid(validation_result: dict) -> bool:
            return validation_result["is_valid"]

        return [
            validate_query,
            check_valid,
            create_rejection,
            retrieve_docs,
            check_figures,
            create_figure_response,
            generate_answer,
        ]

    def test_rag_pipeline_creation(self, rag_pipeline_nodes):
        """RAG pipeline with three 'response' producers should build."""
        pipeline = Pipeline(
            nodes=rag_pipeline_nodes,
            name="rag_pipeline",
        )
        assert pipeline is not None

    def test_rejection_path(self, rag_pipeline_nodes):
        """Invalid query produces rejection response."""
        pipeline = Pipeline(nodes=rag_pipeline_nodes, name="rag_pipeline")
        result = pipeline.run(inputs={"query": "What's the weather?"})
        assert result["response"] == "Rejected: What's the weather?"

    def test_figure_response_path(self, rag_pipeline_nodes):
        """Valid query with figures produces figure response."""
        pipeline = Pipeline(nodes=rag_pipeline_nodes, name="rag_pipeline")
        result = pipeline.run(inputs={"query": "Show medical figure"})
        assert "See figures" in result["response"]

    def test_generation_path(self, rag_pipeline_nodes):
        """Valid query without figures produces generated answer."""
        pipeline = Pipeline(nodes=rag_pipeline_nodes, name="rag_pipeline")
        result = pipeline.run(inputs={"query": "Medical treatment"})
        assert "Answer based on" in result["response"]


# === Workaround Test (for comparison) ===


class TestBranchDifferentOutputNames:
    """Workaround: Using different output names (for comparison)."""

    def test_different_output_names_works(self):
        """Different output names work but require checking multiple keys."""

        @node(output_name="validated")
        def validate_v2(value: int) -> dict:
            return {"value": value, "is_valid": value > 0}

        @node(output_name="valid_response")  # Different name
        def handle_valid_v2(validated: dict) -> str:
            return f"Valid: {validated['value']}"

        @node(output_name="invalid_response")  # Different name
        def handle_invalid_v2(validated: dict) -> str:
            return f"Invalid: {validated['value']}"

        @branch(when_true=handle_valid_v2, when_false=handle_invalid_v2)
        def is_valid_v2(validated: dict) -> bool:
            return validated["is_valid"]

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
