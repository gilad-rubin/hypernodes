"""
Tests adapted from scripts/test_pipeline_as_node.py for:
- Pipeline.with_name() behavior
- Direct pipeline nesting execution
"""

from hypernodes import Pipeline, node


def test_pipeline_with_name_sets_name_and_is_chainable():
    @node(output_name="cleaned")
    def clean_text(text: str) -> str:
        return text.strip()

    @node(output_name="uppercased")
    def uppercase_text(cleaned: str) -> str:
        return cleaned.upper()

    p = Pipeline(nodes=[clean_text, uppercase_text]).with_name("preprocessing")
    assert p.name == "preprocessing"
    # chained call should keep same instance
    p2 = p.with_name("pre2")
    assert p2 is p
    assert p.name == "pre2"


def test_direct_pipeline_nesting_runs_and_propagates_outputs():
    @node(output_name="cleaned")
    def clean_text(text: str) -> str:
        return text.strip()

    @node(output_name="uppercased")
    def uppercase_text(cleaned: str) -> str:
        return cleaned.upper()

    @node(output_name="result")
    def add_prefix(uppercased: str, prefix: str) -> str:
        return f"{prefix}: {uppercased}"

    inner = Pipeline(nodes=[clean_text, uppercase_text]).with_name("preprocessing")
    outer = Pipeline(nodes=[inner, add_prefix]).with_name("main_pipeline")

    result = outer.run(inputs={"text": "  hello world  ", "prefix": "OUTPUT"})
    assert result == {
        "cleaned": "hello world",
        "uppercased": "HELLO WORLD",
        "result": "OUTPUT: HELLO WORLD",
    }


def test_direct_nesting_equivalent_to_as_node():
    @node(output_name="cleaned")
    def clean_text(text: str) -> str:
        return text.strip()

    @node(output_name="uppercased")
    def uppercase_text(cleaned: str) -> str:
        return cleaned.upper()

    @node(output_name="result")
    def add_prefix(uppercased: str, prefix: str) -> str:
        return f"{prefix}: {uppercased}"

    inner = Pipeline(nodes=[clean_text, uppercase_text]).with_name("preprocessing")

    # Method 1: Direct nesting
    outer1 = Pipeline(nodes=[inner, add_prefix])

    # Method 2: Using as_node()
    node_pipe = inner.as_node()
    outer2 = Pipeline(nodes=[node_pipe, add_prefix])

    inputs = {"text": "  test  ", "prefix": "RESULT"}
    result1 = outer1.run(inputs=inputs)
    result2 = outer2.run(inputs=inputs)
    assert result1 == result2
