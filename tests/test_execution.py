"""Tests for basic pipeline execution with SeqEngine (default)."""

from hypernodes import Pipeline, node


def test_single_node_pipeline():
    """Test pipeline with a single node."""

    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    pipeline = Pipeline(nodes=[double])
    result = pipeline.run(inputs={"x": 5})

    assert result == {"doubled": 10}


def test_sequential_nodes():
    """Test pipeline with sequential dependencies."""

    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    @node(output_name="result")
    def add_one(doubled: int) -> int:
        return doubled + 1

    pipeline = Pipeline(nodes=[double, add_one])
    result = pipeline.run(inputs={"x": 5})

    assert result == {"doubled": 10, "result": 11}


def test_diamond_dependency():
    """Test pipeline with diamond dependency pattern."""

    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    @node(output_name="tripled")
    def triple(x: int) -> int:
        return x * 3

    @node(output_name="result")
    def combine(doubled: int, tripled: int) -> int:
        return doubled + tripled

    pipeline = Pipeline(nodes=[double, triple, combine])
    result = pipeline.run(inputs={"x": 5})

    assert result == {"doubled": 10, "tripled": 15, "result": 25}


def test_multiple_outputs():
    """Test node with multiple outputs."""

    @node(output_name=("doubled", "tripled"))
    def process(x: int) -> tuple:
        return x * 2, x * 3

    @node(output_name="result")
    def combine(doubled: int, tripled: int) -> int:
        return doubled + tripled

    pipeline = Pipeline(nodes=[process, combine])
    result = pipeline.run(inputs={"x": 5})

    assert result == {"doubled": 10, "tripled": 15, "result": 25}


def test_selective_output():
    """Test requesting only specific outputs."""

    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    @node(output_name="tripled")
    def triple(x: int) -> int:
        return x * 3

    pipeline = Pipeline(nodes=[double, triple])
    result = pipeline.run(inputs={"x": 5}, output_name="doubled")

    assert result == {"doubled": 10}
    assert "tripled" not in result


def test_selective_multiple_outputs():
    """Test requesting multiple specific outputs."""

    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    @node(output_name="tripled")
    def triple(x: int) -> int:
        return x * 3

    @node(output_name="quadrupled")
    def quadruple(x: int) -> int:
        return x * 4

    pipeline = Pipeline(nodes=[double, triple, quadruple])
    result = pipeline.run(inputs={"x": 5}, output_name=["doubled", "quadrupled"])

    assert result == {"doubled": 10, "quadrupled": 20}
    assert "tripled" not in result
