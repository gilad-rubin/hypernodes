"""Test basic construction of nodes and pipelines after refactor."""

from hypernodes import Pipeline, node
from hypernodes.pipeline_node import PipelineNode


def test_single_node_construction():
    """Test single node construction with proper attributes."""
    @node(output_name="result")
    def add_one(x: int) -> int:
        return x + 1

    assert add_one.output_name == "result"
    assert add_one.root_args == ("x",)
    assert add_one.cache is True
    assert add_one.name == "add_one"
    assert len(add_one.code_hash) == 64  # SHA256 hex digest


def test_pipeline_with_single_node():
    """Test pipeline construction with a single node."""
    @node(output_name="result")
    def add_one(x: int) -> int:
        return x + 1

    pipeline = Pipeline(nodes=[add_one])
    
    assert pipeline.graph.root_args == ["x"]
    assert pipeline.graph.available_output_names == ["result"]
    assert pipeline.graph.execution_order == [add_one]


def test_pipeline_with_two_sequential_nodes():
    """Test pipeline with two nodes in sequence."""
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    @node(output_name="final")
    def add_one_to_doubled(doubled: int) -> int:
        return doubled + 1

    pipeline = Pipeline(nodes=[double, add_one_to_doubled])
    
    assert pipeline.graph.root_args == ["x"]
    assert set(pipeline.graph.available_output_names) == {"doubled", "final"}
    assert pipeline.graph.execution_order == [double, add_one_to_doubled]


def test_pipeline_node_construction():
    """Test PipelineNode wrapping a pipeline."""
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    inner_pipeline = Pipeline(nodes=[double])
    pipeline_node = PipelineNode(pipeline=inner_pipeline)
    
    assert pipeline_node.root_args == ("x",)
    assert pipeline_node.output_name == "doubled"
    assert pipeline_node.cache is True
    assert len(pipeline_node.code_hash) == 64
    assert pipeline_node.pipeline is inner_pipeline


def test_nested_pipeline_construction():
    """Test nested pipeline: PipelineNode used within another pipeline."""
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    @node(output_name="tripled")
    def triple(doubled: int) -> int:
        return doubled + doubled // 2

    # Create inner pipeline
    inner = Pipeline(nodes=[double])
    
    # Wrap as PipelineNode
    inner_node = PipelineNode(pipeline=inner)
    
    # Create outer pipeline that uses the PipelineNode
    outer = Pipeline(nodes=[inner_node, triple])
    
    # Verify inner pipeline
    assert inner.graph.available_output_names == ["doubled"]
    
    # Verify PipelineNode
    assert inner_node.output_name == "doubled"
    
    # Verify outer pipeline
    assert outer.graph.root_args == ["x"]
    assert set(outer.graph.available_output_names) == {"doubled", "tripled"}
    assert outer.graph.execution_order == [inner_node, triple]


def test_node_with_custom_name():
    """Test that node name property works correctly."""
    @node(output_name="result")
    def my_function(x: int) -> int:
        return x + 1

    assert my_function.name == "my_function"


def test_pipeline_node_with_custom_name():
    """Test PipelineNode with custom name."""
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    inner = Pipeline(nodes=[double], name="DoubleOp")
    pipeline_node = PipelineNode(pipeline=inner, name="CustomDouble")
    
    assert pipeline_node.name == "CustomDouble"


def test_pipeline_node_without_custom_name():
    """Test PipelineNode uses pipeline name when no custom name provided."""
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    inner = Pipeline(nodes=[double], name="DoubleOp")
    pipeline_node = PipelineNode(pipeline=inner)
    
    assert pipeline_node.name == "DoubleOp"

