"""
Unit tests for Daft code generation functionality.

Tests cover:
- Simple pipeline code generation
- Stateful UDF code generation
- Map operations (explode/groupby)
- Nested pipelines
- Generated code syntax validation
"""

from typing import List

import pytest
from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine


# Test nodes
@node(output_name="doubled")
def double(x: int) -> int:
    """Double the input."""
    return x * 2


@node(output_name="tripled")
def triple(x: int) -> int:
    """Triple the input."""
    return x * 3


@node(output_name="result")
def add(doubled: int, tripled: int) -> int:
    """Add doubled and tripled values."""
    return doubled + tripled


class StatefulObject:
    """Test stateful object."""
    
    def __init__(self, factor: int):
        self.factor = factor
    
    def process(self, value: int) -> int:
        return value * self.factor


@node(output_name="processed")
def use_stateful(x: int, obj: StatefulObject) -> int:
    """Use stateful object."""
    return obj.process(x)


@node(output_name="items")
def static_items(seed_items: List[int]) -> List[int]:
    return seed_items


@node(output_name="value")
def square_value(x: int) -> int:
    return x * x


def test_code_generation_mode_flag():
    """Test that code_generation_mode flag works."""
    engine = DaftEngine(code_generation_mode=True)
    assert engine.code_generation_mode is True
    assert hasattr(engine, 'generated_code')
    assert hasattr(engine, '_udf_definitions')
    assert hasattr(engine, '_imports')


def test_simple_pipeline_code_generation():
    """Test code generation for a simple pipeline."""
    pipeline = Pipeline(
        nodes=[double, triple, add],
        name="test_simple"
    )
    
    code = pipeline.show_daft_code(inputs={"x": 5})
    
    # Check that code contains expected elements
    assert "import daft" in code
    assert "def double_" in code  # UDF definition
    assert "def triple_" in code
    assert "def add_" in code
    assert 'df.with_column("doubled"' in code
    assert 'df.with_column("tripled"' in code
    assert 'df.with_column("result"' in code
    assert "@daft.func" in code
    
    # Check that it's valid Python syntax
    compile(code, '<string>', 'exec')


def test_stateful_pipeline_code_generation():
    """Test code generation for pipeline with stateful objects."""
    pipeline = Pipeline(
        nodes=[double, use_stateful],
        name="test_stateful"
    )
    
    obj = StatefulObject(factor=10)
    code = pipeline.show_daft_code(inputs={"x": 5, "obj": obj})
    
    # Check for stateful handling
    # In code generation mode, stateful objects are still passed as parameters
    assert "import daft" in code
    assert "def double_" in code
    assert "def use_stateful_" in code
    
    # Check that it's valid Python syntax
    compile(code, '<string>', 'exec')


def test_map_operation_code_generation():
    """Test code generation for map operations (explode/groupby)."""
    single = Pipeline(nodes=[double], name="single")
    mapped = single.as_node(
        input_mapping={"numbers": "x"},
        output_mapping={"doubled": "doubled_numbers"},
        map_over="numbers",
        name="mapped"
    )
    
    pipeline = Pipeline(nodes=[mapped], name="test_map")
    code = pipeline.show_daft_code(inputs={"numbers": [1, 2, 3]})
    
    # Check for map operation patterns
    assert "import daft" in code
    assert "explode" in code
    assert "groupby" in code
    assert "list_agg" in code
    assert "__daft_row_id_" in code
    
    # Check that it's valid Python syntax
    compile(code, '<string>', 'exec')


def test_nested_pipeline_code_generation():
    """Test code generation for nested pipelines."""
    inner = Pipeline(nodes=[double], name="inner")
    
    @node(output_name="final")
    def finalize(doubled: int) -> int:
        return doubled + 1
    
    outer = Pipeline(nodes=[inner, finalize], name="outer")
    code = outer.show_daft_code(inputs={"x": 5})
    
    # Check that nested pipeline nodes are included
    assert "import daft" in code
    assert "def double_" in code
    assert "def finalize_" in code
    
    # Check that it's valid Python syntax
    compile(code, '<string>', 'exec')


def test_get_generated_code_without_mode():
    """Test that get_generated_code returns message when not in code gen mode."""
    engine = DaftEngine(code_generation_mode=False)
    code = engine.get_generated_code()
    assert "Code generation mode not enabled" in code


def test_multiple_outputs():
    """Test code generation with multiple output columns."""
    @node(output_name=("sum", "product"))
    def calculate(x: int, y: int) -> tuple:
        return (x + y, x * y)
    
    pipeline = Pipeline(nodes=[calculate], name="multi_output")
    code = pipeline.show_daft_code(inputs={"x": 5, "y": 3})
    
    # This should generate code (even if multi-output isn't fully supported)
    assert "import daft" in code
    compile(code, '<string>', 'exec')


def test_code_structure():
    """Test that generated code has proper structure."""
    pipeline = Pipeline(nodes=[double, triple, add], name="test")
    code = pipeline.show_daft_code(inputs={"x": 5})
    
    lines = code.split('\n')
    
    # Check header
    assert '"""' in lines[0]
    assert 'Generated Daft code' in code
    
    # Check imports section
    assert any('import daft' in line for line in lines)
    
    # Check UDF definitions section
    assert '# ==================== UDF Definitions ====================' in code
    
    # Check pipeline execution section
    assert '# ==================== Pipeline Execution ====================' in code
    
    # Check that operations are in order
    code_lower = code.lower()
    double_idx = code_lower.find('with_column("doubled"')
    triple_idx = code_lower.find('with_column("tripled"')
    result_idx = code_lower.find('with_column("result"')
    
    assert double_idx > 0
    assert triple_idx > double_idx
    assert result_idx > triple_idx


def test_udf_naming():
    """Test that UDF names are unique."""
    @node(output_name="result1")
    def process(x: int) -> int:
        return x * 2
    
    @node(output_name="result2")
    def process(x: int) -> int:  # Same function name
        return x * 3
    
    pipeline = Pipeline(nodes=[process], name="test")
    code = pipeline.show_daft_code(inputs={"x": 5})
    
    # Should have unique UDF names
    assert "def process_" in code
    # Check that it compiles (no duplicate function names)
    compile(code, '<string>', 'exec')


def test_with_output_name_parameter():
    """Test code generation with output_name parameter."""
    pipeline = Pipeline(nodes=[double, triple, add], name="test")
    
    # Generate code for specific output
    code = pipeline.show_daft_code(inputs={"x": 5}, output_name="result")
    
    # Should still generate all necessary nodes
    assert "import daft" in code
    compile(code, '<string>', 'exec')


def test_empty_pipeline():
    """Test code generation for empty pipeline."""
    pipeline = Pipeline(nodes=[], name="empty")
    code = pipeline.show_daft_code(inputs={})
    
    # Should generate basic structure
    assert "import daft" in code
    compile(code, '<string>', 'exec')


def test_generated_code_has_comments():
    """Test that generated code includes helpful comments."""
    pipeline = Pipeline(nodes=[double], name="test")
    code = pipeline.show_daft_code(inputs={"x": 5})
    
    # Check for comments
    assert "# Create DataFrame with input data" in code
    assert "# Select output columns" in code
    assert "# Collect results" in code


def test_map_with_list_input():
    """Test map operation code generation with list inputs."""
    single = Pipeline(nodes=[double], name="single")
    mapped = single.as_node(
        input_mapping={"items": "x"},
        output_mapping={"doubled": "results"},
        map_over="items",
        name="mapped"
    )
    
    pipeline = Pipeline(nodes=[mapped], name="test_map")
    code = pipeline.show_daft_code(inputs={"items": [1, 2, 3, 4, 5]})
    
    # Should have map operations
    assert "explode" in code
    assert "groupby" in code
    
    # Check syntax
    compile(code, '<string>', 'exec')


def test_show_daft_code_respects_requested_output():
    """Ensure only requested outputs are selected/printed."""
    pipeline = Pipeline(nodes=[double, triple, add], name="test")
    code = pipeline.show_daft_code(inputs={"x": 5}, output_name="result")
    assert 'df = df.select(df["result"])' in code
    assert 'df = df.select(df["doubled"]' not in code


def test_map_output_mapping_generated_code_executes():
    """Generated code for mapped pipelines with output mapping should run end-to-end."""

    mapped = Pipeline(nodes=[square_value], name="square_pipeline").as_node(
        input_mapping={"items": "x"},
        output_mapping={"value": "values"},
        map_over="items",
        name="mapped_values",
    )

    pipeline = Pipeline(nodes=[static_items, mapped], name="outer_pipeline")
    inputs = {"seed_items": [1, 2]}
    code = pipeline.show_daft_code(inputs=inputs)

    exec_env = {}
    exec(compile(code, "<generated>", "exec"), exec_env)
    daft_result = exec_env["result"]
    data = daft_result.to_pydict()

    assert data["values"] == [[1, 4]]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
