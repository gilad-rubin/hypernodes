"""Tests for Daft code generation with nested pipelines.

These tests verify that code generation works correctly for:
1. Simple nested pipelines (no map_over)
2. Nested pipelines with map_over
3. Code generation without providing input values
"""

import pytest
from unittest.mock import MagicMock, patch
import sys

# Check if real daft is available
try:
    import daft as real_daft
    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False
    # Mock daft if not installed - must be done before importing modules that use daft
    mock_daft = MagicMock()
    mock_daft.DataType = MagicMock()
    mock_daft.DataType.python = MagicMock(return_value=MagicMock())
    sys.modules["daft"] = mock_daft
    sys.modules["daft.context"] = MagicMock()
    sys.modules["daft.daft"] = MagicMock()
    sys.modules["daft.subscribers"] = MagicMock()
    sys.modules["daft.subscribers.abc"] = MagicMock()

from hypernodes import Pipeline, node

# Import DaftEngine - use mock or real depending on availability
if DAFT_AVAILABLE:
    from hypernodes.integrations.daft.engine import DaftEngine
else:
    with patch("hypernodes.integrations.daft.engine.DAFT_AVAILABLE", True):
        from hypernodes.integrations.daft.engine import DaftEngine


class TestNestedPipelineCodeGen:
    """Tests for nested pipeline code generation."""

    def test_simple_nested_pipeline_codegen(self):
        """Test code generation for a simple nested pipeline (no map_over)."""
        
        @node(output_name="doubled")
        def double(x: int) -> int:
            return x * 2

        @node(output_name="result")
        def add_ten(doubled: int) -> int:
            return doubled + 10

        # Create inner pipeline
        inner = Pipeline(nodes=[double], name="inner")
        
        # Create outer pipeline with nested inner
        outer = Pipeline(nodes=[inner.as_node(), add_ten], name="outer")

        with patch("hypernodes.integrations.daft.engine.DAFT_AVAILABLE", True):
            engine = DaftEngine()
            code = engine.generate_code(outer, inputs={"x": 5})

        print("Generated code:")
        print(code)

        # Verify the code includes the inner pipeline's nodes
        assert "import daft" in code
        assert "doubled" in code, "Inner pipeline output should be in generated code"
        assert "double" in code.lower(), "Inner pipeline function should be in generated code"
        assert "result" in code, "Outer pipeline output should be in generated code"
        # Should not crash with AttributeError

    def test_nested_pipeline_with_map_over_codegen(self):
        """Test code generation for nested pipeline with map_over."""
        
        @node(output_name="doubled")
        def double(x: int) -> int:
            return x * 2

        # Create inner pipeline
        inner = Pipeline(nodes=[double], name="inner")
        
        # Create mapped version
        inner_mapped = inner.as_node(
            input_mapping={"items": "x"},
            output_mapping={"doubled": "results"},
            map_over="items",
            name="inner_mapped"
        )

        @node(output_name="total")
        def sum_results(results: list) -> int:
            return sum(results)

        outer = Pipeline(nodes=[inner_mapped, sum_results], name="outer")

        with patch("hypernodes.integrations.daft.engine.DAFT_AVAILABLE", True):
            engine = DaftEngine()
            code = engine.generate_code(outer, inputs={"items": [1, 2, 3]})

        print("Generated code:")
        print(code)

        # Verify nested pipeline structure is represented
        assert "import daft" in code
        assert "items" in code, "Input parameter should be in generated code"
        assert "explode" in code.lower() or "results" in code, \
            "Map operation should generate explode/groupby pattern"
        # Should not crash with AttributeError

    def test_codegen_without_input_values(self):
        """Test code generation works without providing actual input values."""
        
        @node(output_name="doubled")
        def double(x: int) -> int:
            return x * 2

        @node(output_name="result")
        def add_ten(doubled: int) -> int:
            return doubled + 10

        pipeline = Pipeline(nodes=[double, add_ten], name="simple")

        with patch("hypernodes.integrations.daft.engine.DAFT_AVAILABLE", True):
            engine = DaftEngine()
            # Generate code WITHOUT providing inputs
            code = engine.generate_code(pipeline, inputs=None)

        print("Generated code (no inputs):")
        print(code)

        assert "import daft" in code
        # Should include placeholders or at least the structure
        assert "daft.from_pydict" in code or "df =" in code

    def test_codegen_generates_input_structure_from_root_args(self):
        """Test that code generation creates input dict from pipeline.root_args even without values."""
        
        @node(output_name="result")
        def process(name: str, value: int) -> str:
            return f"{name}: {value}"

        pipeline = Pipeline(nodes=[process], name="simple")

        with patch("hypernodes.integrations.daft.engine.DAFT_AVAILABLE", True):
            engine = DaftEngine()
            # Generate code without inputs - should still show input names
            code = engine.generate_code(pipeline, inputs=None)

        print("Generated code (should have input names):")
        print(code)

        assert "import daft" in code
        # The generated code should reference the input parameters by name
        assert "name" in code or "value" in code, \
            "Generated code should reference input parameter names"


class TestRetrievalLikePattern:
    """Tests that mimic the retrieval pipeline pattern."""

    def test_encode_single_as_mapped_node(self):
        """Test pattern: single-item pipeline wrapped with map_over."""
        
        @node(output_name="encoded")
        def encode(item: str) -> dict:
            return {"text": item, "embedding": [0.1, 0.2]}

        # Single-item pipeline
        encode_single = Pipeline(nodes=[encode], name="encode_single")
        
        # Mapped version
        encode_mapped = encode_single.as_node(
            input_mapping={"items": "item"},
            output_mapping={"encoded": "encoded_items"},
            map_over="items",
            name="encode_mapped"
        )

        @node(output_name="count")
        def count_items(encoded_items: list) -> int:
            return len(encoded_items)

        outer = Pipeline(nodes=[encode_mapped, count_items], name="outer")

        with patch("hypernodes.integrations.daft.engine.DAFT_AVAILABLE", True):
            engine = DaftEngine()
            code = engine.generate_code(outer, inputs={"items": ["a", "b", "c"]})

        print("Generated code (retrieval-like pattern):")
        print(code)

        # Should generate valid code without AttributeError
        assert "import daft" in code
        assert "items" in code

    def test_deeply_nested_pipelines(self):
        """Test deeply nested pipeline structure (pipeline in pipeline in pipeline)."""
        
        @node(output_name="x2")
        def double(x: int) -> int:
            return x * 2

        @node(output_name="x3")
        def triple(x2: int) -> int:
            return x2 * 3

        # Level 1: single node
        level1 = Pipeline(nodes=[double], name="level1")
        
        # Level 2: wraps level1
        level2 = Pipeline(nodes=[level1.as_node(), triple], name="level2")
        
        # Level 3: wraps level2
        level3 = Pipeline(nodes=[level2.as_node()], name="level3")

        with patch("hypernodes.integrations.daft.engine.DAFT_AVAILABLE", True):
            engine = DaftEngine()
            code = engine.generate_code(level3, inputs={"x": 5})

        print("Generated code (deeply nested):")
        print(code)

        # Should not crash and should include all node functions
        assert "import daft" in code
        assert "double" in code.lower()


class TestCodeGenExecutionParity:
    """Parity tests - verify generated code produces same results as DaftEngine."""
    
    @pytest.mark.skipif(not DAFT_AVAILABLE, reason="Requires real Daft installation")
    def test_simple_nested_pipeline_parity(self):
        """Generated code should produce same result as DaftEngine for simple nested pipeline."""
        
        @node(output_name="doubled")
        def double(x: int) -> int:
            return x * 2

        @node(output_name="result")
        def add_ten(doubled: int) -> int:
            return doubled + 10

        inner = Pipeline(nodes=[double], name="inner")
        outer = Pipeline(nodes=[inner.as_node(), add_ten], name="outer")

        from hypernodes.integrations.daft.engine import DaftEngine
        engine = DaftEngine()
        
        inputs = {"x": 5}
        
        # 1. Run with DaftEngine
        engine_result = engine.run(outer, inputs)
        
        # 2. Generate and execute code
        code = engine.generate_code(outer, inputs)
        exec_globals = {}
        exec(code, exec_globals)
        
        result_df = exec_globals["df"]
        code_result = result_df.to_pydict()
        
        # 3. Compare results
        assert engine_result["result"] == code_result["result"][0]
        assert engine_result["doubled"] == code_result["doubled"][0]

    @pytest.mark.skipif(not DAFT_AVAILABLE, reason="Requires real Daft installation")
    def test_nested_pipeline_map_over_parity(self):
        """Generated code should produce same result for nested pipeline with map_over."""
        
        @node(output_name="doubled")
        def double(x: int) -> int:
            return x * 2

        inner = Pipeline(nodes=[double], name="inner")
        inner_mapped = inner.as_node(
            input_mapping={"items": "x"},
            output_mapping={"doubled": "results"},
            map_over="items",
            name="inner_mapped"
        )

        @node(output_name="total")
        def sum_results(results: list) -> int:
            return sum(results)

        outer = Pipeline(nodes=[inner_mapped, sum_results], name="outer")

        from hypernodes.integrations.daft.engine import DaftEngine
        engine = DaftEngine()
        
        inputs = {"items": [1, 2, 3]}
        
        # 1. Run with DaftEngine
        engine_result = engine.run(outer, inputs)
        
        # 2. Generate and execute code
        code = engine.generate_code(outer, inputs)
        print("Generated code for map_over parity:")
        print(code)
        
        exec_globals = {}
        exec(code, exec_globals)
        
        result_df = exec_globals["df"]
        code_result = result_df.to_pydict()
        
        # 3. Compare results
        assert engine_result["total"] == code_result["total"][0]
        # Results should be [2, 4, 6]
        assert engine_result["results"] == code_result["results"][0]

    @pytest.mark.skipif(not DAFT_AVAILABLE, reason="Requires real Daft installation")
    def test_deeply_nested_pipeline_parity(self):
        """Generated code should produce same result for deeply nested pipelines."""
        
        @node(output_name="x2")
        def double(x: int) -> int:
            return x * 2

        @node(output_name="x3")
        def triple(x2: int) -> int:
            return x2 * 3

        level1 = Pipeline(nodes=[double], name="level1")
        level2 = Pipeline(nodes=[level1.as_node(), triple], name="level2")
        level3 = Pipeline(nodes=[level2.as_node()], name="level3")

        from hypernodes.integrations.daft.engine import DaftEngine
        engine = DaftEngine()
        
        inputs = {"x": 5}
        
        # 1. Run with DaftEngine
        engine_result = engine.run(level3, inputs)
        
        # 2. Generate and execute code
        code = engine.generate_code(level3, inputs)
        exec_globals = {}
        exec(code, exec_globals)
        
        result_df = exec_globals["df"]
        code_result = result_df.to_pydict()
        
        # 3. Compare results
        # x=5 -> x2=10 -> x3=30
        assert engine_result["x2"] == code_result["x2"][0]
        assert engine_result["x3"] == code_result["x3"][0]
        assert engine_result["x2"] == 10
        assert engine_result["x3"] == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

