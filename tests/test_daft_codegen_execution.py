"""Test that generated Daft code is executable and produces correct results using real Daft."""

import pytest
import sys
import textwrap

# Skip if daft is not installed
daft = pytest.importorskip("daft")

from hypernodes import Pipeline, node
from hypernodes.integrations.daft.engine import DaftEngine

def test_codegen_execution_correctness_simple():
    """Verify that generated code produces correct results for a simple pipeline."""
    
    @node(output_name="y")
    def add_one(x: int) -> int:
        return x + 1

    pipeline = Pipeline(nodes=[add_one])
    
    # Use list input for Daft compatibility
    inputs = {"x": [1, 2, 3]}
    
    engine = DaftEngine()
    
    # 1. Generate Code
    code = engine.generate_code(pipeline, inputs)
    print("Generated Code:\n", code)
    
    # 2. Execute Code
    exec_globals = {}
    exec(code, exec_globals)
    
    # 3. Verify Result
    assert "df" in exec_globals
    result_df = exec_globals["df"]
    result_data = result_df.to_pydict()
    
    assert "y" in result_data
    assert result_data["y"] == [2, 3, 4]
    
    # 4. Verify Parity with Engine (using map for multi-row)
    # engine.map handles list inputs as multiple rows
    map_results = engine.map(pipeline, inputs, map_over="x")
    # map returns list of dicts: [{'y': 2}, {'y': 3}, ...]
    engine_values = [r["y"] for r in map_results]
    assert engine_values == [2, 3, 4]


def test_codegen_execution_correctness_batch():
    """Verify generated code correctness for batch operations."""
    
    @node(output_name="z")
    def add(x: int, y: int) -> int:
        return x + y

    pipeline = Pipeline(nodes=[add])
    inputs = {"x": [1, 2], "y": [3, 4]}
    
    # Force batch usage
    engine = DaftEngine(use_batch_udf=True)
    
    # 1. Generate Code
    code = engine.generate_code(pipeline, inputs)
    print("Generated Batch Code:\n", code)
    
    # 2. Execute Code
    exec_globals = {}
    exec(code, exec_globals)
    
    # 3. Verify Result
    assert "df" in exec_globals
    result_df = exec_globals["df"]
    result_data = result_df.to_pydict()
    
    assert "z" in result_data
    assert result_data["z"] == [4, 6]


def test_codegen_scalar_input_handling():
    """Test that generated code handles scalar inputs by wrapping them."""
    
    @node(output_name="y")
    def add_one(x: int) -> int:
        return x + 1

    pipeline = Pipeline(nodes=[add_one])
    inputs = {"x": 1} # Scalar input
    
    engine = DaftEngine()
    
    # We need to ensure generate_code produces valid Daft input (lists)
    # The current implementation of generate_code uses repr(inputs).
    # repr({'x': 1}) -> "{'x': 1}". daft.from_pydict({'x': 1}) fails.
    # We expect this test to fail initially, prompting a fix in engine.py
    
    code = engine.generate_code(pipeline, inputs)
    
    exec_globals = {}
    exec(code, exec_globals)
    
    result_df = exec_globals["df"]
    result_data = result_df.to_pydict()
    assert result_data["y"] == [2]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
