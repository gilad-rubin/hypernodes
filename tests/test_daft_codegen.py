"""Tests for Daft code generation."""

import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock daft if not installed
if "daft" not in sys.modules:
    sys.modules["daft"] = MagicMock()

from hypernodes import Pipeline, node
from hypernodes.integrations.daft import DaftEngine

def test_codegen_simple_pipeline():
    """Test code generation for a simple pipeline."""
    
    @node(output_name="y")
    def add_one(x: int) -> int:
        return x + 1

    pipeline = Pipeline(nodes=[add_one])
    
    # Mock DAFT_AVAILABLE to True for this test
    with patch("hypernodes.integrations.daft.engine.DAFT_AVAILABLE", True):
        engine = DaftEngine()
        code = engine.generate_code(pipeline, inputs={"x": 1})
    
    print(code)
    
    assert "import daft" in code
    assert "@daft.func" in code
    assert "def add_one" in code
    assert '.with_column("y", add_one(daft.col("x")))' in code

def test_codegen_batch_pipeline():
    """Test code generation for batch pipeline."""
    
    @node(output_name="y")
    def add_one(x: int) -> int:
        return x + 1

    pipeline = Pipeline(nodes=[add_one])
    
    with patch("hypernodes.integrations.daft.engine.DAFT_AVAILABLE", True):
        engine = DaftEngine(use_batch_udf=True)
        # Force map context simulation
        engine._is_map_context = True 
        code = engine.generate_code(pipeline, inputs={"x": [1, 2]})
    
    print(code)
    
    assert "@daft.func.batch" in code
    assert "def add_one_batch" in code

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
