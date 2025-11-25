"""Tests for Daft code generation."""

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

# Import DaftEngine - use real or mock depending on availability
if DAFT_AVAILABLE:
    from hypernodes.integrations.daft.engine import DaftEngine
else:
    with patch("hypernodes.integrations.daft.engine.DAFT_AVAILABLE", True):
        from hypernodes.integrations.daft.engine import DaftEngine

def test_codegen_simple_pipeline():
    """Test code generation for a simple pipeline."""
    
    @node(output_name="y")
    def add_one(x: int) -> int:
        return x + 1

    pipeline = Pipeline(nodes=[add_one])
    
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
    
    engine = DaftEngine(use_batch_udf=True)
    # Force map context simulation
    engine._is_map_context = True 
    code = engine.generate_code(pipeline, inputs={"x": [1, 2]}, mode="map")
    
    print(code)
    
    assert "@daft.func.batch" in code
    assert "def add_one_batch" in code

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
