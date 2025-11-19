"""Test for the PyArrow contract in DualNode batch execution.

Verifies that:
1. SeqEngine automatically converts list inputs to PyArrow arrays for DualNodes.
2. Batch functions must receive and return PyArrow arrays.
3. Missing pyarrow dependency raises appropriate errors.
"""

import pytest
from typing import List, Any
import sys

from hypernodes import DualNode, Pipeline, node
from hypernodes.sequential_engine import SeqEngine


# Check if pyarrow is available
try:
    import pyarrow as pa
    import pyarrow.compute as pc
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow not installed")
def test_pyarrow_contract_success():
    """Test that a DualNode expecting PyArrow arrays works with SeqEngine."""
    
    # 1. Define a DualNode that explicitly uses PyArrow
    def double_one(x: int) -> int:
        return x * 2
        
    def double_batch(x: pa.Array) -> pa.Array:
        # Contract: Input is pa.Array, Output must be pa.Array
        # This would FAIL if passed a list (AttributeError: 'list' object has no attribute 'multiply' or similar)
        return pc.multiply(x, 2)
        
    node = DualNode(
        output_name="doubled",
        singular=double_one,
        batch=double_batch
    )
    
    pipeline = Pipeline(nodes=[node])
    
    # 2. Run with SeqEngine (default) via .map()
    # Input is a standard list
    inputs = {"x": [1, 2, 3]}
    
    # This triggers _execute_dual_node_batch in SeqEngine
    results = pipeline.map(inputs, map_over="x")
    
    # 3. Verify results
    assert len(results) == 3
    assert results[0]["doubled"] == 2
    assert results[1]["doubled"] == 4
    assert results[2]["doubled"] == 6


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow not installed")
def test_pyarrow_contract_relaxed_return_type():
    """Test that returning a list is allowed (relaxed contract)."""
    
    def double_one(x: int) -> int:
        return x * 2
        
    def double_batch_list(x: pa.Array) -> List[int]:
        # Returns a list instead of Array - should be allowed now
        return [val.as_py() * 2 for val in x]
        
    node = DualNode(
        output_name="doubled",
        singular=double_one,
        batch=double_batch_list
    )
    
    pipeline = Pipeline(nodes=[node])
    
    # Should succeed now
    results = pipeline.map({"x": [1, 2]}, map_over="x")
    assert results == [{"doubled": 2}, {"doubled": 4}]


def test_missing_pyarrow_error(monkeypatch):
    """Test that missing pyarrow raises helpful ImportError."""
    
    # Simulate missing pyarrow
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    
    def simple_one(x): return x
    def simple_batch(xs): return xs
    
    # Should fail at creation time (DualNode init) OR execution time
    # Current implementation checks at DualNode init
    
    with pytest.raises(ImportError, match="DualNode batch execution requires 'pyarrow'"):
        DualNode(
            output_name="out",
            singular=simple_one,
            batch=simple_batch
        )

