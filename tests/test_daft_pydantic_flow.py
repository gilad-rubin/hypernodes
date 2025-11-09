"""Test Pydantic models flowing through Daft pipeline as data.

This tests that Pydantic instances returned from nodes can be pickled correctly.
"""

import pytest

pytest.importorskip("daft")
pytest.importorskip("pydantic")

from typing import List
from pydantic import BaseModel
from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine


def test_pydantic_instances_flow_through_pipeline():
    """Test that Pydantic model instances returned from nodes work correctly."""
    
    # Define Pydantic model at module level (simulating script pattern)
    class Item(BaseModel):
        id: str
        value: int
        
        model_config = {"frozen": True}
    
    @node(output_name="items")
    def create_items(count: int) -> List[Item]:
        """Create list of Pydantic items."""
        return [Item(id=f"item_{i}", value=i * 10) for i in range(count)]
    
    @node(output_name="total")
    def sum_values(items: List[Item]) -> int:
        """Sum values from Pydantic items."""
        return sum(item.value for item in items)
    
    # Run pipeline
    pipeline = Pipeline(
        nodes=[create_items, sum_values],
        engine=DaftEngine()
    )
    result = pipeline.run(inputs={"count": 5})
    
    # 0*10 + 1*10 + 2*10 + 3*10 + 4*10 = 0 + 10 + 20 + 30 + 40 = 100
    assert result["total"] == 100


def test_pydantic_instances_with_multiprocessing():
    """Test Pydantic instances with process-based execution (closer to Daft's UDF workers)."""
    import cloudpickle
    from pydantic import BaseModel
    
    # Define at function level to simulate script
    class Document(BaseModel):
        id: str
        text: str
        
    # Create instance
    doc = Document(id="doc1", text="hello world")
    
    # Simulate what happens in Daft UDF worker
    # The instance needs to be pickleable
    try:
        pickled = cloudpickle.dumps(doc)
        unpickled = cloudpickle.loads(pickled)
        assert unpickled.id == "doc1"
        assert unpickled.text == "hello world"
    except Exception as e:
        pytest.fail(f"Failed to pickle/unpickle Pydantic instance: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

