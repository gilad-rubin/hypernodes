#!/usr/bin/env python3
"""
Test script that mimics test_modal.py structure to debug serialization issues.

This script defines Pydantic models and nodes at MODULE level (not inside functions)
to exactly reproduce the pattern that fails in test_modal.py.
"""

from typing import List
from pydantic import BaseModel
from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine


# Define Pydantic models at MODULE level (like test_modal.py)
class Item(BaseModel):
    id: str
    value: int
    
    model_config = {"frozen": True}


class Result(BaseModel):
    total: int
    count: int
    
    model_config = {"frozen": True}


# Define nodes at MODULE level (like test_modal.py)
@node(output_name="items")
def create_items(count: int) -> List[Item]:
    """Create list of items."""
    return [Item(id=f"item_{i}", value=i * 10) for i in range(count)]


@node(output_name="total")
def sum_values(items: List[Item]) -> int:
    """Sum values from items."""
    return sum(item.value for item in items)


@node(output_name="result")
def create_result(total: int, count: int) -> Result:
    """Create result object."""
    return Result(total=total, count=count)


# Build pipeline at MODULE level (like test_modal.py)
pipeline = Pipeline(
    nodes=[create_items, sum_values, create_result],
    name="test_pipeline"
)


def main():
    """Run the pipeline with DaftEngine."""
    print("Running pipeline with DaftEngine...")
    print(f"Pipeline has {len(pipeline.nodes)} nodes")
    
    # Create engine and run (like in test_modal.py's run_pipeline function)
    engine = DaftEngine(debug=True)
    pipeline_with_engine = pipeline.with_engine(engine)
    
    print("Executing pipeline...")
    result = pipeline_with_engine.run(inputs={"count": 5})
    
    print(f"Result: {result}")
    assert result["result"].total == 100  # 0+10+20+30+40
    assert result["result"].count == 5
    print("✓ Success!")


if __name__ == "__main__":
    main()

