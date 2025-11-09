#!/usr/bin/env python3
"""Test Protocol classes with Daft to see if they cause serialization issues."""

from typing import Protocol, Any, List
from pydantic import BaseModel
from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine


# Define Protocol at module level (like test_modal.py)
class Processor(Protocol):
    """Protocol for processors."""
    def process(self, x: int) -> int: ...


# Define concrete implementation
class ConcreteProcessor:
    def __init__(self, multiplier: int):
        self.multiplier = multiplier
    
    def process(self, x: int) -> int:
        return x * self.multiplier


# Define node that uses Protocol type hint
@node(output_name="result")
def process_value(x: int, processor: Processor) -> int:
    """Process a value using a Processor."""
    return processor.process(x)


def main():
    """Run the pipeline with DaftEngine."""
    print("Running pipeline with Protocol type hints...")
    
    processor = ConcreteProcessor(multiplier=5)
    
    pipeline = Pipeline(nodes=[process_value], name="protocol_test")
    engine = DaftEngine(debug=True)
    pipeline_with_engine = pipeline.with_engine(engine)
    
    print("Executing pipeline...")
    result = pipeline_with_engine.run(inputs={"x": 10, "processor": processor})
    
    print(f"Result: {result}")
    assert result["result"] == 50
    print("✓ Success with Protocol types!")


if __name__ == "__main__":
    main()

