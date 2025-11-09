#!/usr/bin/env python3
"""Test with from __future__ import annotations - the missing piece!"""

from __future__ import annotations  # THIS IS THE KEY!

from typing import List
from pydantic import BaseModel

from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine


class Item(BaseModel):
    id: str
    value: int
    model_config = {"frozen": True}


@node(output_name="items")
def create_items(count: int) -> List[Item]:
    return [Item(id=f"item_{i}", value=i * 10) for i in range(count)]


@node(output_name="sum")
def sum_items(items: List[Item]) -> int:
    return sum(item.value for item in items)


def main():
    print("Testing with 'from __future__ import annotations'...")
    print(f"Annotation for create_items return: {create_items.__annotations__['return']}")
    print(f"Type: {type(create_items.__annotations__['return'])}")
    
    pipeline = Pipeline(nodes=[create_items, sum_items], name="future_annotations")
    engine = DaftEngine(debug=True)
    pipeline_with_engine = pipeline.with_engine(engine)
    
    print("\nExecuting...")
    result = pipeline_with_engine.run(inputs={"count": 3})
    
    print(f"Result: {result}")
    assert result["sum"] == 30
    print("✓ Success!")


if __name__ == "__main__":
    main()

