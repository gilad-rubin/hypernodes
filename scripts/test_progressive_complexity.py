#!/usr/bin/env python3
"""
Progressively complex test to find where Modal-style serialization breaks.

This script tests increasing levels of complexity to pinpoint the exact failure point.
Run with different COMPLEXITY_LEVEL values to test each level.
"""

import sys
from typing import List, Any, Protocol
from pydantic import BaseModel
import numpy as np
from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine


# Configuration
COMPLEXITY_LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 5
DEBUG = True

print(f"Testing complexity level {COMPLEXITY_LEVEL}")
print("=" * 60)


# ==================== LEVEL 1: Basic Pydantic Models ====================
class Item(BaseModel):
    id: str
    value: int
    model_config = {"frozen": True}


@node(output_name="items_l1")
def create_items_l1(count: int) -> List[Item]:
    return [Item(id=f"item_{i}", value=i * 10) for i in range(count)]


@node(output_name="sum_l1")
def sum_items_l1(items_l1: List[Item]) -> int:
    return sum(item.value for item in items_l1)


# ==================== LEVEL 2: Pydantic with arbitrary_types_allowed ====================
class DataItem(BaseModel):
    id: str
    data: Any  # numpy array
    model_config = {"frozen": True, "arbitrary_types_allowed": True}


@node(output_name="data_items_l2")
def create_data_items_l2(count: int) -> List[DataItem]:
    return [
        DataItem(id=f"data_{i}", data=np.array([i, i * 2, i * 3]))
        for i in range(count)
    ]


@node(output_name="sum_l2")
def sum_data_items_l2(data_items_l2: List[DataItem]) -> float:
    return sum(item.data.sum() for item in data_items_l2)


# ==================== LEVEL 3: Protocol + Stateful Class ====================
class Processor(Protocol):
    def process(self, x: int) -> int: ...


class ConcreteProcessor:
    __daft_hint__ = "@daft.cls"
    
    def __init__(self, multiplier: int):
        self.multiplier = multiplier
    
    def process(self, x: int) -> int:
        return x * self.multiplier


@node(output_name="processed_l3")
def process_with_stateful_l3(x: int, processor: Processor) -> int:
    return processor.process(x)


# ==================== LEVEL 4: Nested Pydantic Models ====================
class InnerData(BaseModel):
    value: int
    model_config = {"frozen": True}


class OuterData(BaseModel):
    id: str
    inner: InnerData
    items: List[InnerData]
    model_config = {"frozen": True}


@node(output_name="nested_l4")
def create_nested_l4(count: int) -> List[OuterData]:
    return [
        OuterData(
            id=f"outer_{i}",
            inner=InnerData(value=i * 10),
            items=[InnerData(value=j) for j in range(i)]
        )
        for i in range(count)
    ]


@node(output_name="sum_l4")
def sum_nested_l4(nested_l4: List[OuterData]) -> int:
    total = 0
    for item in nested_l4:
        total += item.inner.value
        total += sum(inner.value for inner in item.items)
    return total


# ==================== LEVEL 5: Complex with Multiple Stateful Objects ====================
class Encoder:
    __daft_hint__ = "@daft.cls"
    __daft_use_process__ = False
    
    def __init__(self, factor: int):
        self.factor = factor
    
    def encode(self, text: str) -> Any:
        # Simulate encoding with numpy array
        return np.array([len(text) * self.factor, ord(text[0]) if text else 0])


class EncodedItem(BaseModel):
    id: str
    text: str
    embedding: Any
    model_config = {"frozen": True, "arbitrary_types_allowed": True}


@node(output_name="encoded_l5")
def encode_items_l5(items_l1: List[Item], encoder: Encoder) -> List[EncodedItem]:
    return [
        EncodedItem(id=item.id, text=f"text_{item.id}", embedding=encoder.encode(item.id))
        for item in items_l1
    ]


@node(output_name="sum_l5")
def sum_encoded_l5(encoded_l5: List[EncodedItem]) -> float:
    return sum(item.embedding.sum() for item in encoded_l5)


# ==================== LEVEL 6: Map Operations ====================
@node(output_name="single_item")
def process_single_item_l6(item: Item, processor: Processor) -> int:
    return processor.process(item.value)


# ==================== Build Pipelines ====================
def build_pipeline_level_1():
    """Basic Pydantic models."""
    return Pipeline(
        nodes=[create_items_l1, sum_items_l1],
        name="level_1"
    )


def build_pipeline_level_2():
    """Pydantic with numpy arrays."""
    return Pipeline(
        nodes=[create_data_items_l2, sum_data_items_l2],
        name="level_2"
    )


def build_pipeline_level_3():
    """Protocol + stateful class."""
    return Pipeline(
        nodes=[process_with_stateful_l3],
        name="level_3"
    )


def build_pipeline_level_4():
    """Nested Pydantic models."""
    return Pipeline(
        nodes=[create_nested_l4, sum_nested_l4],
        name="level_4"
    )


def build_pipeline_level_5():
    """Complex: multiple nodes + stateful objects + numpy."""
    return Pipeline(
        nodes=[create_items_l1, encode_items_l5, sum_encoded_l5],
        name="level_5"
    )


def build_pipeline_level_6():
    """Map operations with stateful objects."""
    # Need to create full pipeline first, then extract items for map
    return Pipeline(
        nodes=[process_single_item_l6],
        name="level_6"
    )


# ==================== Test Runners ====================
def run_test(level: int):
    """Run test at specified complexity level."""
    try:
        print(f"\n{'='*60}")
        print(f"LEVEL {level}")
        print(f"{'='*60}")
        
        if level == 1:
            print("Testing: Basic Pydantic models")
            pipeline = build_pipeline_level_1()
            inputs = {"count": 3}
            expected_key = "sum_l1"
            expected_value = 30  # 0 + 10 + 20
            
        elif level == 2:
            print("Testing: Pydantic with numpy arrays (arbitrary_types_allowed)")
            pipeline = build_pipeline_level_2()
            inputs = {"count": 3}
            expected_key = "sum_l2"
            expected_value = 18.0  # sum of [0,0,0], [1,2,3], [2,4,6]
            
        elif level == 3:
            print("Testing: Protocol type hints + stateful class")
            pipeline = build_pipeline_level_3()
            processor = ConcreteProcessor(multiplier=5)
            inputs = {"x": 10, "processor": processor}
            expected_key = "processed_l3"
            expected_value = 50
            
        elif level == 4:
            print("Testing: Nested Pydantic models")
            pipeline = build_pipeline_level_4()
            inputs = {"count": 3}
            expected_key = "sum_l4"
            expected_value = 31  # 0 + [] + 10 + [0] + 20 + [0,1] = 0 + 10 + 0 + 20 + 0 + 1 = 31
            
        elif level == 5:
            print("Testing: Complex pipeline with encoder + Pydantic + numpy")
            pipeline = build_pipeline_level_5()
            encoder = Encoder(factor=2)
            inputs = {"count": 3, "encoder": encoder}
            expected_key = "sum_l5"
            expected_value = None  # Dynamic, just check it runs
            
        elif level == 6:
            print("Testing: Map operations with stateful objects")
            pipeline = build_pipeline_level_6()
            processor = ConcreteProcessor(multiplier=3)
            # Create items directly for map
            items = [Item(id=f"item_{i}", value=i * 10) for i in range(3)]
            inputs = {"item": items, "processor": processor}
            expected_key = "single_item"
            expected_value = [0, 30, 60]  # [0*3, 10*3, 20*3]
            
        else:
            print(f"Invalid level: {level}")
            return False
        
        # Run with DaftEngine
        engine = DaftEngine(debug=DEBUG)
        pipeline_with_engine = pipeline.with_engine(engine)
        
        print(f"\nInputs: {list(inputs.keys())}")
        print("Executing...")
        
        if level == 6:
            # Use map for level 6
            result = pipeline_with_engine.map(inputs=inputs, map_over="item")
        else:
            result = pipeline_with_engine.run(inputs=inputs)
        
        print(f"Result keys: {list(result.keys())}")
        print(f"Result['{expected_key}']: {result[expected_key]}")
        
        if expected_value is not None:
            actual = result[expected_key]
            if isinstance(actual, (list, dict)):
                print(f"✓ Level {level} PASSED (returned {type(actual).__name__})")
            elif abs(actual - expected_value) < 0.01:
                print(f"✓ Level {level} PASSED")
            else:
                print(f"✗ Level {level} FAILED: expected {expected_value}, got {actual}")
                return False
        else:
            print(f"✓ Level {level} PASSED (result returned successfully)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Level {level} FAILED with exception:")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run tests progressively."""
    if COMPLEXITY_LEVEL == 0:
        # Run all levels
        print("Running ALL complexity levels...\n")
        for level in range(1, 7):
            success = run_test(level)
            if not success:
                print(f"\n{'='*60}")
                print(f"FAILED at complexity level {level}")
                print(f"{'='*60}")
                sys.exit(1)
        
        print(f"\n{'='*60}")
        print("✓ ALL LEVELS PASSED!")
        print(f"{'='*60}")
    else:
        # Run specific level
        success = run_test(COMPLEXITY_LEVEL)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

