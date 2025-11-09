#!/usr/bin/env python3
"""
Progressive complexity test WITH Modal to find where serialization breaks.

This uses the same Modal setup as test_modal.py to reproduce the actual failure.
"""

import sys
from pathlib import Path
from typing import List, Any, Protocol

import numpy as np
from pydantic import BaseModel

import modal
from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine

# Configuration - default to level 1, can be overridden via --level arg
DEFAULT_LEVEL = 1


# ==================== Modal Setup (like test_modal.py) ====================
app = modal.App("hypernodes-progressive-test")

# Get paths
hypernodes_dir = Path("/Users/giladrubin/python_workspace/hypernodes/src/hypernodes")

# Define Modal image
modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTHONUNBUFFERED": "1", "PYTHONPATH": "/root"})
    .uv_pip_install(
        "numpy",
        "pydantic",
        "pyarrow",
        "daft",
    )
    .add_local_dir(str(hypernodes_dir), remote_path="/root/hypernodes")
)


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


# ==================== Build Pipelines ====================
pipeline_l1 = Pipeline(nodes=[create_items_l1, sum_items_l1], name="level_1")
pipeline_l2 = Pipeline(nodes=[create_data_items_l2, sum_data_items_l2], name="level_2")
pipeline_l3 = Pipeline(nodes=[process_with_stateful_l3], name="level_3")
pipeline_l4 = Pipeline(nodes=[create_nested_l4, sum_nested_l4], name="level_4")
pipeline_l5 = Pipeline(nodes=[create_items_l1, encode_items_l5, sum_encoded_l5], name="level_5")


# ==================== Modal Function (like test_modal.py) ====================
@app.function(image=modal_image, timeout=600)
def run_pipeline(pipeline: Pipeline, inputs: dict, level: int) -> Any:
    """Run pipeline with DaftEngine (like test_modal.py's run_pipeline)."""
    from hypernodes.engines import DaftEngine
    
    print(f"Running level {level} with DaftEngine in Modal...")
    print(f"Inputs: {list(inputs.keys())}")
    
    # Create engine (with debug like in test_modal.py)
    engine = DaftEngine(debug=True)
    pipeline = pipeline.with_engine(engine)
    
    print("Executing pipeline...")
    result = pipeline.run(inputs=inputs)
    
    print(f"Success! Result keys: {list(result.keys())}")
    return result


# ==================== Test Runner ====================
@app.local_entrypoint()
def main(level: int = DEFAULT_LEVEL):
    """Run test at specified complexity level."""
    print(f"Testing complexity level {level} with Modal")
    print("=" * 60)
    
    try:
        print(f"\n{'='*60}")
        print(f"LEVEL {level} - Modal .local() Execution")
        print(f"{'='*60}")
        
        if level == 1:
            print("Testing: Basic Pydantic models")
            pipeline = pipeline_l1
            inputs = {"count": 3}
            expected_key = "sum_l1"
            expected_value = 30
            
        elif level == 2:
            print("Testing: Pydantic with numpy arrays (arbitrary_types_allowed)")
            pipeline = pipeline_l2
            inputs = {"count": 3}
            expected_key = "sum_l2"
            expected_value = 18.0
            
        elif level == 3:
            print("Testing: Protocol type hints + stateful class")
            pipeline = pipeline_l3
            processor = ConcreteProcessor(multiplier=5)
            inputs = {"x": 10, "processor": processor}
            expected_key = "processed_l3"
            expected_value = 50
            
        elif level == 4:
            print("Testing: Nested Pydantic models")
            pipeline = pipeline_l4
            inputs = {"count": 3}
            expected_key = "sum_l4"
            expected_value = 31
            
        elif level == 5:
            print("Testing: Complex pipeline with encoder + Pydantic + numpy")
            pipeline = pipeline_l5
            encoder = Encoder(factor=2)
            inputs = {"count": 3, "encoder": encoder}
            expected_key = "sum_l5"
            expected_value = None  # Dynamic
            
        else:
            print(f"Invalid level: {level}")
            sys.exit(1)
        
        # Run with Modal .local() (like test_modal.py)
        print("\nCalling Modal run_pipeline.local()...")
        result = run_pipeline.local(pipeline, inputs, level)
        
        print(f"\nResult received: {result}")
        print(f"Result['{expected_key}']: {result[expected_key]}")
        
        if expected_value is not None:
            actual = result[expected_key]
            if abs(actual - expected_value) < 0.01:
                print(f"\n✓ Level {level} PASSED with Modal!")
            else:
                print(f"\n✗ Level {level} FAILED: expected {expected_value}, got {actual}")
                sys.exit(1)
        else:
            print(f"\n✓ Level {level} PASSED with Modal!")
        
    except Exception as e:
        print(f"\n✗ Level {level} FAILED with Modal:")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

