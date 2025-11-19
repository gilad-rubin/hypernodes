"""Tests for DualNode - nodes with both singular and batch implementations."""

import pytest

try:
    import pyarrow as pa
    import pyarrow.compute as pc
except ImportError:
    pass

from dataclasses import dataclass
from typing import List

from hypernodes import DualNode, Pipeline

# Check if DaftEngine is available
try:
    from hypernodes.engines import DaftEngine

    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False


# ============================================================================
# Test Data Models
# ============================================================================


@dataclass
class Item:
    """Simple test item."""

    value: int


@dataclass
class ProcessedItem:
    """Processed item with doubled value."""

    original: int
    doubled: int


# ============================================================================
# Test: Basic Stateless DualNode
# ============================================================================


def test_dual_node_creation():
    """Test creating a basic DualNode."""

    def double_singular(x: int) -> int:
        return x * 2

    def double_batch(x: pa.Array) -> pa.Array:
        return pc.multiply(x, 2)

    node = DualNode(
        output_name="doubled",
        singular=double_singular,
        batch=double_batch,
    )

    assert node.output_name == "doubled"
    assert node.singular == double_singular
    assert node.batch == double_batch
    assert "x" in node.root_args


def test_dual_node_singular_execution():
    """Test DualNode uses singular function for .run()."""

    call_log = []

    def add_singular(x: int, y: int) -> int:
        call_log.append(("singular", x, y))
        return x + y

    def add_batch(x: pa.Array, y: pa.Array) -> pa.Array:
        call_log.append(("batch", x, y))
        return pc.add(x, y)

    node = DualNode(
        output_name="sum",
        singular=add_singular,
        batch=add_batch,
    )

    pipeline = Pipeline(nodes=[node])
    result = pipeline.run(inputs={"x": 5, "y": 3})

    assert result["sum"] == 8
    assert len(call_log) == 1
    assert call_log[0][0] == "singular"  # Used singular function


def test_dual_node_sequential_map():
    """Test DualNode with SeqEngine uses batch optimization for single DualNode pipelines."""

    call_log = []

    def multiply_singular(x: int, factor: int) -> int:
        call_log.append(("singular", x, factor))
        return x * factor

    def multiply_batch(x: pa.Array, factor: int) -> pa.Array:
        call_log.append(("batch", x, factor))
        return pc.multiply(x, factor)

    node = DualNode(
        output_name="result",
        singular=multiply_singular,
        batch=multiply_batch,
    )

    pipeline = Pipeline(nodes=[node])
    results = pipeline.map(inputs={"x": [1, 2, 3], "factor": 10}, map_over="x")

    assert results == [{"result": 10}, {"result": 20}, {"result": 30}]
    # SeqEngine now optimizes: calls batch once for single-DualNode pipelines
    assert len(call_log) == 1
    assert call_log[0][0] == "batch"


@pytest.mark.skipif(not DAFT_AVAILABLE, reason="Daft not installed")
def test_dual_node_daft_map():
    """Test DualNode with DaftEngine uses batch function."""

    call_log = []

    def multiply_singular(x: int, factor: int) -> int:
        call_log.append(("singular", x, factor))
        return x * factor

    def multiply_batch(x: pa.Array, factor: int) -> pa.Array:
        call_log.append(("batch", x, factor))
        return pc.multiply(x, factor)

    node = DualNode(
        output_name="result",
        singular=multiply_singular,
        batch=multiply_batch,
    )

    pipeline = Pipeline(nodes=[node], engine=DaftEngine())
    results = pipeline.map(inputs={"x": [1, 2, 3], "factor": 10}, map_over="x")

    assert results == [{"result": 10}, {"result": 20}, {"result": 30}]
    # DaftEngine calls batch once (not singular)
    assert len(call_log) == 1
    assert call_log[0][0] == "batch"
    # Batch received all values as list/array
    # Note: The value comparison depends on the engine's internal conversion
    # but it should contain 1, 2, 3
    batch_arg = call_log[0][1]
    if hasattr(batch_arg, "tolist"):
        assert batch_arg.tolist() == [1, 2, 3]
    elif hasattr(batch_arg, "to_pylist"):
        assert batch_arg.to_pylist() == [1, 2, 3]
    else:
        assert list(batch_arg) == [1, 2, 3]


# ============================================================================
# Test: Stateful DualNode
# ============================================================================


def test_stateful_dual_node():
    """Test DualNode with stateful class methods."""

    class Processor:
        """Stateful processor with initialization."""

        def __init__(self, multiplier: int):
            self.multiplier = multiplier
            self.init_count = 0
            self.init_count += 1

        def process_singular(self, x: int) -> int:
            return x * self.multiplier

        def process_batch(self, x: pa.Array) -> pa.Array:
            return pc.multiply(x, self.multiplier)

    # Create instance
    processor = Processor(multiplier=5)

    # Create DualNode from bound methods
    node = DualNode(
        output_name="processed",
        singular=processor.process_singular,
        batch=processor.process_batch,
    )

    pipeline = Pipeline(nodes=[node])

    # Test singular
    result = pipeline.run(inputs={"x": 3})
    assert result["processed"] == 15

    # Test batch
    results = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
    assert results == [{"processed": 5}, {"processed": 10}, {"processed": 15}]

    # Instance initialized only once
    assert processor.init_count == 1


# ============================================================================
# Test: Singular Wraps Batch Pattern
# ============================================================================


def test_singular_wraps_batch_pattern():
    """Test the pattern where singular calls batch internally."""

    class Encoder:
        """Encoder where singular delegates to batch."""

        def __init__(self, dim: int):
            self.dim = dim

        def encode_singular(self, text: str) -> List[float]:
            """Singular wraps batch - single source of truth!"""
            return self.encode_batch([text])[0]

        def encode_batch(self, text: pa.Array) -> List[List[float]]:
            """Real implementation - batch only."""
            # Use Arrow compute to calculate lengths
            lengths = pc.utf8_length(text)
            # Convert to numpy/list for list comprehension structure construction
            # Since we're returning complex nested structure, list is fine
            len_list = lengths.to_pylist()
            return [[float(l)] * self.dim for l in len_list]

    encoder = Encoder(dim=3)

    node = DualNode(
        output_name="embedding",
        singular=encoder.encode_singular,
        batch=encoder.encode_batch,
    )

    pipeline = Pipeline(nodes=[node])

    # Singular execution
    result = pipeline.run(inputs={"text": "hello"})
    assert result["embedding"] == [5.0, 5.0, 5.0]

    # Batch execution
    results = pipeline.map(inputs={"text": ["hi", "bye", "test"]}, map_over="text")
    assert results[0]["embedding"] == [2.0, 2.0, 2.0]  # "hi" = 2 chars
    assert results[1]["embedding"] == [3.0, 3.0, 3.0]  # "bye" = 3 chars
    assert results[2]["embedding"] == [4.0, 4.0, 4.0]  # "test" = 4 chars


# ============================================================================
# Test: Engine Consistency
# ============================================================================


@pytest.mark.skipif(not DAFT_AVAILABLE, reason="Daft not installed")
def test_dual_node_engine_consistency():
    """Test both engines produce identical results."""

    def add_singular(x: int, y: int) -> int:
        return x + y

    def add_batch(x: pa.Array, y: pa.Array) -> pa.Array:
        return pc.add(x, y)

    node = DualNode(
        output_name="sum",
        singular=add_singular,
        batch=add_batch,
    )

    inputs = {"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]}

    # SeqEngine
    pipeline_seq = Pipeline(nodes=[node])
    results_seq = pipeline_seq.map(inputs=inputs, map_over=["x", "y"])

    # DaftEngine
    pipeline_daft = Pipeline(nodes=[node], engine=DaftEngine())
    results_daft = pipeline_daft.map(inputs=inputs, map_over=["x", "y"])

    # Both should produce identical results
    assert results_seq == results_daft
    assert results_seq == [
        {"sum": 11},
        {"sum": 22},
        {"sum": 33},
        {"sum": 44},
        {"sum": 55},
    ]


# ============================================================================
# Test: Multiple DualNodes in Pipeline
# ============================================================================


def test_multiple_dual_nodes():
    """Test pipeline with multiple DualNodes chained together."""

    # Node 1: Double the input
    def double_singular(x: int) -> int:
        return x * 2

    def double_batch(x: pa.Array) -> pa.Array:
        return pc.multiply(x, 2)

    double_node = DualNode(
        output_name="doubled",
        singular=double_singular,
        batch=double_batch,
    )

    # Node 2: Add 10
    def add_ten_singular(doubled: int) -> int:
        return doubled + 10

    def add_ten_batch(doubled: pa.Array) -> pa.Array:
        return pc.add(doubled, 10)

    add_node = DualNode(
        output_name="result",
        singular=add_ten_singular,
        batch=add_ten_batch,
    )

    # Chain them
    pipeline = Pipeline(nodes=[double_node, add_node])

    # Test singular
    result = pipeline.run(inputs={"x": 5})
    assert result["result"] == 20  # (5 * 2) + 10

    # Test batch
    results = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
    assert results == [
        {"doubled": 2, "result": 12},  # (1 * 2) + 10
        {"doubled": 4, "result": 14},  # (2 * 2) + 10
        {"doubled": 6, "result": 16},  # (3 * 2) + 10
    ]


# ============================================================================
# Test: Edge Cases
# ============================================================================


def test_dual_node_empty_batch():
    """Test DualNode with empty batch."""

    def process_singular(x: int) -> int:
        return x * 2

    def process_batch(x: pa.Array) -> pa.Array:
        return pc.multiply(x, 2)

    node = DualNode(
        output_name="result",
        singular=process_singular,
        batch=process_batch,
    )

    pipeline = Pipeline(nodes=[node])

    # Empty batch
    results = pipeline.map(inputs={"x": []}, map_over="x")
    assert results == []


def test_dual_node_single_item_batch():
    """Test DualNode with batch of size 1."""

    call_log = []

    def process_singular(x: int) -> int:
        call_log.append("singular")
        return x * 2

    def process_batch(x: pa.Array) -> pa.Array:
        call_log.append("batch")
        return pc.multiply(x, 2)

    node = DualNode(
        output_name="result",
        singular=process_singular,
        batch=process_batch,
    )

    pipeline = Pipeline(nodes=[node])

    # Single item batch: SeqEngine now optimizes and uses batch
    results = pipeline.map(inputs={"x": [5]}, map_over="x")
    assert results == [{"result": 10}]
    # SeqEngine optimization: uses batch even for single item
    assert "batch" in call_log
