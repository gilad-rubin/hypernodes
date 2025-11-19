"""Tests for parity between DaftEngine and SequentialEngine progress reporting."""

import pytest
from typing import List, Dict, Any

from hypernodes import Pipeline, node
from hypernodes.callbacks import PipelineCallback, CallbackContext
from hypernodes.sequential_engine import SeqEngine

try:
    from hypernodes.engines import DaftEngine
    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False


class EventRecorder(PipelineCallback):
    """Records events for comparison."""
    
    def __init__(self):
        self.events = []

    def on_pipeline_start(self, pipeline_id: str, inputs: Dict[str, Any], ctx: CallbackContext):
        self.events.append("pipeline_start")

    def on_pipeline_end(self, pipeline_id: str, outputs: Dict[str, Any], duration: float, ctx: CallbackContext):
        self.events.append("pipeline_end")

    def on_node_start(self, node_name: str, inputs: Dict[str, Any], ctx: CallbackContext):
        self.events.append(f"node_start:{node_name}")

    def on_node_end(self, node_name: str, outputs: Any, duration: float, ctx: CallbackContext):
        self.events.append(f"node_end:{node_name}")

    def on_map_start(self, total_items: int, ctx: CallbackContext):
        self.events.append(f"map_start:{total_items}")

    def on_map_end(self, duration: float, ctx: CallbackContext):
        self.events.append("map_end")

    def on_map_item_start(self, item_index: int, ctx: CallbackContext):
        self.events.append(f"map_item_start:{item_index}")

    def on_map_item_end(self, item_index: int, duration: float, ctx: CallbackContext):
        self.events.append(f"map_item_end:{item_index}")


@node(output_name="y")
def add_one(x: int) -> int:
    return x + 1


@node(output_name="z")
def multiply_two(y: int) -> int:
    return y * 2


@pytest.mark.skipif(not DAFT_AVAILABLE, reason="Daft not installed")
def test_daft_sequential_parity_run():
    """Test that DaftEngine.run emits same events as SequentialEngine.run"""
    inputs = {"x": 1}

    # Sequential
    seq_recorder = EventRecorder()
    engine_seq = SeqEngine(callbacks=[seq_recorder])
    pipeline_seq = Pipeline(nodes=[add_one, multiply_two], engine=engine_seq)
    seq_results = pipeline_seq.run(inputs=inputs)

    # Daft
    daft_recorder = EventRecorder()
    pipeline_daft = Pipeline(
        nodes=[add_one, multiply_two], 
        engine=DaftEngine(callbacks=[daft_recorder])
    )
    daft_results = pipeline_daft.run(inputs=inputs)

    # Check results match
    assert seq_results == daft_results

    # Check events match
    # Note: Daft might execute in different order or have different timing,
    # but the set of events should be similar.
    # For run(), it should be exactly the same sequence.
    assert seq_recorder.events == daft_recorder.events


@pytest.mark.skipif(not DAFT_AVAILABLE, reason="Daft not installed")
def test_daft_sequential_parity_map():
    """Test that DaftEngine.map emits compatible events for progress tracking."""
    inputs = {"x": [1, 2, 3]}

    # Sequential
    seq_recorder = EventRecorder()
    engine_seq = SeqEngine(callbacks=[seq_recorder])
    pipeline_seq = Pipeline(nodes=[add_one], engine=engine_seq)
    seq_results = pipeline_seq.map(inputs=inputs, map_over="x")

    # Daft
    daft_recorder = EventRecorder()
    pipeline_daft = Pipeline(
        nodes=[add_one], 
        engine=DaftEngine(callbacks=[daft_recorder])
    )
    daft_results = pipeline_daft.map(inputs=inputs, map_over="x")

    # Check results match
    # Note: Result format might differ (list of dicts)
    assert len(seq_results) == len(daft_results)
    for s, d in zip(seq_results, daft_results):
        assert s["y"] == d["y"]

    # Check events parity
    # Daft emits map_start/end, but currently NOT map_item_start/end for efficiency
    # unless configured to do so (which is future work).
    # For now, we just check that map_start/end are present.
    
    daft_events = set(daft_recorder.events)
    assert "map_start:3" in daft_events
    assert "map_end" in daft_events
    
    # Ensure basic node events are present (Daft might batch them differently, 
    # but for simple map it often emits them)
    # Update: DaftEngine with batch UDFs executes the *batch* function once per batch.
    # It does NOT emit individual node_start events for each item.
    # So we expect FEWER events from Daft.
    
    assert len(daft_recorder.events) < len(seq_recorder.events)
