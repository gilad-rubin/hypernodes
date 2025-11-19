from typing import Dict

from hypernodes import Pipeline, node
from hypernodes.callbacks import CallbackContext, PipelineCallback
from hypernodes.integrations.daft.engine import DaftEngine
from hypernodes.sequential_engine import SequentialEngine


class EventRecorder(PipelineCallback):
    def __init__(self):
        self.events = []

    def _record(self, event_type: str, **kwargs):
        # Filter out non-deterministic fields like duration, timestamps, context objects
        clean_kwargs = {}
        for k, v in kwargs.items():
            if k in ["duration", "ctx", "error"]:
                continue
            if k == "outputs" and isinstance(v, dict):
                # Normalize outputs if needed, but usually they are deterministic
                clean_kwargs[k] = v
            else:
                clean_kwargs[k] = v

        self.events.append((event_type, clean_kwargs))

    def on_pipeline_start(
        self, pipeline_id: str, inputs: Dict, ctx: CallbackContext
    ) -> None:
        self._record("pipeline_start", pipeline_id=pipeline_id, inputs=inputs)

    def on_pipeline_end(
        self, pipeline_id: str, outputs: Dict, duration: float, ctx: CallbackContext
    ) -> None:
        self._record("pipeline_end", pipeline_id=pipeline_id, outputs=outputs)

    def on_node_start(self, node_id: str, inputs: Dict, ctx: CallbackContext) -> None:
        self._record("node_start", node_id=node_id, inputs=inputs)

    def on_node_end(
        self, node_id: str, outputs: Dict, duration: float, ctx: CallbackContext
    ) -> None:
        # For Daft parity, we might receive _progress_increment in outputs
        # We should normalize this for comparison if Sequential doesn't emit it
        # OR we explicitly check that Daft emits it and Sequential doesn't,
        # but for "parity" in terms of user experience, the key is that the events happen.

        # Let's strip _progress_increment for strict equality check of the *data*
        # but we might want to verify it exists separately.
        clean_outputs = outputs.copy()
        if "_progress_increment" in clean_outputs:
            del clean_outputs["_progress_increment"]

        self._record("node_end", node_id=node_id, outputs=clean_outputs)

    def on_map_start(self, total_items: int, ctx: CallbackContext) -> None:
        self._record("map_start", total_items=total_items)

    def on_map_end(self, duration: float, ctx: CallbackContext) -> None:
        self._record("map_end")


@node(output_name="y")
def add_one(x: int) -> int:
    return x + 1


@node(output_name="z")
def multiply_two(y: int) -> int:
    return y * 2


def test_daft_sequential_parity_run():
    """Test that DaftEngine.run emits same events as SequentialEngine.run"""
    inputs = {"x": 1}

    # Sequential
    seq_recorder = EventRecorder()
    pipeline_seq = Pipeline(nodes=[add_one, multiply_two], callbacks=[seq_recorder])
    SequentialEngine().run(pipeline_seq, inputs)

    # Daft
    daft_recorder = EventRecorder()
    pipeline_daft = Pipeline(nodes=[add_one, multiply_two], callbacks=[daft_recorder])
    DaftEngine().run(pipeline_daft, inputs)

    # Compare events
    seq_events = [
        (e[0], e[1]["node_id"]) for e in seq_recorder.events if "node_id" in e[1]
    ]
    daft_events = [
        (e[0], e[1]["node_id"]) for e in daft_recorder.events if "node_id" in e[1]
    ]

    assert seq_events == daft_events


def test_daft_sequential_parity_map():
    """Test that DaftEngine.map emits compatible events for progress tracking."""
    inputs = {"x": [1, 2, 3]}

    # Sequential
    seq_recorder = EventRecorder()
    pipeline_seq = Pipeline(nodes=[add_one], callbacks=[seq_recorder])
    SequentialEngine().map(pipeline_seq, inputs, map_over="x")

    # Daft
    daft_recorder = EventRecorder()
    pipeline_daft = Pipeline(nodes=[add_one], callbacks=[daft_recorder])
    DaftEngine().map(pipeline_daft, inputs, map_over="x")

    # Verify Map Start/End
    seq_map_start = next(e for e in seq_recorder.events if e[0] == "map_start")
    daft_map_start = next(e for e in daft_recorder.events if e[0] == "map_start")
    assert seq_map_start == daft_map_start

    # Verify Node Execution
    daft_node_starts = [
        e
        for e in daft_recorder.events
        if e[0] == "node_start" and e[1]["node_id"] == "add_one"
    ]
    daft_node_ends = [
        e
        for e in daft_recorder.events
        if e[0] == "node_end" and e[1]["node_id"] == "add_one"
    ]

    assert len(daft_node_starts) == 1
    assert len(daft_node_ends) == 1
