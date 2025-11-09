#!/usr/bin/env python3
"""
Shared utilities for benchmarking Daft stateful input handling.

The before/after scripts import this module to run the same HyperNodes pipeline
with different Daft engine behaviors.
"""

from __future__ import annotations

import time
from typing import List, Sequence, Type

import daft

from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine


@daft.cls(use_process=False)
class LargeStatefulObject:
    """Stateful object with a sizable payload to emphasize serialization cost."""

    def __init__(self, payload_size: int = 5_000_000):
        # Allocate a bytearray (~5 MB by default) to mimic model weights.
        self._payload = bytearray(payload_size)
        self._scale = len(self._payload) % 1024 or 1

    def apply(self, value: int) -> int:
        """Simple transformation that depends on the payload."""
        return (value * self._scale) % 1_000_000

    def __getstate__(self):
        """Simulate expensive serialization like a large ML model."""
        return {"payload": bytes(self._payload), "scale": self._scale}

    def __setstate__(self, state):
        self._payload = bytearray(state["payload"])
        self._scale = state["scale"]


@node(output_name="numbers")
def build_numbers(count: int) -> List[int]:
    """Generate a deterministic list of integers."""
    return list(range(count))


@node(output_name="result")
def apply_model(number: int, model: LargeStatefulObject) -> int:
    """Apply the stateful model to a single number."""
    return model.apply(number)


_single_number_pipeline = Pipeline(nodes=[apply_model], name="apply_model")
_mapped_numbers = _single_number_pipeline.as_node(
    input_mapping={"numbers": "number"},
    output_mapping={"result": "processed_numbers"},
    map_over="numbers",
    name="map_numbers",
)
STATEFUL_BENCHMARK_PIPELINE = Pipeline(
    nodes=[build_numbers, _mapped_numbers], name="stateful_benchmark"
)


class LegacyStatefulDaftEngine(DaftEngine):
    """DaftEngine variant that disables stateful auto-detection (simulates 'before')."""

    def _should_capture_stateful_input(self, value):  # type: ignore[override]
        # Force the old behavior: always treat inputs as regular columns.
        return False


def _format_runs(runs: Sequence[float]) -> str:
    return ", ".join(f"{run:.4f}" for run in runs)


def run_benchmark(
    engine_cls: Type[DaftEngine],
    *,
    count: int = 2000,
    payload_size: int = 5_000_000,
    repeats: int = 3,
) -> float:
    """Execute the benchmark pipeline with the provided engine class."""

    engine = engine_cls()
    pipeline = STATEFUL_BENCHMARK_PIPELINE.with_engine(engine)
    model = LargeStatefulObject(payload_size=payload_size)
    inputs = {"count": count, "model": model}

    durations: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        pipeline.run(inputs=inputs, output_name="processed_numbers")
        durations.append(time.perf_counter() - start)

    average = sum(durations) / len(durations)
    print(
        f"{engine_cls.__name__}: avg {average:.4f}s "
        f"(runs: {_format_runs(durations)}) for {count} items"
    )
    return average


__all__ = [
    "run_benchmark",
    "LegacyStatefulDaftEngine",
    "LargeStatefulObject",
    "STATEFUL_BENCHMARK_PIPELINE",
]
