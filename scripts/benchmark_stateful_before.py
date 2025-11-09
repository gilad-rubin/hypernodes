#!/usr/bin/env python3
"""Benchmark script that simulates the legacy daft.lit behavior."""

from __future__ import annotations

from benchmark_stateful_shared import LegacyStatefulDaftEngine, run_benchmark


if __name__ == "__main__":
    run_benchmark(LegacyStatefulDaftEngine, count=5000, payload_size=8_000_000)
