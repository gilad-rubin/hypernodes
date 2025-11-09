#!/usr/bin/env python3
"""Benchmark script that uses the optimized DaftEngine implementation."""

from __future__ import annotations

from hypernodes.engines import DaftEngine

from benchmark_stateful_shared import run_benchmark


if __name__ == "__main__":
    run_benchmark(DaftEngine, count=5000, payload_size=8_000_000)
