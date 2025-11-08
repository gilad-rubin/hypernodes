# Daft Engine Progress Report

## Scope

This document captures the recent round of DaftEngine work: benchmarking Hypernodes vs Daft, closing the performance gap for stateful workloads, and paving the way for further optimizations.

## Benchmark Harness (`scripts/daft_engine_benchmark.py`)

- Rebuilt the benchmark script to run identical pipelines across:
  - HypernodesEngine (`sequential`, `threaded`, `parallel` map executors).
  - DaftEngine automatic translation.
  - Optional “native Daft” baselines built with handcrafted `@daft.func`, `@daft.cls`, and `@daft.func.batch` UDFs.
- Scenarios covered:
  - Text preprocessing DAG (clean → tokenize → count).
  - Stateful encoder (constant heavy object passed into `.map`).
  - Vectorized numeric normalization (shows batch UDF wins).
  - Nested `.as_node(map_over=...)` pipeline with aggregation.
- CLI options support scaling, selective scenarios, warmup/repeat, JSON export, and toggling native baselines.
- Example commands:
  ```bash
  uv run python scripts/daft_engine_benchmark.py --scale small --scenarios text stateful --repeats 1 --warmup 0
  uv run python scripts/daft_engine_benchmark.py --scale small --scenarios stateful --repeats 1 --warmup 0 --skip-native
  ```
- New `--daft-return-formats` flag runs DaftEngine auto-conversion with multiple output modes (e.g., `python`, `daft`, `arrow`) so we can quantify the benefit of skipping Pythonization.
- New `--daft-python-strategies` flag benchmarks different Python materialization paths (`auto`, `pydict`, `arrow`, `pandas`) whenever `return_format=python`, making it easy to profile conversion trade-offs side-by-side.

## Engine Enhancements (`src/hypernodes/integrations/daft/engine.py`)

1. **Selective Collection**
   - For both `run()` and `map()`, only the requested outputs are collected from the Daft DataFrame, reducing materialization cost when `output_name` filters are used.

2. **Stateful UDF Detection**
   - Recognizes constant inputs that carry `__daft_hint__ = "@daft.cls"` (or `__daft_stateful__`).
   - Removes those columns from the DataFrame, then wraps the node function in a generated `@daft.cls` wrapper so the heavy object initializes once per worker.
   - Handles nested pipelines / `PipelineNode.as_node()` via recursive state propagation.

3. **Columnar `.map()` Fast-Path**
   - New helper `_map_dataframe()` ingests the original list inputs, builds a single Daft DataFrame, and executes the entire pipeline lazily before a single `collect()`.
   - Exposed via `DaftEngine.map_columnar(...)` so `Pipeline.map()` can bypass the list-of-dicts expansion when `map_mode="zip"`.

4. **Result Conversion**
   - All outputs are converted via `_convert_output_value()` as before, but now the conversion only touches the requested columns (matching the selective-collect change above).

5. **Return Format Passthrough**
   - `Pipeline.map(..., return_format=...)` lets callers request `python` (default), `daft`, or `arrow` outputs when DaftEngine handles the columnar fast-path.
   - `return_format="daft"` returns the collected Daft DataFrame/Table directly, matching native baselines by skipping Python list conversion; `"arrow"` surfaces a PyArrow table when the dependency is available.
6. **Python Materialization Strategies**
   - `DaftEngine(python_return_strategy=...)` now exposes `"auto"` (current default), `"pydict"` (legacy `to_pydict()`), `"arrow"` (chunked PyArrow conversion), and `"pandas"` options.
   - `"auto"` currently prefers the proven `to_pydict()` path, while `"arrow"`/`"pandas"` are opt-in for targeted experiments.
   - Strategies can be mixed per benchmark via `--daft-python-strategies` to profile string-heavy vs numeric-heavy pipelines without touching user code.

## Pipeline Adjustments (`src/hypernodes/pipeline.py`)

- `Pipeline.map()` now probes the engine for `map_columnar`. If available (DaftEngine), it passes the varying/fixed inputs column-wise directly to the engine and skips the old “list of execution plans” path.
- Callback bookkeeping was updated so we still fire `on_map_start`/`on_map_end`, and contexts are popped exactly once regardless of which path ran.
- Fallback behavior (per-item expansion) remains unchanged for other engines.
- New `return_format` argument (default `"python"`) propagates to engines with columnar support so we can opt into Daft-native or PyArrow outputs without affecting legacy code.

## Findings

- **Stateful workloads**: The largest slowdown was caused by Hypernodes expanding `.map()` inputs into Python dicts per item. The columnar fast-path plus @daft.cls wrapping dropped the 5k-item benchmark from ~2.4 s to ~0.38 s (DaftEngine now sits close to the threaded Hypernodes executor).
- **Residual gap vs native Daft**: The native script’s 80 ms runtime only measures Daft’s `collect()`. Our automated path must still convert the resulting columns to Python lists to satisfy today’s `Pipeline.map` signature, and that pythonization accounts for the remaining ~0.3 s delta.
- **Parallel Hypernodes**: Running `map_executor="parallel"` is still slower for small Python-heavy workloads because of process spin-up and pickling; keeping sequential/threaded as the baseline is more representative.
- **Daft return formats**: With `return_format="daft"` the automated engine now skips Python list conversion entirely, matching the native baseline (timing focuses purely on Daft’s `collect()`); `"arrow"` offers a middle ground for downstream Arrow consumers.
- **Python conversion profiling**: For text/stateful workloads, the legacy `to_pydict()` path still wins (~35 ms vs ~47 ms for the Arrow strategy over 5 k items). For numeric/vectorized jobs the Arrow path matches or slightly beats `to_pydict()` (~9–10 ms). These measurements come from `--daft-python-strategies auto pydict arrow --warmup 1`.

## Next Exploration Targets

1. **Return-format ergonomics**: Extend the new flag beyond top-level `.map()`—e.g., allow nested `PipelineNode.map_over` calls or `.run()` to request Daft/Arrow passthrough, and consider session-level defaults.
2. **Python strategy heuristics**: Auto-detect numeric/Arrow-friendly outputs so `"auto"` can switch to the Arrow path only when it actually wins, keeping string-heavy DAGs on the `to_pydict()` fast-path.
3. **Batch/vectorization hints**: Extend the hint system with `__daft_hint__="@daft.func.batch"` for numeric nodes or derive it automatically from type hints (e.g., numpy arrays). This would let the converter emit vectorized UDFs for math-heavy DAGs.
4. **Chunked conversion**: If Python outputs remain the default, experiment with chunked `pyarrow` extraction or a thread pool to accelerate `to_pylist()` on very large columns.
5. **Callback coverage**: Ensure node-level callbacks (before/after node execution) can hook into the columnar execution path, not just pipeline-level callbacks.
6. **Documentation & API surface**: Continue documenting `__daft_hint__`, the columnar behavior, and the new return-format options so downstream teams know how to opt in.

## Useful References

- Daft UDF guide: https://docs.daft.ai/en/stable/custom-code/func/
- Daft class-based UDFs: https://docs.daft.ai/en/stable/custom-code/cls/
- Hypernodes Daft docs already in repo: `docs/advanced/daft-engine.md`, `docs/advanced/daft-engine-README.md`

## Relevant Files

- `scripts/daft_engine_benchmark.py`
- `src/hypernodes/integrations/daft/engine.py`
- `src/hypernodes/pipeline.py`
- Existing Daft docs under `docs/advanced/`
