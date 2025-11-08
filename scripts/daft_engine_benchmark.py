#!/usr/bin/env python3
"""Benchmark HypernodesEngine vs DaftEngine (and native Daft baselines).

This script recreates the historical notebook benchmarks in a repeatable CLI form.
It focuses on the scenarios that surfaced in the legacy `daft_benchmarks.ipynb`:
1. Text preprocessing pipelines with .map()
2. Stateful encoders that benefit from @daft.cls
3. Vectorized numeric operations that shine with @daft.func.batch
4. Nested pipelines that rely on `.as_node(map_over=...)`

For each scenario we measure:
    • HypernodesEngine with threaded executors (current baseline)
    • DaftEngine automatic conversion (target integration)
    • Optional native Daft implementation that uses next-gen UDF APIs

Run with:
    uv run python scripts/daft_engine_benchmark.py --scale medium --repeats 5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from hypernodes import Pipeline, node
from hypernodes.engines import HypernodesEngine

try:
    from hypernodes.engines import DaftEngine
except ImportError:  # pragma: no cover - optional dependency
    DaftEngine = None

try:
    import daft
    from daft import DataType, Series
except ImportError:  # pragma: no cover - optional dependency
    daft = None
    DataType = None  # type: ignore
    Series = None  # type: ignore

PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None


SCALE_TO_ITEMS = {
    "tiny": 200,
    "small": 5_000,
    "medium": 25_000,
    "large": 60_000,
}


@dataclass
class BackendStats:
    """Holds timing information for a single backend run."""

    label: str
    durations: List[float]
    note: str = ""
    success: bool = True

    @property
    def mean(self) -> float:
        return sum(self.durations) / len(self.durations) if self.durations else math.nan

    @property
    def best(self) -> float:
        return min(self.durations) if self.durations else math.nan

    @property
    def stdev(self) -> float:
        if len(self.durations) < 2:
            return 0.0
        return statistics.pstdev(self.durations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "durations": self.durations,
            "mean": self.mean,
            "best": self.best,
            "stdev": self.stdev,
            "note": self.note,
            "success": self.success,
        }


@dataclass
class ScenarioResult:
    name: str
    description: str
    n_items: int
    metrics: Dict[str, BackendStats] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "n_items": self.n_items,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "notes": self.notes,
        }


@dataclass
class BenchmarkContext:
    n_items: int
    repeats: int
    warmup: int
    include_native: bool
    daft_return_formats: List[str]
    daft_python_strategies: List[str]


def _daft_metric_key(
    return_format: str, python_strategy: Optional[str]
) -> str:
    if return_format == "python":
        strategy_suffix = python_strategy or "auto"
        return f"daft_engine_python_{strategy_suffix}"
    return f"daft_engine_{return_format}"


def _daft_label(return_format: str, python_strategy: Optional[str]) -> str:
    parts = [f"return={return_format}"]
    if return_format == "python":
        parts.append(f"strategy={python_strategy or 'auto'}")
    suffix = ", ".join(parts)
    return f"DaftEngine({suffix})"


def _daft_note(
    base: str, return_format: str, python_strategy: Optional[str]
) -> str:
    details = [f"return_format={return_format}"]
    if return_format == "python":
        details.append(f"python_strategy={python_strategy or 'auto'}")
    return f"{base} [{' | '.join(details)}]"


def _iter_daft_variants(
    ctx: BenchmarkContext,
) -> List[Tuple[str, Optional[str]]]:
    variants: List[Tuple[str, Optional[str]]] = []
    for fmt in ctx.daft_return_formats:
        if fmt == "python":
            for strategy in ctx.daft_python_strategies:
                variants.append((fmt, strategy))
        else:
            variants.append((fmt, None))
    return variants


def format_seconds(value: float) -> str:
    if math.isnan(value):
        return "--"
    if value < 1.0:
        return f"{value * 1_000:.1f} ms"
    return f"{value:.3f} s"


def benchmark_callable(
    fn: Callable[[], Any], ctx: BenchmarkContext
) -> List[float]:
    """Benchmark helper with warmup."""
    for _ in range(ctx.warmup):
        fn()
    durations: List[float] = []
    for _ in range(ctx.repeats):
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)
    return durations


def unavailable_backend(label: str, reason: str) -> BackendStats:
    return BackendStats(label=label, durations=[], note=reason, success=False)


# ---------------------------------------------------------------------------
# Scenario 1: Text preprocessing pipeline with map()
# ---------------------------------------------------------------------------
def _benchmark_hn_map_variants(
    name: str,
    make_pipeline: Callable[[], Pipeline],
    inputs: Dict[str, Any],
    map_over: str,
    ctx: BenchmarkContext,
) -> Dict[str, BackendStats]:
    """Benchmark HypernodesEngine map with multiple executor variants."""

    variants = [
        ("hypernodes_seq", "sequential", "sequential"),
        ("hypernodes_threaded", "threaded", "threaded"),
        ("hypernodes_parallel", "threaded", "parallel"),
    ]
    metrics: Dict[str, BackendStats] = {}

    for label, node_exec, map_exec in variants:
        pipeline = make_pipeline().with_engine(
            HypernodesEngine(node_executor=node_exec, map_executor=map_exec)
        )
        durations = benchmark_callable(
            lambda: pipeline.map(inputs=inputs, map_over=map_over),
            ctx,
        )
        metrics[label] = BackendStats(
            label=f"HypernodesEngine({label.split('_')[-1]})",
            durations=durations,
            note=f"node={node_exec}, map={map_exec}",
        )
    return metrics


def scenario_text_preprocessing(ctx: BenchmarkContext) -> ScenarioResult:
    texts = [f"  Hello World {i}  " for i in range(ctx.n_items)]

    def build_pipeline() -> Pipeline:
        @node(output_name="cleaned_text")
        def clean_text(text: str) -> str:
            return text.strip().lower()

        @node(output_name="tokens")
        def tokenize(cleaned_text: str) -> List[str]:
            return cleaned_text.split()

        @node(output_name="token_count")
        def token_count(tokens: List[str]) -> int:
            return len(tokens)

        return Pipeline(
            nodes=[clean_text, tokenize, token_count],
            name="text_preprocessing",
        )

    metrics: Dict[str, BackendStats] = {}

    metrics.update(
        _benchmark_hn_map_variants(
            name="text_preprocessing",
            make_pipeline=build_pipeline,
            inputs={"text": texts},
            map_over="text",
            ctx=ctx,
        )
    )

    # DaftEngine auto conversion
    if DaftEngine is None:
        for fmt, strategy in _iter_daft_variants(ctx):
            metrics[_daft_metric_key(fmt, strategy)] = unavailable_backend(
                _daft_label(fmt, strategy), "Daft not installed"
            )
    else:
        for fmt, strategy in _iter_daft_variants(ctx):
            if fmt == "arrow" and not PYARROW_AVAILABLE:
                metrics[_daft_metric_key(fmt, strategy)] = unavailable_backend(
                    _daft_label(fmt, strategy), "pyarrow not installed"
                )
                continue
            if fmt == "python" and strategy == "pandas" and not PANDAS_AVAILABLE:
                metrics[_daft_metric_key(fmt, strategy)] = unavailable_backend(
                    _daft_label(fmt, strategy), "pandas not installed"
                )
                continue
            if fmt == "python" and strategy == "arrow" and not PYARROW_AVAILABLE:
                metrics[_daft_metric_key(fmt, strategy)] = unavailable_backend(
                    _daft_label(fmt, strategy), "pyarrow not installed"
                )
                continue

            pipeline_daft = build_pipeline().with_engine(
                DaftEngine(python_return_strategy=strategy or "auto")
            )
            daft_durations = benchmark_callable(
                lambda pipeline=pipeline_daft, fmt=fmt: pipeline.map(
                    inputs={"text": texts},
                    map_over="text",
                    return_format=fmt,
                ),
                ctx,
            )
            metrics[_daft_metric_key(fmt, strategy)] = BackendStats(
                label=_daft_label(fmt, strategy),
                durations=daft_durations,
                note=_daft_note(
                    "automatic conversion to Daft DataFrame", fmt, strategy
                ),
            )

    # Native Daft baseline using explicit UDFs
    if ctx.include_native and daft is not None:
        @daft.func(return_dtype=DataType.string())  # type: ignore[misc]
        def strip_lower(text: str) -> str:
            return text.strip().lower()

        @daft.func(return_dtype=DataType.list(DataType.string()))  # type: ignore[misc]
        def tokenize_udf(text: str) -> List[str]:
            return text.split()

        @daft.func(return_dtype=DataType.int64())  # type: ignore[misc]
        def count_udf(tokens: List[str]) -> int:
            return len(tokens)

        def run_native() -> None:
            df = daft.from_pydict({"text": texts})
            df = df.with_column("cleaned_text", strip_lower(df["text"]))
            df = df.with_column("tokens", tokenize_udf(df["cleaned_text"]))
            df = df.with_column("token_count", count_udf(df["tokens"]))
            df.collect()

        native_durations = benchmark_callable(run_native, ctx)
        metrics["native_daft"] = BackendStats(
            label="Daft native (@daft.func)",
            durations=native_durations,
            note="manual UDF wiring (ideal target)",
        )

    return ScenarioResult(
        name="text_preprocessing",
        description="Clean/tokenize/count with .map() over ~text corpus",
        n_items=len(texts),
        metrics=metrics,
        notes=[],
    )


# ---------------------------------------------------------------------------
# Scenario 2: Stateful encoder that should use @daft.cls
# ---------------------------------------------------------------------------
class SimpleEncoder:
    """Simulates an embedding model with heavy initialization."""

    # Hint to future DaftEngine heuristics that this class benefits from @daft.cls
    __daft_hint__ = "@daft.cls"

    def __init__(self, dim: int = 128, seed: int = 42, init_delay: float = 0.05):
        time.sleep(init_delay)
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    def encode(self, text: str) -> List[float]:
        vector = self.rng.random(self.dim, dtype=np.float32)
        return vector.tolist()


def scenario_stateful_encoder(ctx: BenchmarkContext) -> ScenarioResult:
    n = min(ctx.n_items, 5_000)
    texts = [f"document_{i}" for i in range(n)]
    shared_encoder = SimpleEncoder(dim=128, seed=7)

    @node(output_name="embedding")
    def encode_text(text: str, encoder: SimpleEncoder) -> List[float]:
        return encoder.encode(text)

    metrics: Dict[str, BackendStats] = {}

    metrics.update(
        _benchmark_hn_map_variants(
            name="stateful_encoder",
            make_pipeline=lambda: Pipeline(nodes=[encode_text], name="stateful_encoder"),
            inputs={"text": texts, "encoder": shared_encoder},
            map_over="text",
            ctx=ctx,
        )
    )

    if DaftEngine is None:
        for fmt, strategy in _iter_daft_variants(ctx):
            metrics[_daft_metric_key(fmt, strategy)] = unavailable_backend(
                _daft_label(fmt, strategy), "Daft not installed"
            )
    else:
        for fmt, strategy in _iter_daft_variants(ctx):
            if fmt == "arrow" and not PYARROW_AVAILABLE:
                metrics[_daft_metric_key(fmt, strategy)] = unavailable_backend(
                    _daft_label(fmt, strategy), "pyarrow not installed"
                )
                continue
            if fmt == "python" and strategy == "pandas" and not PANDAS_AVAILABLE:
                metrics[_daft_metric_key(fmt, strategy)] = unavailable_backend(
                    _daft_label(fmt, strategy), "pandas not installed"
                )
                continue
            if fmt == "python" and strategy == "arrow" and not PYARROW_AVAILABLE:
                metrics[_daft_metric_key(fmt, strategy)] = unavailable_backend(
                    _daft_label(fmt, strategy), "pyarrow not installed"
                )
                continue

            pipeline_daft = Pipeline(
                nodes=[encode_text], name="stateful_encoder"
            ).with_engine(DaftEngine(python_return_strategy=strategy or "auto"))
            daft_durations = benchmark_callable(
                lambda pipeline=pipeline_daft, fmt=fmt: pipeline.map(
                    inputs={"text": texts, "encoder": shared_encoder},
                    map_over="text",
                    return_format=fmt,
                ),
                ctx,
            )
            metrics[_daft_metric_key(fmt, strategy)] = BackendStats(
                label=_daft_label(fmt, strategy),
                durations=daft_durations,
                note=_daft_note(
                    "auto-converts encoder inputs to @daft.cls stateful UDFs",
                    fmt,
                    strategy,
                ),
            )

    if ctx.include_native and daft is not None and DataType is not None:
        @daft.cls  # type: ignore[misc]
        class EncoderUDF:
            def __init__(self, dim: int = 128, seed: int = 7):
                self.encoder = SimpleEncoder(dim=dim, seed=seed, init_delay=0.05)

            @daft.method(return_dtype=DataType.python())  # type: ignore[attr-defined]
            def encode(self, text: str) -> List[float]:
                return self.encoder.encode(text)

        encoder_udf = EncoderUDF(dim=128, seed=7)

        def run_native() -> None:
            df = daft.from_pydict({"text": texts})
            df = df.with_column("embedding", encoder_udf.encode(df["text"]))
            df.collect()

        native_durations = benchmark_callable(run_native, ctx)
        metrics["native_daft"] = BackendStats(
            label="Daft native (@daft.cls)",
            durations=native_durations,
            note="stateful UDF initializes encoder once per worker",
        )

    return ScenarioResult(
        name="stateful_encoder",
        description="Map heavy encoder over documents (shows @daft.cls gains)",
        n_items=n,
        metrics=metrics,
        notes=[
            "Gap here indicates need for automatic detection of stateful nodes.",
            "SimpleEncoder.__daft_hint__='@daft.cls' documents the desired behavior.",
        ],
    )


# ---------------------------------------------------------------------------
# Scenario 3: Vectorized numeric ops -> @daft.func.batch
# ---------------------------------------------------------------------------
def scenario_vectorized_numeric(ctx: BenchmarkContext) -> ScenarioResult:
    values = list(np.linspace(0.0, 10_000.0, ctx.n_items))
    mean_val = 5_000.0
    std_val = 750.0

    @node(output_name="normalized")
    def normalize(value: float, mean: float, std: float) -> float:
        return (value - mean) / std

    metrics: Dict[str, BackendStats] = {}

    metrics.update(
        _benchmark_hn_map_variants(
            name="normalize_numeric",
            make_pipeline=lambda: Pipeline(nodes=[normalize], name="normalize_numeric"),
            inputs={"value": values, "mean": mean_val, "std": std_val},
            map_over="value",
            ctx=ctx,
        )
    )

    if DaftEngine is None:
        for fmt, strategy in _iter_daft_variants(ctx):
            metrics[_daft_metric_key(fmt, strategy)] = unavailable_backend(
                _daft_label(fmt, strategy), "Daft not installed"
            )
    else:
        for fmt, strategy in _iter_daft_variants(ctx):
            if fmt == "arrow" and not PYARROW_AVAILABLE:
                metrics[_daft_metric_key(fmt, strategy)] = unavailable_backend(
                    _daft_label(fmt, strategy), "pyarrow not installed"
                )
                continue
            if fmt == "python" and strategy == "pandas" and not PANDAS_AVAILABLE:
                metrics[_daft_metric_key(fmt, strategy)] = unavailable_backend(
                    _daft_label(fmt, strategy), "pandas not installed"
                )
                continue
            if fmt == "python" and strategy == "arrow" and not PYARROW_AVAILABLE:
                metrics[_daft_metric_key(fmt, strategy)] = unavailable_backend(
                    _daft_label(fmt, strategy), "pyarrow not installed"
                )
                continue

            pipeline_daft = Pipeline(
                nodes=[normalize], name="normalize_numeric"
            ).with_engine(DaftEngine(python_return_strategy=strategy or "auto"))
            daft_durations = benchmark_callable(
                lambda pipeline=pipeline_daft, fmt=fmt: pipeline.map(
                    inputs={"value": values, "mean": mean_val, "std": std_val},
                    map_over="value",
                    return_format=fmt,
                ),
                ctx,
            )
            metrics[_daft_metric_key(fmt, strategy)] = BackendStats(
                label=_daft_label(fmt, strategy),
                durations=daft_durations,
                note=_daft_note("automatic Daft translation", fmt, strategy),
            )

    if ctx.include_native and daft is not None and Series is not None and DataType is not None:
        @daft.func.batch(return_dtype=DataType.float64())  # type: ignore[misc]
        def normalize_batch(value: Series, mean: float, std: float) -> Series:
            arr = value.to_arrow().to_numpy()
            normalized = (arr - mean) / std
            return Series.from_numpy(normalized)

        def run_native() -> None:
            df = daft.from_pydict({"value": values})
            df = df.with_column(
                "normalized", normalize_batch(df["value"], mean_val, std_val)
            )
            df.collect()

        native_durations = benchmark_callable(run_native, ctx)
        metrics["native_daft"] = BackendStats(
            label="Daft native (@daft.func.batch)",
            durations=native_durations,
            note="vectorized batch UDF",
        )

    return ScenarioResult(
        name="vectorized_numeric",
        description="Normalize numeric column to highlight batch/vectorized paths",
        n_items=len(values),
        metrics=metrics,
        notes=[
            "Fastest baseline uses Daft's batch UDF to avoid Python per-row cost.",
        ],
    )


# ---------------------------------------------------------------------------
# Scenario 4: Nested pipeline using .as_node(map_over=...)
# ---------------------------------------------------------------------------
def scenario_nested_map(ctx: BenchmarkContext) -> ScenarioResult:
    texts = [f"Sentence number {i} with some tokens" for i in range(min(ctx.n_items, 5_000))]

    @node(output_name="cleaned")
    def clean(text: str) -> str:
        return text.strip().lower()

    @node(output_name="tokens")
    def tokenize(cleaned: str) -> List[str]:
        return cleaned.split()

    @node(output_name="length")
    def length(tokens: List[str]) -> int:
        return len(tokens)

    inner = Pipeline(nodes=[clean, tokenize, length], name="inner_text")
    mapped_node = inner.as_node(
        input_mapping={"texts": "text"},
        output_mapping={
            "cleaned": "all_cleaned",
            "tokens": "all_tokens",
            "length": "lengths",
        },
        map_over="texts",
        name="inner_text_map",
    )

    @node(output_name="avg_length")
    def avg_length(lengths: List[int]) -> float:
        return float(sum(lengths) / len(lengths)) if lengths else 0.0

    @node(output_name="vocabulary")
    def vocabulary(all_tokens: List[List[str]]) -> List[str]:
        vocab = sorted({token for tokens in all_tokens for token in tokens})
        return vocab

    pipeline = Pipeline(
        nodes=[mapped_node, avg_length, vocabulary],
        name="nested_text_pipeline",
    )

    metrics: Dict[str, BackendStats] = {}

    pipeline_hn = pipeline.with_engine(
        HypernodesEngine(node_executor="threaded", map_executor="threaded")
    )
    hn_durations = benchmark_callable(
        lambda: pipeline_hn.run(inputs={"texts": texts}),
        ctx,
    )
    metrics["hypernodes"] = BackendStats(
        label="HypernodesEngine(threaded)",
        durations=hn_durations,
        note="PipelineNode map handled in Python",
    )

    if DaftEngine is None:
        metrics["daft_engine"] = unavailable_backend(
            "DaftEngine", "Daft not installed"
        )
    else:
        pipeline_daft = Pipeline(
            nodes=[mapped_node, avg_length, vocabulary],
            name="nested_text_pipeline",
        ).with_engine(DaftEngine())
        daft_durations = benchmark_callable(
            lambda: pipeline_daft.run(inputs={"texts": texts}),
            ctx,
        )
        metrics["daft_engine"] = BackendStats(
            label="DaftEngine(auto)",
            durations=daft_durations,
            note="leverages .as_node(map_over=...) translation",
        )

    # No native baseline (Daft code would mirror engine translation closely)

    return ScenarioResult(
        name="nested_map_pipeline",
        description="Nested pipeline converted to PipelineNode with map_over",
        n_items=len(texts),
        metrics=metrics,
        notes=[
            "Validates explode/groupby/list_agg strategy matches notebook behavior."
        ],
    )


SCENARIOS: Dict[str, Callable[[BenchmarkContext], ScenarioResult]] = {
    "text": scenario_text_preprocessing,
    "stateful": scenario_stateful_encoder,
    "numeric": scenario_vectorized_numeric,
    "nested": scenario_nested_map,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark HypernodesEngine vs DaftEngine vs native Daft baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scale",
        choices=SCALE_TO_ITEMS.keys(),
        default="small",
        help="Preset number of items per scenario (overridden by --items).",
    )
    parser.add_argument(
        "--items",
        type=int,
        default=None,
        help="Override number of items per scenario.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["all", *SCENARIOS.keys()],
        default=["all"],
        help="Which scenarios to run.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed repetitions per backend.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs per backend before timing begins.",
    )
    parser.add_argument(
        "--daft-return-formats",
        nargs="+",
        choices=["python", "daft", "arrow"],
        default=["python"],
        help="Return formats to benchmark for DaftEngine map() scenarios.",
    )
    parser.add_argument(
        "--daft-python-strategies",
        nargs="+",
        choices=["auto", "pydict", "arrow", "pandas"],
        default=["auto"],
        help="Python conversion strategies to compare when return_format=python.",
    )
    parser.add_argument(
        "--skip-native",
        action="store_true",
        help="Skip the manual/native Daft baselines.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to dump JSON results.",
    )
    return parser.parse_args()


def print_summary(results: List[ScenarioResult]) -> None:
    backends = sorted(
        {backend for result in results for backend in result.metrics.keys()}
    )
    header = ["Scenario", "Items", *backends]
    col_widths = [24, 10, *[22 for _ in backends]]
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(" ".join(title.ljust(width) for title, width in zip(header, col_widths)))
    for result in results:
        row = [
            result.name.ljust(col_widths[0]),
            str(result.n_items).rjust(col_widths[1]),
        ]
        for backend, width in zip(backends, col_widths[2:]):
            stats = result.metrics.get(backend)
            if stats is None or not stats.success:
                cell = "--"
                if stats and stats.note:
                    cell = stats.note
                row.append(cell.ljust(width))
                continue
            cell = f"{format_seconds(stats.mean)} (best {format_seconds(stats.best)})"
            row.append(cell.ljust(width))
        print(" ".join(row))

    print("\nDETAILS")
    print("-" * 80)
    for result in results:
        print(f"\n[{result.name}] {result.description} ({result.n_items} items)")
        for backend_name, stats in result.metrics.items():
            if not stats.success:
                print(f"  - {backend_name}: {stats.note}")
                continue
            print(
                f"  - {backend_name}: mean={format_seconds(stats.mean)}, "
                f"best={format_seconds(stats.best)}, stdev={format_seconds(stats.stdev)}"
            )
            if stats.note:
                print(f"      note: {stats.note}")
        for note in result.notes:
            print(f"    * {note}")


def main() -> None:
    args = parse_args()
    n_items = args.items or SCALE_TO_ITEMS[args.scale]
    selected = args.scenarios
    if "all" in selected:
        scenario_keys = list(SCENARIOS.keys())
    else:
        scenario_keys = selected

    ctx = BenchmarkContext(
        n_items=n_items,
        repeats=args.repeats,
        warmup=args.warmup,
        include_native=not args.skip_native,
        daft_return_formats=args.daft_return_formats,
        daft_python_strategies=args.daft_python_strategies,
    )

    results: List[ScenarioResult] = []
    for key in scenario_keys:
        scenario_fn = SCENARIOS[key]
        print(f"\nRunning scenario '{key}' ({ctx.n_items} base items)...")
        try:
            result = scenario_fn(ctx)
            results.append(result)
        except Exception as exc:  # pragma: no cover - surfaced to CLI
            print(f"  ! Scenario '{key}' failed: {exc}")
            raise

    print_summary(results)

    if args.json:
        payload = {
            "scale": args.scale,
            "items": n_items,
            "results": [result.to_dict() for result in results],
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":  # pragma: no cover
    main()
