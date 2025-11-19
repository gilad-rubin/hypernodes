"""Benchmark tests comparing Sequential vs DAFT execution engines.

Covers various scenarios:
- Simple execution (baseline)
- Map operations (parallelization)
- Nested pipelines
- Nested pipelines with map
- I/O-heavy workloads
- CPU-heavy workloads

Run with: pytest tests/test_benchmarks_engines.py -v -s
"""

import time
from dataclasses import dataclass
from typing import Dict, List

import pytest

from hypernodes import Pipeline, node
from hypernodes.pipeline_node import PipelineNode
from hypernodes.sequential_engine import SeqEngine

# Try to import DaftEngine
try:
    from hypernodes.integrations.daft.engine import DaftEngine

    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    test_name: str
    engine_name: str
    duration: float
    speedup: float = 1.0


class BenchmarkSuite:
    """Benchmark suite for comparing engines."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.sequential_times: Dict[str, float] = {}

    def run_benchmark(
        self,
        test_name: str,
        engine,
        pipeline: Pipeline,
        inputs: Dict,
        map_over: str = None,
        map_mode: str = "zip",
    ) -> float:
        """Run a single benchmark and return duration."""
        start = time.time()
        try:
            if map_over:
                pipeline.map(inputs=inputs, map_over=map_over, map_mode=map_mode)
            else:
                pipeline.run(inputs=inputs)
        except Exception as e:
            pytest.fail(f"Benchmark failed: {type(e).__name__}: {str(e)[:80]}")
            return 0.0
        duration = time.time() - start
        return duration

    def create_result(
        self, test_name: str, engine_name: str, duration: float
    ) -> BenchmarkResult:
        """Create a benchmark result with speedup calculation."""
        sequential_time = self.sequential_times.get(test_name, duration)
        speedup = sequential_time / duration if duration > 0 else 1.0
        return BenchmarkResult(test_name, engine_name, duration, speedup)

    def print_table(self) -> None:
        """Print results in a formatted table."""
        print("\n" + "=" * 100)
        print("BENCHMARK RESULTS")
        print("=" * 100)

        # Group by test
        test_groups: Dict[str, List[BenchmarkResult]] = {}
        for result in self.results:
            if result.test_name not in test_groups:
                test_groups[result.test_name] = []
            test_groups[result.test_name].append(result)

        # Print headers
        print(f"{'Test Name':<40} {'Engine':<20} {'Duration (s)':<15} {'Speedup':<10}")
        print("-" * 100)

        # Print results grouped by test
        for test_name in sorted(test_groups.keys()):
            results = test_groups[test_name]
            for i, result in enumerate(results):
                test_display = test_name if i == 0 else ""
                speedup_str = (
                    f"{result.speedup:.2f}x" if result.speedup > 1 else "1.00x"
                )
                print(
                    f"{test_display:<40} {result.engine_name:<20} "
                    f"{result.duration:<15.4f} {speedup_str:<10}"
                )
            print("-" * 100)

        print("=" * 100)

    # ===== Test Cases =====

    def test_simple_execution(self) -> None:
        """Test 1: Simple sequential execution (baseline)."""
        test_name = "Simple Execution"

        @node(output_name="add_result")
        def add_one(x: int) -> int:
            return x + 1

        @node(output_name="result")
        def multiply_by_two(add_result: int) -> int:
            return add_result * 2

        # Sequential engine
        seq_engine = SeqEngine()
        seq_pipeline = Pipeline(nodes=[add_one, multiply_by_two], engine=seq_engine)
        seq_time = self.run_benchmark(test_name, seq_engine, seq_pipeline, {"x": 5})
        self.sequential_times[test_name] = seq_time
        self.results.append(self.create_result(test_name, "SeqEngine", seq_time))
        assert seq_time >= 0

        # Daft engine
        if DAFT_AVAILABLE:
            daft_engine = DaftEngine()
            daft_pipeline = Pipeline(
                nodes=[add_one, multiply_by_two], engine=daft_engine
            )
            daft_time = self.run_benchmark(
                test_name, daft_engine, daft_pipeline, {"x": 5}
            )
            self.results.append(self.create_result(test_name, "DaftEngine", daft_time))
            assert daft_time >= 0

    def test_map_basic(self) -> None:
        """Test 2: Basic map operation (shows parallelization)."""
        test_name = "Map - Basic (10 items)"

        @node(output_name="add_result")
        def add_one(x: int) -> int:
            return x + 1

        @node(output_name="result")
        def multiply_by_two(add_result: int) -> int:
            return add_result * 2

        inputs = {"x": list(range(10))}

        # Sequential engine
        seq_engine = SeqEngine()
        seq_pipeline = Pipeline(nodes=[add_one, multiply_by_two], engine=seq_engine)
        seq_time = self.run_benchmark(
            test_name, seq_engine, seq_pipeline, inputs, map_over="x"
        )
        self.sequential_times[test_name] = seq_time
        self.results.append(self.create_result(test_name, "SeqEngine", seq_time))
        assert seq_time >= 0

        # Daft engine
        if DAFT_AVAILABLE:
            daft_engine = DaftEngine()
            daft_pipeline = Pipeline(
                nodes=[add_one, multiply_by_two], engine=daft_engine
            )
            daft_time = self.run_benchmark(
                test_name, daft_engine, daft_pipeline, inputs, map_over="x"
            )
            self.results.append(self.create_result(test_name, "DaftEngine", daft_time))
            assert daft_time >= 0

    def test_map_io_heavy(self) -> None:
        """Test 3: Map with I/O-heavy operations (sleep)."""
        test_name = "Map - I/O Heavy (0.1s each)"

        @node(output_name="result")
        def io_operation(x: int) -> int:
            """Simulate I/O operation with sleep."""
            time.sleep(0.05)  # Reduced from 0.1 for faster testing
            return x * 2

        inputs = {"x": list(range(3))}  # Reduced from 5

        # Sequential engine
        seq_engine = SeqEngine()
        seq_pipeline = Pipeline(nodes=[io_operation], engine=seq_engine)
        seq_time = self.run_benchmark(
            test_name, seq_engine, seq_pipeline, inputs, map_over="x"
        )
        self.sequential_times[test_name] = seq_time
        self.results.append(self.create_result(test_name, "SeqEngine", seq_time))
        assert seq_time >= 0

        # Daft engine
        if DAFT_AVAILABLE:
            daft_engine = DaftEngine()
            daft_pipeline = Pipeline(nodes=[io_operation], engine=daft_engine)
            daft_time = self.run_benchmark(
                test_name, daft_engine, daft_pipeline, inputs, map_over="x"
            )
            self.results.append(self.create_result(test_name, "DaftEngine", daft_time))
            assert daft_time >= 0

    def test_map_cpu_heavy(self) -> None:
        """Test 4: Map with CPU-heavy operations."""
        test_name = "Map - CPU Heavy"

        @node(output_name="result")
        def cpu_operation(x: int) -> int:
            """Simulate CPU operation."""
            result = x
            for _ in range(500000):  # Reduced from 1000000
                result = (result * 7 + 11) % 1000000
            return result

        inputs = {"x": list(range(3))}  # Reduced from 5

        # Sequential engine
        seq_engine = SeqEngine()
        seq_pipeline = Pipeline(nodes=[cpu_operation], engine=seq_engine)
        seq_time = self.run_benchmark(
            test_name, seq_engine, seq_pipeline, inputs, map_over="x"
        )
        self.sequential_times[test_name] = seq_time
        self.results.append(self.create_result(test_name, "SeqEngine", seq_time))
        assert seq_time >= 0

        # Daft engine
        if DAFT_AVAILABLE:
            daft_engine = DaftEngine()
            daft_pipeline = Pipeline(nodes=[cpu_operation], engine=daft_engine)
            daft_time = self.run_benchmark(
                test_name, daft_engine, daft_pipeline, inputs, map_over="x"
            )
            self.results.append(self.create_result(test_name, "DaftEngine", daft_time))
            assert daft_time >= 0

    def test_nested_pipeline(self) -> None:
        """Test 5: Nested pipelines."""
        test_name = "Nested Pipeline"

        @node(output_name="inner_result")
        def inner_add(x: int) -> int:
            return x + 10

        @node(output_name="inner_mult")
        def inner_multiply(inner_result: int) -> int:
            return inner_result * 2

        inner_pipeline = Pipeline(nodes=[inner_add, inner_multiply])

        @node(output_name="outer_result")
        def outer_transform(inner_mult: int) -> int:
            return inner_mult * 3

        # Sequential engine
        pipeline_node_seq = PipelineNode(pipeline=inner_pipeline)
        seq_engine = SeqEngine()
        outer_pipeline_seq = Pipeline(
            nodes=[pipeline_node_seq, outer_transform], engine=seq_engine
        )
        seq_time = self.run_benchmark(
            test_name, seq_engine, outer_pipeline_seq, {"x": 5}
        )
        self.sequential_times[test_name] = seq_time
        self.results.append(self.create_result(test_name, "SeqEngine", seq_time))
        assert seq_time >= 0

        # Daft engine
        if DAFT_AVAILABLE:
            pipeline_node_daft = PipelineNode(pipeline=inner_pipeline)
            daft_engine = DaftEngine()
            outer_pipeline_daft = Pipeline(
                nodes=[pipeline_node_daft, outer_transform], engine=daft_engine
            )
            daft_time = self.run_benchmark(
                test_name, daft_engine, outer_pipeline_daft, {"x": 5}
            )
            self.results.append(self.create_result(test_name, "DaftEngine", daft_time))
            assert daft_time >= 0

    def test_nested_with_map(self) -> None:
        """Test 6: Nested pipeline with map."""
        test_name = "Nested Pipeline + Map"

        @node(output_name="inner_result")
        def inner_add(x: int) -> int:
            return x + 10

        @node(output_name="inner_mult")
        def inner_multiply(inner_result: int) -> int:
            return inner_result * 2

        inner_pipeline = Pipeline(nodes=[inner_add, inner_multiply])

        @node(output_name="outer_result")
        def outer_transform(inner_mult: int) -> int:
            return inner_mult * 3

        inputs = {"x": list(range(5))}

        # Sequential engine
        pipeline_node_seq = PipelineNode(pipeline=inner_pipeline)
        seq_engine = SeqEngine()
        outer_pipeline_seq = Pipeline(
            nodes=[pipeline_node_seq, outer_transform], engine=seq_engine
        )
        seq_time = self.run_benchmark(
            test_name, seq_engine, outer_pipeline_seq, inputs, map_over="x"
        )
        self.sequential_times[test_name] = seq_time
        self.results.append(self.create_result(test_name, "SeqEngine", seq_time))
        assert seq_time >= 0

        # Daft engine
        if DAFT_AVAILABLE:
            pipeline_node_daft = PipelineNode(pipeline=inner_pipeline)
            daft_engine = DaftEngine()
            outer_pipeline_daft = Pipeline(
                nodes=[pipeline_node_daft, outer_transform], engine=daft_engine
            )
            daft_time = self.run_benchmark(
                test_name, daft_engine, outer_pipeline_daft, inputs, map_over="x"
            )
            self.results.append(self.create_result(test_name, "DaftEngine", daft_time))
            assert daft_time >= 0

    def test_complex_graph(self) -> None:
        """Test 7: Complex DAG with multiple branches."""
        test_name = "Complex DAG"

        @node(output_name="a")
        def compute_a(x: int) -> int:
            return x + 1

        @node(output_name="b")
        def compute_b(x: int) -> int:
            return x * 2

        @node(output_name="c")
        def compute_c(a: int, b: int) -> int:
            return a + b

        @node(output_name="result")
        def compute_result(a: int, b: int, c: int) -> int:
            return a + b + c

        # Sequential engine
        seq_engine = SeqEngine()
        seq_pipeline = Pipeline(
            nodes=[compute_a, compute_b, compute_c, compute_result], engine=seq_engine
        )
        seq_time = self.run_benchmark(test_name, seq_engine, seq_pipeline, {"x": 5})
        self.sequential_times[test_name] = seq_time
        self.results.append(self.create_result(test_name, "SeqEngine", seq_time))
        assert seq_time >= 0

        # Daft engine
        if DAFT_AVAILABLE:
            daft_engine = DaftEngine()
            daft_pipeline = Pipeline(
                nodes=[compute_a, compute_b, compute_c, compute_result],
                engine=daft_engine,
            )
            daft_time = self.run_benchmark(
                test_name, daft_engine, daft_pipeline, {"x": 5}
            )
            self.results.append(self.create_result(test_name, "DaftEngine", daft_time))
            assert daft_time >= 0

    def test_map_multiple_params(self) -> None:
        """Test 8: Map over multiple parameters (zip mode)."""
        test_name = "Map - Multiple Params"

        @node(output_name="sum_result")
        def add_two(x: int, y: int) -> int:
            return x + y

        @node(output_name="result")
        def multiply(sum_result: int) -> int:
            return sum_result * 2

        inputs = {"x": list(range(10)), "y": list(range(10, 20))}

        # Sequential engine
        seq_engine = SeqEngine()
        seq_pipeline = Pipeline(nodes=[add_two, multiply], engine=seq_engine)
        seq_time = self.run_benchmark(
            test_name,
            seq_engine,
            seq_pipeline,
            inputs,
            map_over=["x", "y"],
            map_mode="zip",
        )
        self.sequential_times[test_name] = seq_time
        self.results.append(self.create_result(test_name, "SeqEngine", seq_time))
        assert seq_time >= 0

        # Daft engine
        if DAFT_AVAILABLE:
            daft_engine = DaftEngine()
            daft_pipeline = Pipeline(nodes=[add_two, multiply], engine=daft_engine)
            daft_time = self.run_benchmark(
                test_name,
                daft_engine,
                daft_pipeline,
                inputs,
                map_over=["x", "y"],
                map_mode="zip",
            )
            self.results.append(self.create_result(test_name, "DaftEngine", daft_time))
            assert daft_time >= 0


# ===== Pytest Fixtures and Tests =====


@pytest.fixture(scope="session")
def benchmark_suite():
    """Create and run benchmark suite once per session."""
    suite = BenchmarkSuite()
    return suite


def test_benchmarks_simple_execution(benchmark_suite):
    """Run simple execution benchmark."""
    benchmark_suite.test_simple_execution()


def test_benchmarks_map_basic(benchmark_suite):
    """Run basic map benchmark."""
    benchmark_suite.test_map_basic()


def test_benchmarks_map_io_heavy(benchmark_suite):
    """Run I/O-heavy map benchmark."""
    benchmark_suite.test_map_io_heavy()


def test_benchmarks_map_cpu_heavy(benchmark_suite):
    """Run CPU-heavy map benchmark."""
    benchmark_suite.test_map_cpu_heavy()


def test_benchmarks_nested_pipeline(benchmark_suite):
    """Run nested pipeline benchmark."""
    benchmark_suite.test_nested_pipeline()


def test_benchmarks_nested_with_map(benchmark_suite):
    """Run nested pipeline with map benchmark."""
    benchmark_suite.test_nested_with_map()


def test_benchmarks_complex_graph(benchmark_suite):
    """Run complex DAG benchmark."""
    benchmark_suite.test_complex_graph()


def test_benchmarks_map_multiple_params(benchmark_suite):
    """Run map with multiple parameters benchmark."""
    benchmark_suite.test_map_multiple_params()


def test_benchmarks_results_summary(benchmark_suite):
    """Print final benchmark results table."""
    benchmark_suite.print_table()
    print(f"\nTotal benchmarks: {len(benchmark_suite.results)}")
