"""Tests for Daft optimization features (stateful params, batch_size, etc.)."""

import pytest
import time

try:
    import daft
    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False

from hypernodes import Pipeline, node
from hypernodes.integrations.daft import DaftEngine


pytestmark = pytest.mark.skipif(not DAFT_AVAILABLE, reason="Daft not installed")


class ExpensiveResource:
    """Simulate an expensive resource for testing."""
    def __init__(self, init_time: float = 0.05):
        self.init_time = init_time
        self.init_count = 0
        time.sleep(init_time)
        self.init_count += 1
    
    def process(self, text: str) -> str:
        return text.strip().lower()


class StatefulResource:
    """Resource with stateful hint."""
    __daft_stateful__ = True
    
    def __init__(self):
        self.counter = 0
    
    def __call__(self, text: str) -> str:
        self.counter += 1
        return text.lower()


def test_stateful_param_explicit_hint():
    """Test stateful parameters with explicit hint."""
    resource = ExpensiveResource(init_time=0.01)
    
    @node(output_name="result", stateful_params=["resource"])
    def process(text: str, resource: ExpensiveResource) -> str:
        return resource.process(text)
    
    pipeline = Pipeline(
        nodes=[process],
        engine=DaftEngine(use_batch_udf=True)
    )
    
    texts = [f"  TEXT {i}  " for i in range(100)]
    result = pipeline.map(
        inputs={"text": texts, "resource": resource},
        map_over="text"
    )
    
    # Verify results
    assert len(result) == 100
    assert all(r["result"] == f"text {i}" for i, r in enumerate(result))


def test_stateful_param_auto_detection():
    """Test auto-detection of stateful parameters via __daft_stateful__."""
    resource = StatefulResource()
    
    @node(output_name="result")  # No explicit hint!
    def process(text: str, resource: StatefulResource) -> str:
        return resource(text)
    
    pipeline = Pipeline(
        nodes=[process],
        engine=DaftEngine(use_batch_udf=True)
    )
    
    texts = [f"TEXT{i}" for i in range(100)]
    result = pipeline.map(
        inputs={"text": texts, "resource": resource},
        map_over="text"
    )
    
    # Verify results
    assert len(result) == 100
    assert all(r["result"] == f"text{i}" for i, r in enumerate(result))


def test_batch_size_configuration():
    """Test that batch_size configuration is applied."""
    @node(
        output_name="result",
        daft_config={"batch_size": 512}
    )
    def process(value: int) -> int:
        return value * 2
    
    pipeline = Pipeline(
        nodes=[process],
        engine=DaftEngine(use_batch_udf=True)
    )
    
    values = list(range(1000))
    result = pipeline.map(
        inputs={"value": values},
        map_over="value"
    )
    
    # Verify results
    assert len(result) == 1000
    assert all(r["result"] == i * 2 for i, r in enumerate(result))


def test_max_concurrency_configuration():
    """Test that max_concurrency configuration is applied."""
    @node(
        output_name="result",
        daft_config={"max_concurrency": 2}
    )
    def process(text: str) -> str:
        return text.upper()
    
    pipeline = Pipeline(
        nodes=[process],
        engine=DaftEngine(use_batch_udf=True)
    )
    
    texts = [f"text{i}" for i in range(100)]
    result = pipeline.map(
        inputs={"text": texts},
        map_over="text"
    )
    
    # Verify results
    assert len(result) == 100
    assert all(r["result"] == f"TEXT{i}" for i, r in enumerate(result))


def test_use_process_configuration():
    """Test that use_process configuration is applied."""
    @node(
        output_name="word_count",
        daft_config={"use_process": True}
    )
    def count_words(text: str) -> int:
        return len(text.split())
    
    pipeline = Pipeline(
        nodes=[count_words],
        engine=DaftEngine(use_batch_udf=True)
    )
    
    texts = [f"word " * i for i in range(1, 11)]
    result = pipeline.map(
        inputs={"text": texts},
        map_over="text"
    )
    
    # Verify results
    assert len(result) == 10
    assert all(r["word_count"] == i for i, r in enumerate(result, 1))


def test_combined_configuration():
    """Test combining multiple optimization parameters."""
    resource = StatefulResource()
    
    @node(
        output_name="result",
        stateful_params=["resource"],
        daft_config={
            "batch_size": 256,
            "max_concurrency": 2,
            "use_process": False  # Keep False for testing simplicity
        }
    )
    def process(text: str, resource: StatefulResource) -> str:
        return resource(text)
    
    pipeline = Pipeline(
        nodes=[process],
        engine=DaftEngine(use_batch_udf=True)
    )
    
    texts = [f"TEXT{i}" for i in range(500)]
    result = pipeline.map(
        inputs={"text": texts, "resource": resource},
        map_over="text"
    )
    
    # Verify results
    assert len(result) == 500
    assert all(r["result"] == f"text{i}" for i, r in enumerate(result))


def test_engine_level_default_config():
    """Test engine-level default configuration."""
    @node(output_name="result")
    def process(value: int) -> int:
        return value * 3
    
    pipeline = Pipeline(
        nodes=[process],
        engine=DaftEngine(
            use_batch_udf=True,
            default_daft_config={"batch_size": 1024}
        )
    )
    
    values = list(range(100))
    result = pipeline.map(
        inputs={"value": values},
        map_over="value"
    )
    
    # Verify results
    assert len(result) == 100
    assert all(r["result"] == i * 3 for i, r in enumerate(result))


def test_node_config_overrides_engine_config():
    """Test that node-level config overrides engine-level config."""
    @node(
        output_name="result",
        daft_config={"batch_size": 128}  # Override engine default
    )
    def process(value: int) -> int:
        return value * 4
    
    pipeline = Pipeline(
        nodes=[process],
        engine=DaftEngine(
            use_batch_udf=True,
            default_daft_config={"batch_size": 2048}  # This gets overridden
        )
    )
    
    values = list(range(100))
    result = pipeline.map(
        inputs={"value": values},
        map_over="value"
    )
    
    # Verify results
    assert len(result) == 100
    assert all(r["result"] == i * 4 for i, r in enumerate(result))


def test_stateful_performance_benefit():
    """Test that stateful parameters provide performance benefit."""
    # Non-stateful version (resource created in function)
    @node(output_name="result")
    def process_nonstateful(text: str) -> str:
        resource = ExpensiveResource(init_time=0.005)  # Small delay per call
        return resource.process(text)
    
    pipeline_nonstateful = Pipeline(
        nodes=[process_nonstateful],
        engine=DaftEngine(use_batch_udf=True)
    )
    
    texts = [f"TEXT{i}" for i in range(100)]
    
    start = time.time()
    result1 = pipeline_nonstateful.map(
        inputs={"text": texts},
        map_over="text"
    )
    time_nonstateful = time.time() - start
    
    # Stateful version (resource created once)
    resource = ExpensiveResource(init_time=0.1)  # One-time init
    
    @node(output_name="result", stateful_params=["resource"])
    def process_stateful(text: str, resource: ExpensiveResource) -> str:
        return resource.process(text)
    
    pipeline_stateful = Pipeline(
        nodes=[process_stateful],
        engine=DaftEngine(use_batch_udf=True)
    )
    
    start = time.time()
    result2 = pipeline_stateful.map(
        inputs={"text": texts, "resource": resource},
        map_over="text"
    )
    time_stateful = time.time() - start
    
    # Verify results are the same
    assert result1 == result2
    
    # Stateful should be significantly faster
    # (This is a loose check to avoid flakiness)
    print(f"\nNon-stateful: {time_nonstateful:.4f}s")
    print(f"Stateful: {time_stateful:.4f}s")
    print(f"Speedup: {time_nonstateful / time_stateful:.2f}x")
    
    # We expect at least 2x speedup with these parameters
    assert time_stateful < time_nonstateful / 2, \
        f"Expected stateful to be >2x faster, got {time_nonstateful / time_stateful:.2f}x"


def test_multiple_stateful_params():
    """Test multiple stateful parameters in one node."""
    resource1 = StatefulResource()
    resource2 = StatefulResource()
    
    @node(
        output_name="result",
        stateful_params=["r1", "r2"]
    )
    def process(text: str, r1: StatefulResource, r2: StatefulResource) -> str:
        return r1(r2(text))
    
    pipeline = Pipeline(
        nodes=[process],
        engine=DaftEngine(use_batch_udf=True)
    )
    
    texts = [f"TEXT{i}" for i in range(50)]
    result = pipeline.map(
        inputs={"text": texts, "r1": resource1, "r2": resource2},
        map_over="text"
    )
    
    # Verify results
    assert len(result) == 50
    assert all(r["result"] == f"text{i}" for i, r in enumerate(result))


def test_stateful_with_multiple_nodes():
    """Test stateful parameters across multiple nodes in pipeline."""
    resource = StatefulResource()
    
    @node(output_name="step1")
    def first(text: str) -> str:
        return text.upper()
    
    @node(output_name="result", stateful_params=["resource"])
    def second(step1: str, resource: StatefulResource) -> str:
        return resource(step1)
    
    pipeline = Pipeline(
        nodes=[first, second],
        engine=DaftEngine(use_batch_udf=True)
    )
    
    texts = [f"text{i}" for i in range(50)]
    result = pipeline.map(
        inputs={"text": texts, "resource": resource},
        map_over="text"
    )
    
    # Verify results (upper then lower)
    assert len(result) == 50
    assert all(r["result"] == f"text{i}" for i, r in enumerate(result))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

