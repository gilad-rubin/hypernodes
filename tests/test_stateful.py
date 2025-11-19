"""Tests for @stateful decorator and stateful object handling."""

import pytest

from hypernodes import DiskCache, Pipeline, node, stateful
from hypernodes.engines import SeqEngine


class TestStatefulDecorator:
    """Test the @stateful decorator."""

    def test_decorator_creates_lazy_wrapper(self):
        """Test that @stateful creates a lazy initialization wrapper."""

        @stateful
        class TestClass:
            def __init__(self, value: int):
                self.value = value

        # Check the wrapper class has the marker
        assert hasattr(TestClass, "__hypernode_stateful__")
        assert TestClass.__hypernode_stateful__ is True

        # Create an instance - this should NOT call __init__ yet
        instance = TestClass(value=42)
        assert hasattr(instance, "_instance")
        assert instance._instance is None  # Not initialized yet!

        # Accessing attribute should trigger lazy init
        assert instance.value == 42
        assert instance._instance is not None  # Now initialized

    def test_decorator_preserves_functionality(self):
        """Test that @stateful preserves class functionality with lazy init."""

        init_count = {"count": 0}

        @stateful
        class Counter:
            def __init__(self, start: int = 0):
                init_count["count"] += 1
                self.count = start

            def increment(self):
                self.count += 1
                return self.count

        # Creating instance doesn't call __init__ (lazy)
        counter = Counter(10)
        assert init_count["count"] == 0

        # Accessing attribute triggers init
        assert counter.count == 10
        assert init_count["count"] == 1

        # Methods work normally
        assert counter.increment() == 11
        assert counter.count == 11
        assert init_count["count"] == 1  # Still only initialized once


class TestSeqEngineStateful:
    """Test SeqEngine stateful object caching."""

    def test_stateful_object_reused_in_map(self):
        """Test that stateful objects are reused across map items."""

        # Track how many times __init__ is called
        init_count = {"count": 0}

        @stateful
        class ExpensiveResource:
            def __init__(self, multiplier: int):
                init_count["count"] += 1
                self.multiplier = multiplier

            def process(self, x: int) -> int:
                return x * self.multiplier

        @node(output_name="result")
        def process(x: int, resource: ExpensiveResource) -> int:
            return resource.process(x)

        # Create pipeline with stateful resource
        # With lazy init, __init__ is NOT called yet
        resource = ExpensiveResource(multiplier=10)
        assert init_count["count"] == 0  # Not initialized yet

        pipeline = Pipeline(nodes=[process], engine=SeqEngine())

        # Run map over 5 items
        results = pipeline.map(
            inputs={"x": [1, 2, 3, 4, 5], "resource": resource}, map_over="x"
        )

        # Resource should be initialized only once (on first use)
        assert init_count["count"] == 1

        # Results should be correct
        assert results == [
            {"result": 10},
            {"result": 20},
            {"result": 30},
            {"result": 40},
            {"result": 50},
        ]

    def test_non_stateful_object_not_cached(self):
        """Test that non-stateful objects follow normal behavior."""

        # Regular class without @stateful
        class RegularResource:
            def __init__(self, value: int):
                self.value = value

        @node(output_name="result")
        def process(x: int, resource: RegularResource) -> int:
            return x + resource.value

        resource = RegularResource(value=100)
        pipeline = Pipeline(nodes=[process], engine=SeqEngine())

        results = pipeline.map(
            inputs={"x": [1, 2, 3], "resource": resource}, map_over="x"
        )

        # Should still work (resource passed through normally)
        assert results == [{"result": 101}, {"result": 102}, {"result": 103}]


class TestCachingWithStateful:
    """Test caching behavior with stateful objects."""

    def test_cache_with_cache_key(self, tmp_path):
        """Test that stateful objects with __cache_key__() cache correctly."""

        @stateful
        class Model:
            def __init__(self, model_path: str):
                self.model_path = model_path
                self.load_count = 0  # Track loads

            def __cache_key__(self) -> str:
                return f"Model(path={self.model_path})"

            def predict(self, x: int) -> int:
                return x * 2

        call_count = {"count": 0}

        @node(output_name="prediction")
        def predict(x: int, model: Model) -> int:
            call_count["count"] += 1
            return model.predict(x)

        model = Model(model_path="test.pkl")
        cache = DiskCache(path=str(tmp_path / ".cache"))
        pipeline = Pipeline(nodes=[predict], engine=SeqEngine(cache=cache))

        # First run - should execute
        result1 = pipeline.run(inputs={"x": 5, "model": model})
        assert result1 == {"prediction": 10}
        assert call_count["count"] == 1

        # Second run with same inputs - should hit cache
        result2 = pipeline.run(inputs={"x": 5, "model": model})
        assert result2 == {"prediction": 10}
        assert call_count["count"] == 1  # No additional execution

    def test_cache_without_custom_cache_key(self, tmp_path):
        """Test that stateful objects work with default cache key."""

        @stateful
        class ModelWithoutKey:
            def __init__(self, model_path: str):
                self.model_path = model_path

        @node(output_name="result")
        def process(x: int, model: ModelWithoutKey) -> int:
            return x * 2

        model = ModelWithoutKey(model_path="test.pkl")
        cache = DiskCache(path=str(tmp_path / ".cache"))
        pipeline = Pipeline(nodes=[process], engine=SeqEngine(cache=cache))

        # Should work with default cache key (from init args)
        result1 = pipeline.run(inputs={"x": 5, "model": model})
        assert result1 == {"result": 10}

        # Different init args should not hit cache
        model2 = ModelWithoutKey(model_path="different.pkl")
        call_count = {"count": 0}

        @node(output_name="result2")
        def process2(x: int, model: ModelWithoutKey) -> int:
            call_count["count"] += 1
            return x * 2

        pipeline2 = Pipeline(nodes=[process2], engine=SeqEngine(cache=cache))
        result2 = pipeline2.run(inputs={"x": 5, "model": model2})
        assert result2 == {"result2": 10}
        assert call_count["count"] == 1  # Executed (different init args)


class TestDaftEngineStateful:
    """Test DaftEngine stateful object handling.

    StatefulWrapper objects are already lazy and serializable, so DaftEngine
    just passes them through. Daft serializes them with cloudpickle, and they
    lazy-init on workers on first access.
    """

    def test_daft_engine_with_stateful_simple(self):
        """Test that DaftEngine works with stateful objects (lazy init)."""
        try:
            from hypernodes.engines import DaftEngine
        except ImportError:
            pytest.skip("DaftEngine not available (daft not installed)")

        init_count = {"count": 0}

        @stateful
        class Config:
            def __init__(self, multiplier: int):
                init_count["count"] += 1
                self.multiplier = multiplier

        @node(output_name="result")
        def process(x: int, config: Config) -> int:
            return x * config.multiplier

        config = Config(multiplier=10)
        assert init_count["count"] == 0  # Lazy init - not called yet

        pipeline = Pipeline(nodes=[process], engine=DaftEngine())

        result = pipeline.run(inputs={"x": 5, "config": config})
        assert result == {"result": 50}
        # Init should be called during execution
        assert init_count["count"] >= 1

    def test_daft_engine_map_with_stateful_simple(self):
        """Test DaftEngine map operation with stateful objects (lazy init)."""
        try:
            from hypernodes.engines import DaftEngine
        except ImportError:
            pytest.skip("DaftEngine not available (daft not installed)")

        init_count = {"count": 0}

        @stateful
        class Offset:
            def __init__(self, value: int):
                init_count["count"] += 1
                self.value = value

        @node(output_name="result")
        def add_offset(x: int, offset: Offset) -> int:
            return x + offset.value

        offset = Offset(value=100)
        assert init_count["count"] == 0  # Lazy init

        pipeline = Pipeline(nodes=[add_offset], engine=DaftEngine())

        results = pipeline.map(inputs={"x": [1, 2, 3], "offset": offset}, map_over="x")

        assert results == [{"result": 101}, {"result": 102}, {"result": 103}]
        # Init should be called during execution
        assert init_count["count"] >= 1


class TestStatefulWithDualNode:
    """Test stateful objects work with DualNode."""

    def test_dual_node_with_stateful_param(self):
        """Test that DualNode works with stateful parameters."""
        from hypernodes import DualNode

        @stateful
        class BatchProcessor:
            def __init__(self, multiplier: int):
                self.multiplier = multiplier

            def process_one(self, x: int) -> int:
                return x * self.multiplier

            def process_batch(self, xs: list[int]) -> list[int]:
                return [x * self.multiplier for x in xs]

        # Create DualNode with methods from stateful instance
        processor = BatchProcessor(multiplier=5)
        dual = DualNode(
            output_name="result",
            singular=processor.process_one,
            batch=processor.process_batch,
        )

        # Use in pipeline
        pipeline = Pipeline(nodes=[dual], engine=SeqEngine())

        # Test single execution (uses singular)
        result = pipeline.run(inputs={"x": 10})
        assert result == {"result": 50}

        # Test map execution (Sequential uses singular for each item)
        results = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
        assert results == [{"result": 5}, {"result": 10}, {"result": 15}]


# Module-level class for serialization test (can't pickle local classes)
@stateful
class SerializableResource:
    def __init__(self, value: int):
        self.value = value

    def __cache_key__(self) -> str:
        return f"SerializableResource({self.value})"


class TestAsyncSupport:
    """Test async function support with stateful objects."""

    def test_async_node_sequential_engine(self):
        """Test that SeqEngine auto-detects and runs async functions."""
        import asyncio

        @node(output_name="result")
        async def async_task(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * 2

        pipeline = Pipeline(nodes=[async_task], engine=SeqEngine())
        result = pipeline.run(inputs={"x": 5})
        assert result == {"result": 10}

    def test_async_with_stateful_sequential(self):
        """Test async function with stateful parameter in SeqEngine."""
        import asyncio

        @stateful
        class AsyncClient:
            def __init__(self, base_url: str):
                self.base_url = base_url

        @node(output_name="response")
        async def fetch(item_id: str, client: AsyncClient) -> str:
            await asyncio.sleep(0.001)
            return f"{client.base_url}/{item_id}"

        client = AsyncClient(base_url="http://api.com")
        pipeline = Pipeline(nodes=[fetch], engine=SeqEngine())
        result = pipeline.run(inputs={"item_id": "test", "client": client})
        assert result == {"response": "http://api.com/test"}

    def test_async_in_jupyter_context(self):
        """Test async works in nested event loop (Jupyter simulation)."""
        import asyncio

        @node(output_name="data")
        async def async_fetch(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * 10

        pipeline = Pipeline(nodes=[async_fetch], engine=SeqEngine())

        # Simulate Jupyter by running in async context
        async def run_in_loop():
            # This simulates being in Jupyter's running event loop
            return pipeline.run(inputs={"x": 5})

        result = asyncio.run(run_in_loop())
        assert result == {"data": 50}


class TestStatefulEdgeCases:
    """Test edge cases and error conditions."""

    def test_stateful_with_multiple_instances(self):
        """Test that different instances of same stateful class work correctly."""

        @stateful
        class Multiplier:
            def __init__(self, factor: int):
                self.factor = factor

            def __cache_key__(self) -> str:
                return f"Multiplier({self.factor})"

            def multiply(self, x: int) -> int:
                return x * self.factor

        @node(output_name="result")
        def process(x: int, mult: Multiplier) -> int:
            return mult.multiply(x)

        pipeline = Pipeline(nodes=[process], engine=SeqEngine())

        # Different multiplier instances should produce different results
        mult_2 = Multiplier(factor=2)
        mult_3 = Multiplier(factor=3)

        result_2 = pipeline.run(inputs={"x": 10, "mult": mult_2})
        result_3 = pipeline.run(inputs={"x": 10, "mult": mult_3})

        assert result_2 == {"result": 20}
        assert result_3 == {"result": 30}

    def test_stateful_object_serialization(self):
        """Test that stateful objects can be serialized with cloudpickle."""
        try:
            import cloudpickle
        except ImportError:
            pytest.skip("cloudpickle not installed")

        resource = SerializableResource(value=42)

        # Should be picklable with cloudpickle (used by Daft/Dask)
        pickled = cloudpickle.dumps(resource)
        restored = cloudpickle.loads(pickled)

        # Value accessible (triggers lazy init)
        assert restored.value == 42
        assert hasattr(restored, "__hypernode_stateful__") or hasattr(
            restored.__class__, "__hypernode_stateful__"
        )


class TestStatefulErrorCases:
    """Test error handling for stateful objects."""

    def test_daft_engine_rejects_stateful_only_nodes(self):
        """Test that DaftEngine raises error for nodes with only stateful params."""
        try:
            from hypernodes.integrations.daft import DaftEngine
        except ImportError:
            pytest.skip("DaftEngine not available")

        @stateful
        class Counter:
            def __init__(self, start: int = 0):
                self.count = start

            def increment(self) -> int:
                self.count += 1
                return self.count

        @node(output_name="count")
        def get_count(counter: Counter) -> int:
            """Node with ONLY stateful parameter (no dynamic inputs)."""
            return counter.increment()

        counter = Counter(start=10)
        pipeline = Pipeline(nodes=[get_count], engine=DaftEngine())

        # Should raise ValueError with clear message
        with pytest.raises(ValueError, match="only stateful parameters"):
            pipeline.run(inputs={"counter": counter})
