"""Tests for caching behavior with SequentialEngine."""

import tempfile
import time

from hypernodes import DiskCache, Pipeline, SequentialEngine, node

def test_basic_caching():
    """Test that caching prevents re-execution of nodes."""
    
    execution_count = []
    
    @node(output_name="result", cache=True)
    def slow_function(x: int) -> int:
        execution_count.append(1)
        return x * 2
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskCache(path=tmpdir)
        engine = SequentialEngine(cache=cache)
        pipeline = Pipeline(nodes=[slow_function], engine=engine)
        
        # First run - should execute
        result1 = pipeline.run(inputs={"x": 5})
        assert result1 == {"result": 10}
        assert len(execution_count) == 1
        
        # Second run - should use cache
        result2 = pipeline.run(inputs={"x": 5})
        assert result2 == {"result": 10}
        assert len(execution_count) == 1  # Still 1, not executed again


def test_cache_invalidation_on_input_change():
    """Test that cache is invalidated when inputs change."""
    
    execution_count = []
    
    @node(output_name="result", cache=True)
    def process(x: int) -> int:
        execution_count.append(1)
        return x * 2
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskCache(path=tmpdir)
        engine = SequentialEngine(cache=cache)
        pipeline = Pipeline(nodes=[process], engine=engine)
        
        # First input
        result1 = pipeline.run(inputs={"x": 5})
        assert result1 == {"result": 10}
        assert len(execution_count) == 1
        
        # Different input - should execute again
        result2 = pipeline.run(inputs={"x": 10})
        assert result2 == {"result": 20}
        assert len(execution_count) == 2
        
        # Original input again - should use cache
        result3 = pipeline.run(inputs={"x": 5})
        assert result3 == {"result": 10}
        assert len(execution_count) == 2  # Still 2


def test_selective_caching():
    """Test that cache=False prevents caching for specific nodes."""
    
    execution_count = {"cached": [], "uncached": []}
    
    @node(output_name="cached_result", cache=True)
    def cached_function(x: int) -> int:
        execution_count["cached"].append(1)
        return x * 2
    
    @node(output_name="uncached_result", cache=False)
    def uncached_function(cached_result: int) -> int:
        execution_count["uncached"].append(1)
        return cached_result + 1
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskCache(path=tmpdir)
        engine = SequentialEngine(cache=cache)
        pipeline = Pipeline(nodes=[cached_function, uncached_function], engine=engine)
        
        # First run
        pipeline.run(inputs={"x": 5})
        assert len(execution_count["cached"]) == 1
        assert len(execution_count["uncached"]) == 1
        
        # Second run - cached node uses cache, uncached runs again
        pipeline.run(inputs={"x": 5})
        assert len(execution_count["cached"]) == 1  # Used cache
        assert len(execution_count["uncached"]) == 2  # Ran again


def test_nested_pipeline_cache_inheritance():
    """Test that nested pipelines inherit parent's cache."""
    
    execution_count = []
    
    @node(output_name="doubled", cache=True)
    def double(x: int) -> int:
        execution_count.append(1)
        return x * 2
    
    # Inner pipeline with no cache specified
    inner = Pipeline(nodes=[double])
    
    @node(output_name="result")
    def add_one(doubled: int) -> int:
        return doubled + 1
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskCache(path=tmpdir)
        engine = SequentialEngine(cache=cache)
        
        # Outer pipeline with cache - inner should inherit
        outer = Pipeline(
            nodes=[inner.as_node(), add_one],
            engine=engine
        )
        
        # First run - inner node executes
        result1 = outer.run(inputs={"x": 5})
        assert result1 == {"doubled": 10, "result": 11}
        assert len(execution_count) == 1
        
        # Second run - inner node should use cache (inherited)
        result2 = outer.run(inputs={"x": 5})
        assert result2 == {"doubled": 10, "result": 11}
        assert len(execution_count) == 1  # Still 1, used cache


def test_cache_with_map():
    """Test that caching works correctly with map operations."""
    
    execution_count = []
    
    @node(output_name="result", cache=True)
    def expensive_operation(x: int) -> int:
        execution_count.append(x)
        time.sleep(0.01)  # Simulate expensive operation
        return x * 2
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskCache(path=tmpdir)
        engine = SequentialEngine(cache=cache)
        pipeline = Pipeline(nodes=[expensive_operation], engine=engine)
        
        # First map - all items execute
        results1 = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
        assert results1 == [
            {"result": 2},
            {"result": 4},
            {"result": 6},
        ]
        assert len(execution_count) == 3
        
        # Second map with same items - all use cache
        results2 = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
        assert results2 == results1
        assert len(execution_count) == 3  # Still 3, used cache
        
        # Third map with partial overlap - only new item executes
        results3 = pipeline.map(inputs={"x": [2, 3, 4]}, map_over="x")
        assert results3 == [
            {"result": 4},
            {"result": 6},
            {"result": 8},
        ]
        assert len(execution_count) == 4  # Only item 4 executed
        assert execution_count[-1] == 4


def test_code_change_invalidates_cache():
    """Test that changing function code invalidates cache."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskCache(path=tmpdir)
        engine = SequentialEngine(cache=cache)
        
        # First version: add 1
        @node(output_name="result")
        def add_x(x: int) -> int:
            return x + 1
        
        pipeline = Pipeline(nodes=[add_x], engine=engine)
        result1 = pipeline.run(inputs={"x": 5})
        assert result1 == {"result": 6}
        
        # Run again - should use cache
        result2 = pipeline.run(inputs={"x": 5})
        assert result2 == {"result": 6}
        
        # Create new pipeline with "modified" function (add 2 instead)
        @node(output_name="result")
        def add_x_v2(x: int) -> int:
            return x + 2  # Different logic
        
        pipeline2 = Pipeline(nodes=[add_x_v2], engine=engine)
        
        # Should execute with new code
        result3 = pipeline2.run(inputs={"x": 5})
        assert result3 == {"result": 7}  # Different result!


def test_diamond_pattern_with_cache():
    """Test that cache works correctly with diamond dependency pattern.
    
    In a diamond pattern:
       input
       /   \\
      A     B
       \\   /
        C
    
    Each node should be cached independently and cache revalidation
    should follow the dependency graph correctly.
    """
    
    execution_log = []
    
    @node(output_name="doubled")
    def double(x: int) -> int:
        execution_log.append(f"double({x})")
        return x * 2
    
    @node(output_name="tripled")
    def triple(x: int) -> int:
        execution_log.append(f"triple({x})")
        return x * 3
    
    @node(output_name="result")
    def add(doubled: int, tripled: int) -> int:
        execution_log.append(f"add({doubled},{tripled})")
        return doubled + tripled
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskCache(path=tmpdir)
        engine = SequentialEngine(cache=cache)
        pipeline = Pipeline(nodes=[double, triple, add], engine=engine)
        
        # First run - all execute
        result1 = pipeline.run(inputs={"x": 5})
        assert result1 == {"doubled": 10, "tripled": 15, "result": 25}
        assert execution_log == ["double(5)", "triple(5)", "add(10,15)"]
        
        # Second run - all cached
        execution_log.clear()
        result2 = pipeline.run(inputs={"x": 5})
        assert result2 == {"doubled": 10, "tripled": 15, "result": 25}
        assert execution_log == []
        
        # Third run with different input - all execute
        result3 = pipeline.run(inputs={"x": 10})
        assert result3 == {"doubled": 20, "tripled": 30, "result": 50}
        assert execution_log == ["double(10)", "triple(10)", "add(20,30)"]


def test_caching_with_dataclass_instances():
    """Test that class instances with dataclass config produce cache hits.
    
    Important for ML workflows where models have configuration objects.
    Different instances with same config should share cache.
    Private attributes (starting with '_') should be excluded from cache key.
    """
    from dataclasses import dataclass
    
    execution_log = []
    
    @dataclass
    class EncoderConfig:
        dim: int
        model_name: str
    
    class DeterministicEncoder:
        def __init__(self, config: EncoderConfig):
            self.config = config
            self.dim = config.dim
            self._internal_cache = {}  # Private, excluded from hash
        
        def encode(self, text: str) -> list:
            # Deterministic: same text -> same output
            return [float(ord(c)) for c in text[:self.dim]]
    
    @node(output_name="embedding")
    def encode_text(encoder: DeterministicEncoder, text: str) -> list:
        execution_log.append(f"encode_text('{text}')")
        return encoder.encode(text)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskCache(path=tmpdir)
        engine = SequentialEngine(cache=cache)
        pipeline = Pipeline(nodes=[encode_text], engine=engine)
        
        # First run with encoder instance
        config1 = EncoderConfig(dim=4, model_name="test-v1")
        encoder1 = DeterministicEncoder(config1)
        
        result1 = pipeline.run(inputs={"encoder": encoder1, "text": "hello"})
        assert result1 == {"embedding": [104.0, 101.0, 108.0, 108.0]}
        assert execution_log == ["encode_text('hello')"]
        
        # Second run with SAME config but NEW instance - should hit cache
        execution_log.clear()
        config2 = EncoderConfig(dim=4, model_name="test-v1")
        encoder2 = DeterministicEncoder(config2)
        
        result2 = pipeline.run(inputs={"encoder": encoder2, "text": "hello"})
        assert result2 == {"embedding": [104.0, 101.0, 108.0, 108.0]}
        assert execution_log == []  # Cache hit!
        
        # Third run with DIFFERENT config - should miss cache
        config3 = EncoderConfig(dim=4, model_name="test-v2")
        encoder3 = DeterministicEncoder(config3)
        
        result3 = pipeline.run(inputs={"encoder": encoder3, "text": "hello"})
        assert result3 == {"embedding": [104.0, 101.0, 108.0, 108.0]}
        assert execution_log == ["encode_text('hello')"]


def test_caching_with_custom_cache_key():
    """Test that custom __cache_key__() method controls what affects caching.
    
    This allows advanced users to:
    - Exclude secrets/API keys from cache keys
    - Define custom cache semantics
    - Control exactly what invalidates cache
    """
    
    execution_log = []
    
    class ModelWithCustomKey:
        def __init__(self, model_name: str, temperature: float, api_key: str):
            self.model_name = model_name
            self.temperature = temperature
            self._api_key = api_key  # Secret, should not affect cache
            self._call_count = 0
        
        def __cache_key__(self) -> str:
            import json
            # Only model_name and temperature affect caching
            return f"{self.__class__.__name__}::{json.dumps({
                'model': self.model_name,
                'temp': self.temperature
            }, sort_keys=True)}"
        
        def generate(self, prompt: str) -> str:
            self._call_count += 1
            return f"[{self.model_name}] {prompt.upper()}"
    
    @node(output_name="result")
    def generate_text(model: ModelWithCustomKey, prompt: str) -> str:
        execution_log.append(f"generate_text('{prompt}')")
        return model.generate(prompt)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskCache(path=tmpdir)
        engine = SequentialEngine(cache=cache)
        pipeline = Pipeline(nodes=[generate_text], engine=engine)
        
        # First run
        model1 = ModelWithCustomKey("gpt-4", 0.7, "secret-key-123")
        result1 = pipeline.run(inputs={"model": model1, "prompt": "hello"})
        assert result1 == {"result": "[gpt-4] HELLO"}
        assert execution_log == ["generate_text('hello')"]
        
        # Second run: different API key, but same model config - should hit cache
        execution_log.clear()
        model2 = ModelWithCustomKey("gpt-4", 0.7, "different-key-456")
        result2 = pipeline.run(inputs={"model": model2, "prompt": "hello"})
        assert result2 == {"result": "[gpt-4] HELLO"}
        assert execution_log == []  # Cache hit!
        
        # Third run: different temperature - should miss cache
        model3 = ModelWithCustomKey("gpt-4", 0.9, "secret-key-123")
        result3 = pipeline.run(inputs={"model": model3, "prompt": "hello"})
        assert result3 == {"result": "[gpt-4] HELLO"}
        assert execution_log == ["generate_text('hello')"]


def test_private_attributes_excluded_from_cache():
    """Test that private attributes (starting with '_') don't affect cache key.
    
    This ensures:
    - Internal state mutations don't invalidate cache
    - Secrets and sensitive data can be safely excluded
    - Only public configuration affects caching
    """
    
    execution_log = []
    
    class ProcessorWithState:
        def __init__(self, config: str):
            self.config = config  # Public - affects cache
            self._internal_counter = 0  # Private - doesn't affect cache
            self._cache = {}  # Private - doesn't affect cache
        
        def process(self, value: int) -> int:
            self._internal_counter += 1
            return value * 2
    
    @node(output_name="result")
    def process_value(processor: ProcessorWithState, value: int) -> int:
        execution_log.append(f"process_value({value})")
        return processor.process(value)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskCache(path=tmpdir)
        engine = SequentialEngine(cache=cache)
        pipeline = Pipeline(nodes=[process_value], engine=engine)
        
        # First run
        proc1 = ProcessorWithState(config="v1")
        result1 = pipeline.run(inputs={"processor": proc1, "value": 5})
        assert result1 == {"result": 10}
        assert execution_log == ["process_value(5)"]
        
        # Second run: same config, different internal state - should hit cache
        execution_log.clear()
        proc2 = ProcessorWithState(config="v1")
        proc2._internal_counter = 100  # Different internal state
        proc2._cache = {"some": "data"}  # Different internal state
        
        result2 = pipeline.run(inputs={"processor": proc2, "value": 5})
        assert result2 == {"result": 10}
        assert execution_log == []  # Cache hit! Private attrs ignored
        
        # Third run: different config - should miss cache
        proc3 = ProcessorWithState(config="v2")
        result3 = pipeline.run(inputs={"processor": proc3, "value": 5})
        assert result3 == {"result": 10}
        assert execution_log == ["process_value(5)"]

