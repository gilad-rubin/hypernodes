"""
Phase 3 Extended: Class Instance Caching Tests

Test caching functionality with class instances, dataclasses, and custom objects.
"""

import tempfile
from dataclasses import dataclass

from hypernodes import DiskCache, Pipeline, node


# Helper to track execution
execution_log = []


def setup_function():
    """Clear execution log before each test."""
    global execution_log
    execution_log = []


def test_3_7_caching_with_dataclass_instances():
    """Test 3.7: Class instances with dataclass config produce cache hits.
    
    Validates:
    - Different class instances with same dataclass config share cache
    - Private attributes (starting with '_') are excluded from cache key
    - Changing configuration invalidates cache
    """
    global execution_log
    
    with tempfile.TemporaryDirectory() as tmpdir:
        
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
        
        pipeline = Pipeline(nodes=[encode_text], cache=DiskCache(path=tmpdir))
        
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


def test_3_8_caching_with_custom_cache_key():
    """Test 3.8: Custom __cache_key__() method controls what affects caching.
    
    Validates:
    - Custom __cache_key__() method is used when present
    - Private/secret data can be excluded from cache key
    - Internal state changes don't invalidate cache
    """
    global execution_log
    
    with tempfile.TemporaryDirectory() as tmpdir:
        
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
                # Deterministic for testing
                return f"[{self.model_name}] {prompt.upper()}"
        
        @node(output_name="result")
        def generate_text(model: ModelWithCustomKey, prompt: str) -> str:
            execution_log.append(f"generate_text('{prompt}')")
            return model.generate(prompt)
        
        pipeline = Pipeline(nodes=[generate_text], cache=DiskCache(path=tmpdir))
        
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


def test_3_9_caching_with_nested_dataclasses():
    """Test 3.9: Nested dataclass attributes are recursively serialized.
    
    Validates:
    - Nested dataclass attributes are recursively serialized
    - Changes to deeply nested fields correctly invalidate cache
    - Complex object hierarchies work correctly with caching
    """
    global execution_log
    
    with tempfile.TemporaryDirectory() as tmpdir:
        
        @dataclass
        class VectorConfig:
            dim: int
            normalize: bool
        
        @dataclass
        class EncoderConfig:
            vector_config: VectorConfig
            model_version: str
        
        class NestedEncoder:
            def __init__(self, config: EncoderConfig):
                self.config = config
            
            def encode(self, text: str) -> list:
                result = [float(ord(c)) for c in text[:self.config.vector_config.dim]]
                if self.config.vector_config.normalize:
                    total = sum(result) or 1.0
                    result = [x / total for x in result]
                return result
        
        @node(output_name="embedding")
        def encode_with_nested(encoder: NestedEncoder, text: str) -> list:
            execution_log.append(f"encode_with_nested('{text}')")
            return encoder.encode(text)
        
        pipeline = Pipeline(nodes=[encode_with_nested], cache=DiskCache(path=tmpdir))
        
        # First run
        vec_cfg1 = VectorConfig(dim=4, normalize=True)
        enc_cfg1 = EncoderConfig(vector_config=vec_cfg1, model_version="v1")
        encoder1 = NestedEncoder(enc_cfg1)
        
        result1 = pipeline.run(inputs={"encoder": encoder1, "text": "hello"})
        expected = [104.0, 101.0, 108.0, 108.0]
        total = sum(expected)
        normalized = [x / total for x in expected]
        assert result1 == {"embedding": normalized}
        assert execution_log == ["encode_with_nested('hello')"]
        
        # Second run: same nested config - should hit cache
        execution_log.clear()
        vec_cfg2 = VectorConfig(dim=4, normalize=True)
        enc_cfg2 = EncoderConfig(vector_config=vec_cfg2, model_version="v1")
        encoder2 = NestedEncoder(enc_cfg2)
        
        result2 = pipeline.run(inputs={"encoder": encoder2, "text": "hello"})
        assert result2 == {"embedding": normalized}
        assert execution_log == []  # Cache hit!
        
        # Third run: change nested field - should miss cache
        vec_cfg3 = VectorConfig(dim=4, normalize=False)  # Changed normalize
        enc_cfg3 = EncoderConfig(vector_config=vec_cfg3, model_version="v1")
        encoder3 = NestedEncoder(enc_cfg3)
        
        result3 = pipeline.run(inputs={"encoder": encoder3, "text": "hello"})
        assert result3 == {"embedding": [104.0, 101.0, 108.0, 108.0]}  # Not normalized
        assert execution_log == ["encode_with_nested('hello')"]


def test_3_10_deterministic_vs_non_deterministic_classes():
    """Test 3.10: Deterministic classes cache correctly, non-deterministic use cache=False.
    
    Validates:
    - Non-deterministic operations should use cache=False
    - Deterministic operations with seeds can be cached
    - Public seed configuration correctly affects cache key
    """
    global execution_log
    
    with tempfile.TemporaryDirectory() as tmpdir:
        
        class DeterministicEncoder:
            """Deterministic behavior with fixed seed"""
            def __init__(self, dim: int, seed: int = 42):
                self.dim = dim
                self.seed = seed
            
            def encode(self, text: str) -> list:
                # Deterministic: same seed + text -> same output
                import random
                local_rng = random.Random(self.seed)
                return [local_rng.random() for _ in range(self.dim)]
        
        @node(output_name="embedding")
        def encode_deterministic(encoder: DeterministicEncoder, text: str) -> list:
            execution_log.append(f"encode_deterministic('{text}')")
            return encoder.encode(text)
        
        pipeline = Pipeline(nodes=[encode_deterministic], cache=DiskCache(path=tmpdir))
        
        # First run with seed=42
        encoder1 = DeterministicEncoder(dim=4, seed=42)
        result1 = pipeline.run(inputs={"encoder": encoder1, "text": "hello"})
        assert len(result1["embedding"]) == 4
        assert execution_log == ["encode_deterministic('hello')"]
        
        # Second run with same seed - should hit cache
        execution_log.clear()
        encoder2 = DeterministicEncoder(dim=4, seed=42)
        result2 = pipeline.run(inputs={"encoder": encoder2, "text": "hello"})
        assert result2 == result1  # Same output
        assert execution_log == []  # Cache hit!
        
        # Third run with different seed - should miss cache
        encoder3 = DeterministicEncoder(dim=4, seed=999)
        result3 = pipeline.run(inputs={"encoder": encoder3, "text": "hello"})
        assert len(result3["embedding"]) == 4
        assert result3 != result1  # Different output
        assert execution_log == ["encode_deterministic('hello')"]


def test_3_11_private_attributes_excluded_from_cache():
    """Test 3.11: Private attributes don't affect cache key.
    
    Validates:
    - Attributes starting with '_' are excluded from cache key
    - Only public attributes affect caching
    - Internal state mutations don't invalidate cache
    """
    global execution_log
    
    with tempfile.TemporaryDirectory() as tmpdir:
        
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
        
        pipeline = Pipeline(nodes=[process_value], cache=DiskCache(path=tmpdir))
        
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
