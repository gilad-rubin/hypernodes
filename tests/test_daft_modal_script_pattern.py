"""Test DaftEngine with script-pattern stateful classes (Modal use case).

This tests the fix for ModuleNotFoundError when using stateful objects
defined in scripts or inner functions with DaftEngine + Modal.
"""

import pytest

# Skip all tests if daft is not available
pytest.importorskip("daft")

from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine


def test_stateful_class_from_script_pattern():
    """Test that stateful class defined in function (simulating script) works."""
    
    # Define a stateful class inside the test function to simulate script pattern
    class ScriptEncoder:
        """Simulates a class defined in a script like test_modal.py."""
        __daft_hint__ = "@daft.cls"
        __daft_use_process__ = False
        
        def __init__(self, multiplier: int):
            self.multiplier = multiplier
        
        def encode(self, x: int) -> int:
            return x * self.multiplier
    
    # Create node that uses the stateful encoder
    @node(output_name="result")
    def process(x: int, encoder: ScriptEncoder) -> int:
        return encoder.encode(x)
    
    # Create encoder instance
    encoder = ScriptEncoder(multiplier=3)
    
    # Run with DaftEngine
    pipeline = Pipeline(nodes=[process], engine=DaftEngine())
    result = pipeline.run(inputs={"x": 5, "encoder": encoder})
    
    assert result["result"] == 15


def test_multiple_stateful_classes_from_script():
    """Test multiple stateful classes from same 'script' module."""
    
    # Define multiple stateful classes to simulate script pattern
    class Encoder:
        __daft_hint__ = "@daft.cls"
        
        def __init__(self, factor: int):
            self.factor = factor
        
        def encode(self, x: int) -> int:
            return x * self.factor
    
    class Normalizer:
        __daft_hint__ = "@daft.cls"
        
        def __init__(self, offset: int):
            self.offset = offset
        
        def normalize(self, x: int) -> int:
            return x - self.offset
    
    # Create nodes using both classes
    @node(output_name="encoded")
    def encode_step(x: int, encoder: Encoder) -> int:
        return encoder.encode(x)
    
    @node(output_name="result")
    def normalize_step(encoded: int, normalizer: Normalizer) -> int:
        return normalizer.normalize(encoded)
    
    # Create instances
    encoder = Encoder(factor=3)
    normalizer = Normalizer(offset=5)
    
    # Run pipeline
    pipeline = Pipeline(
        nodes=[encode_step, normalize_step],
        engine=DaftEngine()
    )
    result = pipeline.run(inputs={"x": 10, "encoder": encoder, "normalizer": normalizer})
    
    # 10 * 3 = 30, then 30 - 5 = 25
    assert result["result"] == 25


def test_nested_stateful_classes():
    """Test class containing other classes as attributes."""
    
    class InnerProcessor:
        __daft_hint__ = "@daft.cls"
        
        def __init__(self, value: int):
            self.value = value
        
        def process(self, x: int) -> int:
            return x + self.value
    
    class OuterProcessor:
        __daft_hint__ = "@daft.cls"
        
        def __init__(self, inner: InnerProcessor, multiplier: int):
            self.inner = inner
            self.multiplier = multiplier
        
        def process(self, x: int) -> int:
            temp = self.inner.process(x)
            return temp * self.multiplier
    
    @node(output_name="result")
    def process(x: int, processor: OuterProcessor) -> int:
        return processor.process(x)
    
    # Create nested instances
    inner = InnerProcessor(value=5)
    outer = OuterProcessor(inner=inner, multiplier=2)
    
    # Run pipeline
    pipeline = Pipeline(nodes=[process], engine=DaftEngine())
    result = pipeline.run(inputs={"x": 10, "processor": outer})
    
    # (10 + 5) * 2 = 30
    assert result["result"] == 30


def test_stateful_class_pickling():
    """Test that script-pattern classes can be pickled and unpickled with cloudpickle."""
    # Daft uses cloudpickle, so we need to test with cloudpickle
    try:
        import cloudpickle
    except ImportError:
        pytest.skip("cloudpickle not available")
    
    class ScriptClass:
        __daft_hint__ = "@daft.cls"
        
        def __init__(self, value: int):
            self.value = value
        
        def get_value(self) -> int:
            return self.value
    
    # Create instance
    original = ScriptClass(value=42)
    
    # Simulate what DaftEngine does - fix the instance class
    from hypernodes.integrations.daft.engine import _fix_instance_class
    fixed = _fix_instance_class(original)
    
    # Verify we can pickle and unpickle with cloudpickle
    pickled = cloudpickle.dumps(fixed)
    unpickled = cloudpickle.loads(pickled)
    
    # Verify it works correctly
    assert unpickled.get_value() == 42
    assert unpickled.value == 42


def test_script_pattern_with_map():
    """Test script-pattern stateful classes with .map() operation."""

    class Transformer:
        __daft_hint__ = "@daft.cls"
        
        def __init__(self, add_value: int):
            self.add_value = add_value
        
        def transform(self, x: int) -> int:
            return x + self.add_value
    
    @node(output_name="result")
    def transform_item(x: int, transformer: Transformer) -> int:
        return transformer.transform(x)
    
    # Create transformer
    transformer = Transformer(add_value=10)
    
    # Create list of items to map over
    items = [0, 1, 2, 3, 4]
    
    # Run pipeline with map
    pipeline = Pipeline(
        nodes=[transform_item],
        engine=DaftEngine()
    )
    result = pipeline.map(
        inputs={"x": items, "transformer": transformer},
        map_over="x"
    )
    
    # Verify all items were transformed
    assert len(result["result"]) == 5
    expected = [10, 11, 12, 13, 14]  # [0+10, 1+10, 2+10, 3+10, 4+10]
    assert result["result"] == expected


class _ScriptEvaluator:
    def __init__(self):
        self.factor = 2

    def evaluate(self, value: int) -> int:
        return value * self.factor


def test_unhinted_script_stateful_object_is_captured():
    """Even without __daft_hint__, script-style objects must be pickled by value."""

    # Simulate module defined in scripts/ folder that isn't importable in workers
    _ScriptEvaluator.__module__ = "nonexistent_script_module"
    evaluator = _ScriptEvaluator()

    @node(output_name="score")
    def compute_score(x: int, evaluator: _ScriptEvaluator) -> int:
        return evaluator.evaluate(x)

    pipeline = Pipeline(nodes=[compute_score], engine=DaftEngine())

    result = pipeline.run(inputs={"x": 5, "evaluator": evaluator})

    assert result["score"] == 10


def test_should_capture_stateful_input_for_unimportable_classes():
    class Dummy:
        pass

    Dummy.__module__ = "nonexistent_modal_script"

    engine = DaftEngine()

    assert engine._should_capture_stateful_input(Dummy()) is True


def test_make_class_serializable_rebinds_module():
    from hypernodes.integrations.daft.engine import _make_class_serializable_by_value
    import sys
    import types

    class ScriptModel:
        pass

    ScriptModel.__module__ = "custom_script_module_for_test"

    module_obj = types.SimpleNamespace(ScriptModel=ScriptModel)
    sys.modules["custom_script_module_for_test"] = module_obj

    try:
        new_cls = _make_class_serializable_by_value(ScriptModel)
        assert new_cls.__module__ == "__main__"
        assert module_obj.ScriptModel is new_cls
    finally:
        sys.modules.pop("custom_script_module_for_test", None)


def test_stateful_class_without_daft_hint():
    """Test that classes without __daft_hint__ also work."""
    
    # This simulates a class from a script that doesn't have the hint
    # DaftEngine should still handle it correctly
    class SimpleProcessor:
        def __init__(self, factor: int):
            self.factor = factor
        
        def process(self, x: int) -> int:
            return x * self.factor
    
    @node(output_name="result")
    def process(x: int, processor: SimpleProcessor) -> int:
        return processor.process(x)
    
    processor = SimpleProcessor(factor=7)
    
    # Run pipeline - should work even without __daft_hint__
    # because DaftEngine auto-detects stateful objects
    pipeline = Pipeline(nodes=[process], engine=DaftEngine())
    result = pipeline.run(inputs={"x": 6, "processor": processor})
    
    assert result["result"] == 42


def test_class_with_closure_capture():
    """Test that classes capturing variables from outer scope work."""
    
    # Outer scope variables (simulates script-level globals)
    BASE_VALUE = 100
    SCALE_FACTOR = 2
    
    class ClosureProcessor:
        __daft_hint__ = "@daft.cls"
        
        def process(self, x: int) -> int:
            # Uses variables from outer scope
            return (x + BASE_VALUE) * SCALE_FACTOR
    
    @node(output_name="result")
    def process(x: int, processor: ClosureProcessor) -> int:
        return processor.process(x)
    
    processor = ClosureProcessor()
    
    pipeline = Pipeline(nodes=[process], engine=DaftEngine())
    result = pipeline.run(inputs={"x": 50, "processor": processor})
    
    # (50 + 100) * 2 = 300
    assert result["result"] == 300


def test_pydantic_model_from_script():
    """Test Pydantic models defined in script-like context."""
    pytest.importorskip("pydantic")
    from pydantic import BaseModel
    from typing import List
    
    # Define Pydantic model in function (script pattern)
    class Document(BaseModel):
        id: str
        text: str
        score: float = 0.0
        
        model_config = {"frozen": True}
    
    @node(output_name="docs")
    def create_docs(count: int) -> List[Document]:
        return [
            Document(id=f"doc_{i}", text=f"text {i}", score=i * 1.5)
            for i in range(count)
        ]
    
    @node(output_name="total_score")
    def sum_scores(docs: List[Document]) -> float:
        return sum(doc.score for doc in docs)
    
    pipeline = Pipeline(
        nodes=[create_docs, sum_scores],
        engine=DaftEngine()
    )
    result = pipeline.run(inputs={"count": 3})
    
    # 0*1.5 + 1*1.5 + 2*1.5 = 0 + 1.5 + 3.0 = 4.5
    assert result["total_score"] == 4.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
