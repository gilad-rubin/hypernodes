"""Tests for Phase 2: Engine Architecture Renaming.

This test file verifies that the renaming from Backend/Executor to Engine works correctly.
Following TDD: These tests are written first, then implementation follows.
"""

import pytest

from hypernodes import Pipeline, node
from hypernodes.engines import HypernodesEngine


def test_engine_base_class_exists():
    """Verify Engine ABC exists and can be imported."""
    from abc import ABC

    from hypernodes.engine import Engine

    assert issubclass(Engine, ABC)
    assert hasattr(Engine, 'run')
    assert hasattr(Engine, 'map')


def test_engine_abstract_methods():
    """Verify Engine has required abstract methods with correct signatures."""
    import inspect

    from hypernodes.engine import Engine

    # Check run method signature
    run_sig = inspect.signature(Engine.run)
    assert 'pipeline' in run_sig.parameters
    assert 'inputs' in run_sig.parameters
    assert 'output_name' in run_sig.parameters
    assert '_ctx' in run_sig.parameters  # Should be prefixed with underscore

    # Check map method signature
    map_sig = inspect.signature(Engine.map)
    assert 'pipeline' in map_sig.parameters
    assert 'items' in map_sig.parameters
    assert 'inputs' in map_sig.parameters
    assert 'output_name' in map_sig.parameters
    assert '_ctx' in map_sig.parameters  # Should be prefixed with underscore


def test_hypernodes_engine_exists():
    """Verify HypernodesEngine exists and inherits from Engine."""
    from hypernodes.engine import Engine
    from hypernodes.engines import HypernodesEngine

    assert issubclass(HypernodesEngine, Engine)


def test_hypernodes_engine_creation_default():
    """Verify HypernodesEngine can be created with defaults."""
    engine = HypernodesEngine()

    # Should have default sequential executors
    assert engine is not None


def test_hypernodes_engine_has_executor_parameters():
    """Verify HypernodesEngine has execution parameters.

    After refactoring, we use the new parameter names (node_executor/map_executor).
    """
    import inspect

    sig = inspect.signature(HypernodesEngine.__init__)

    # Should have new parameter names
    assert 'node_executor' in sig.parameters
    assert 'map_executor' in sig.parameters
    assert 'max_workers' in sig.parameters


def test_hypernodes_engine_basic_execution():
    """Verify HypernodesEngine can execute a simple pipeline.

    Note: Using backend= for now. Phase 5 will update Pipeline to use engine=.
    """

    @node(output_name="result")
    def double(x: int) -> int:
        return x * 2

    engine = HypernodesEngine()
    pipeline = Pipeline(nodes=[double], backend=engine)  # Phase 5 will change to engine=

    result = pipeline.run(inputs={"x": 5})
    assert result["result"] == 10


def test_engine_exports():
    """Verify correct classes are exported from hypernodes.engines."""
    from hypernodes import engines

    # Should export new names
    assert hasattr(engines, 'Engine')
    assert hasattr(engines, 'HypernodesEngine')

    # Should NOT export old names
    assert not hasattr(engines, 'Executor')
    assert not hasattr(engines, 'LocalExecutor')


def test_main_package_exports():
    """Verify correct classes are exported from main hypernodes package."""
    import hypernodes

    # Should export HypernodesEngine
    assert hasattr(hypernodes, 'HypernodesEngine')

    # Should NOT export old Backend/LocalBackend
    assert not hasattr(hypernodes, 'Backend')
    assert not hasattr(hypernodes, 'LocalBackend')


def test_ctx_parameter_is_private():
    """Verify _ctx parameter is clearly marked as internal in Engine interface."""
    import inspect

    from hypernodes.engine import Engine

    # Check run method
    run_sig = inspect.signature(Engine.run)
    ctx_param = run_sig.parameters.get('_ctx')
    assert ctx_param is not None, "_ctx parameter should exist in run()"

    # Check map method
    map_sig = inspect.signature(Engine.map)
    ctx_param = map_sig.parameters.get('_ctx')
    assert ctx_param is not None, "_ctx parameter should exist in map()"


def test_hypernodes_engine_sequential_execution():
    """Verify HypernodesEngine executes nodes in correct order.

    Note: Using backend= for now. Phase 5 will update Pipeline to use engine=.
    """
    execution_order = []

    @node(output_name="a")
    def step_a() -> str:
        execution_order.append("a")
        return "a"

    @node(output_name="b")
    def step_b(a: str) -> str:
        execution_order.append("b")
        return a + "b"

    @node(output_name="c")
    def step_c(b: str) -> str:
        execution_order.append("c")
        return b + "c"

    engine = HypernodesEngine()
    pipeline = Pipeline(nodes=[step_a, step_b, step_c], backend=engine)  # Phase 5 will change to engine=

    result = pipeline.run(inputs={})

    assert result["c"] == "abc"
    assert execution_order == ["a", "b", "c"]


def test_hypernodes_engine_map_execution():
    """Verify HypernodesEngine can execute map operations.

    Note: Using backend= for now. Phase 5 will update Pipeline to use engine=.
    """

    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    engine = HypernodesEngine()
    pipeline = Pipeline(nodes=[double], backend=engine)  # Phase 5 will change to engine=

    result = pipeline.map(
        inputs={"x": [1, 2, 3]},
        map_over="x",
    )

    assert result["doubled"] == [2, 4, 6]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
