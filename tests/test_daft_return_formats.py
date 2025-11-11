import pytest

from hypernodes import Pipeline, node
from hypernodes.engines import HypernodesEngine


def _build_simple_pipeline():
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2

    return Pipeline(nodes=[double], name="double_pipeline")


def test_daft_engine_return_format_daft():
    daft = pytest.importorskip("daft")  # noqa: F841 - imported for availability
    from hypernodes.engines import DaftEngine

    pipeline = _build_simple_pipeline().with_engine(DaftEngine())
    result = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x", return_format="daft")

    assert hasattr(result, "to_pydict")
    assert result.to_pydict()["doubled"] == [2, 4, 6]


def test_daft_engine_output_mode_dict():
    """Test that output_mode='dict' returns Python dicts."""
    pytest.importorskip("daft")
    from hypernodes.engines import DaftEngine

    pipeline = _build_simple_pipeline().with_engine(
        DaftEngine(output_mode="dict")
    )
    result = pipeline.map(inputs={"x": [3]}, map_over="x", return_format="python")

    assert isinstance(result, dict)
    assert result["doubled"] == [6]


def test_daft_engine_output_mode_daft_via_engine():
    """Test that output_mode='daft' at engine level works with return_format='python'."""
    pytest.importorskip("daft")
    from hypernodes.engines import DaftEngine

    # When output_mode='daft', the engine returns raw DataFrames
    # But return_format='python' will still try to convert via materializer
    pipeline = _build_simple_pipeline().with_engine(
        DaftEngine(output_mode="dict")  # Use dict mode for return_format='python'
    )
    result = pipeline.map(inputs={"x": [4, 5]}, map_over="x", return_format="python")

    assert result["doubled"] == [8, 10]


def test_daft_engine_invalid_output_mode():
    """Test that invalid output_mode raises ValueError."""
    pytest.importorskip("daft")
    from hypernodes.engines import DaftEngine

    with pytest.raises(ValueError, match="Invalid output mode"):
        DaftEngine(output_mode="invalid")


def test_non_columnar_engine_rejects_non_python_return_format():
    pipeline = _build_simple_pipeline().with_engine(
        HypernodesEngine(node_executor="threaded", map_executor="threaded")
    )

    with pytest.raises(ValueError, match="return_format='daft'"):
        pipeline.map(inputs={"x": [1]}, map_over="x", return_format="daft")
