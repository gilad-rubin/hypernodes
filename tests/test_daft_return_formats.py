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


def test_daft_engine_return_format_arrow():
    daft = pytest.importorskip("daft")  # noqa: F841 - imported for availability
    pyarrow = pytest.importorskip("pyarrow")
    from hypernodes.engines import DaftEngine

    pipeline = _build_simple_pipeline().with_engine(DaftEngine())
    table = pipeline.map(
        inputs={"x": [1, 2, 3]},
        map_over="x",
        return_format="arrow",
    )

    assert isinstance(table, pyarrow.Table)
    assert table.column("doubled").to_pylist() == [2, 4, 6]


def test_daft_engine_python_strategy_arrow():
    pytest.importorskip("daft")
    pytest.importorskip("pyarrow")
    from hypernodes.engines import DaftEngine

    pipeline = _build_simple_pipeline().with_engine(
        DaftEngine(python_return_strategy="arrow")
    )
    result = pipeline.map(inputs={"x": [1, 2]}, map_over="x", return_format="python")

    assert result == {"doubled": [2, 4]}


def test_daft_engine_python_strategy_pydict():
    pytest.importorskip("daft")
    from hypernodes.engines import DaftEngine

    pipeline = _build_simple_pipeline().with_engine(
        DaftEngine(python_return_strategy="pydict")
    )
    result = pipeline.map(inputs={"x": [3]}, map_over="x", return_format="python")

    assert result["doubled"] == [6]


def test_daft_engine_python_strategy_pandas():
    pytest.importorskip("daft")
    pytest.importorskip("pandas")
    from hypernodes.engines import DaftEngine

    pipeline = _build_simple_pipeline().with_engine(
        DaftEngine(python_return_strategy="pandas")
    )
    result = pipeline.map(inputs={"x": [4, 5]}, map_over="x", return_format="python")

    assert result["doubled"] == [8, 10]


def test_daft_engine_invalid_python_strategy():
    pytest.importorskip("daft")
    from hypernodes.engines import DaftEngine

    with pytest.raises(ValueError):
        DaftEngine(python_return_strategy="invalid")


def test_non_columnar_engine_rejects_non_python_return_format():
    pipeline = _build_simple_pipeline().with_engine(
        HypernodesEngine(node_executor="threaded", map_executor="threaded")
    )

    with pytest.raises(ValueError, match="return_format='daft'"):
        pipeline.map(inputs={"x": [1]}, map_over="x", return_format="daft")
