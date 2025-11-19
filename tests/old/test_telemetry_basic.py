"""
Tests for telemetry callbacks (Progress and Tracing).

Tests basic functionality without requiring actual logfire cloud connection.
"""

import time
from hypernodes import node, Pipeline
from hypernodes.telemetry import ProgressCallback


def test_progress_callback_basic():
    """Test ProgressCallback with a simple pipeline."""
    
    @node(output_name="doubled")
    def double(x: int) -> int:
        time.sleep(0.1)  # Small delay to see progress
        return x * 2
    
    @node(output_name="result")
    def add_one(doubled: int) -> int:
        time.sleep(0.1)
        return doubled + 1
    
    # Create pipeline with progress callback
    progress = ProgressCallback(enable=False)  # Disable for tests
    pipeline = Pipeline(nodes=[double, add_one], engine=SequentialEngine(callbacks=[progress]))
    
    result = pipeline.run(inputs={"x": 5})
    
    assert result == {"doubled": 10, "result": 11}


def test_progress_callback_disabled():
    """Test that ProgressCallback can be disabled for testing."""
    
    @node(output_name="result")
    def simple(x: int) -> int:
        return x + 1
    
    progress = ProgressCallback(enable=False)
    pipeline = Pipeline(nodes=[simple], engine=SequentialEngine(callbacks=[progress]))
    
    result = pipeline.run(inputs={"x": 5})
    
    # Should work without displaying progress
    assert result == {"result": 6}
    assert len(progress._bars) == 0  # No bars created


def test_telemetry_callback_import():
    """Test that TelemetryCallback can be imported."""
    try:
        from hypernodes.telemetry import TelemetryCallback
        assert TelemetryCallback is not None
    except ImportError as e:
        # Expected if logfire not installed
        assert "logfire" in str(e)


def test_waterfall_chart_no_data():
    """Test waterfall chart with no data."""
    from hypernodes.telemetry.waterfall import create_waterfall_chart
    
    try:
        fig = create_waterfall_chart([])
        assert fig is not None
        # Should create empty figure with message
    except ImportError:
        # Expected if plotly not installed
        pass


def test_waterfall_chart_with_data():
    """Test waterfall chart with sample data."""
    from hypernodes.telemetry.waterfall import create_waterfall_chart
    
    span_data = [
        {
            'name': 'node1',
            'start_time': 0.0,
            'duration': 0.5,
            'depth': 0,
            'parent': None,
            'cached': False,
            'type': 'node'
        },
        {
            'name': 'node2',
            'start_time': 0.5,
            'duration': 0.3,
            'depth': 0,
            'parent': None,
            'cached': True,
            'type': 'node'
        }
    ]
    
    try:
        fig = create_waterfall_chart(span_data)
        assert fig is not None
        # Should have traces for each span
        assert len(fig.data) >= len(span_data)
    except ImportError:
        # Expected if plotly not installed
        pass


def test_environment_detection():
    """Test environment detection utility."""
    from hypernodes.telemetry.environment import is_jupyter
    
    # In pytest, should not be Jupyter
    result = is_jupyter()
    assert isinstance(result, bool)
    # Usually False in pytest, but depends on environment
