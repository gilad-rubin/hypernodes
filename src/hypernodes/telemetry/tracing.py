"""Telemetry and tracing callback using Logfire."""

from typing import Dict, Optional, Any, List
import time

from ..callbacks import PipelineCallback, CallbackContext


class TelemetryCallback(PipelineCallback):
    """Distributed tracing with Logfire (OpenTelemetry-based).
    
    Uses logfire for creating spans and tracking execution.
    User must call logfire.configure() before using this callback.
    
    Features:
    - Hierarchical span tree (automatic parent-child relationships)
    - Tracks: inputs, outputs, duration, cache hits, errors
    - Collects span data for waterfall chart generation
    - Context propagation (for future remote backends)
    
    Example:
        >>> import logfire
        >>> from hypernodes import Pipeline
        >>> from hypernodes.telemetry import TelemetryCallback
        >>> 
        >>> # Configure logfire (user controls settings)
        >>> logfire.configure()  # or logfire.configure(send_to_logfire=False)
        >>> 
        >>> pipeline = Pipeline(
        ...     nodes=[node1, node2],
        ...     callbacks=[TelemetryCallback()]
        ... )
        >>> result = pipeline.run(inputs={...})
        >>> 
        >>> # Generate waterfall chart (Jupyter only)
        >>> telemetry = pipeline.callbacks[0]
        >>> chart = telemetry.get_waterfall_chart()
        >>> chart  # Auto-displays in Jupyter
    """
    
    def __init__(
        self,
        trace_map_items: bool = True,
        logfire_instance: Optional[Any] = None
    ):
        """Initialize telemetry callback.
        
        Args:
            trace_map_items: Whether to create spans for individual map items
            logfire_instance: Optional logfire instance (uses default if None)
        """
        try:
            import logfire
            self.logfire = logfire_instance or logfire
        except ImportError:
            raise ImportError(
                "logfire is not installed. Install with: pip install 'hypernodes[telemetry]'"
            )
        
        self.trace_map_items = trace_map_items
        self.span_data: List[Dict[str, Any]] = []  # For waterfall charts
    
    def on_pipeline_start(self, pipeline_id: str, inputs: Dict, ctx: CallbackContext) -> None:
        """Create span for pipeline execution."""
        # Logfire automatically handles parent context from OpenTelemetry
        span = self.logfire.span(
            f'pipeline:{pipeline_id}',
            pipeline_id=pipeline_id,
            depth=ctx.depth,
            inputs=str(inputs)[:500]  # Truncate large inputs
        ).__enter__()
        
        # Store for children and waterfall data
        ctx.set('current_span', span)
        ctx.set(f'span:{pipeline_id}', span)
        ctx.set(f'span_start:{pipeline_id}', time.time())
    
    def on_node_start(self, node_id: str, inputs: Dict, ctx: CallbackContext) -> None:
        """Create span for node execution."""
        # Logfire automatically handles parent from OpenTelemetry context
        span = self.logfire.span(
            f'node:{node_id}',
            node_id=node_id,
            depth=ctx.depth
        ).__enter__()
        
        ctx.set(f'span:{node_id}', span)
        ctx.set(f'span_start:{node_id}', time.time())
    
    def on_node_end(self, node_id: str, outputs: Dict, duration: float, ctx: CallbackContext) -> None:
        """Close node span and collect data."""
        span = ctx.get(f'span:{node_id}')
        if span:
            span.set_attribute('duration', duration)
            span.set_attribute('outputs', str(outputs)[:500])
            span.__exit__(None, None, None)
        
        # Collect for waterfall
        start_time = ctx.get(f'span_start:{node_id}')
        if start_time:
            self.span_data.append({
                'name': node_id,
                'start_time': start_time,
                'duration': duration,
                'depth': ctx.depth,
                'parent': ctx.current_pipeline_id,
                'cached': False,
                'type': 'node'
            })
    
    def on_node_cached(self, node_id: str, signature: str, ctx: CallbackContext) -> None:
        """Handle cached node."""
        span = ctx.get(f'span:{node_id}')
        if span:
            span.set_attribute('cached', True)
            span.set_attribute('cache_signature', signature)
            span.__exit__(None, None, None)
        
        # Collect for waterfall (cached nodes have ~0 duration)
        start_time = ctx.get(f'span_start:{node_id}')
        if start_time:
            self.span_data.append({
                'name': node_id,
                'start_time': start_time,
                'duration': 0.001,  # minimal duration for cached
                'depth': ctx.depth,
                'parent': ctx.current_pipeline_id,
                'cached': True,
                'type': 'node'
            })
    
    def on_pipeline_end(self, pipeline_id: str, outputs: Dict, duration: float, ctx: CallbackContext) -> None:
        """Close pipeline span."""
        span = ctx.get(f'span:{pipeline_id}')
        if span:
            span.set_attribute('duration', duration)
            span.set_attribute('outputs', str(outputs)[:500])
            span.__exit__(None, None, None)
        
        # Collect for waterfall
        start_time = ctx.get(f'span_start:{pipeline_id}')
        if start_time:
            self.span_data.append({
                'name': pipeline_id,
                'start_time': start_time,
                'duration': duration,
                'depth': ctx.depth,
                'parent': ctx.parent_pipeline_id,
                'cached': False,
                'type': 'pipeline'
            })
    
    def on_error(self, node_id: str, error: Exception, ctx: CallbackContext) -> None:
        """Record error in span."""
        span = ctx.get(f'span:{node_id}')
        if span:
            span.set_attribute('error', str(error))
            span.set_attribute('error_type', type(error).__name__)
            # Logfire will automatically record the exception
            span.__exit__(type(error), error, error.__traceback__)
        else:
            # Span already closed, just log
            self.logfire.error(
                f'Error in {node_id}: {error}',
                node_id=node_id,
                error_type=type(error).__name__
            )
    
    def on_map_start(self, total_items: int, ctx: CallbackContext) -> None:
        """Create span for map operation."""
        span = self.logfire.span(
            'map_operation',
            total_items=total_items,
            depth=ctx.depth
        ).__enter__()
        
        ctx.set('map_span', span)
        ctx.set('map_start_time', time.time())
    
    def on_map_item_start(self, item_index: int, ctx: CallbackContext) -> None:
        """Create span for individual map item (if enabled)."""
        if not self.trace_map_items:
            return
        
        span = self.logfire.span(
            f'map_item[{item_index}]',
            item_index=item_index,
            depth=ctx.depth + 1
        ).__enter__()
        
        ctx.set(f'map_item_span:{item_index}', span)
    
    def on_map_item_end(self, item_index: int, duration: float, ctx: CallbackContext) -> None:
        """Close map item span."""
        if not self.trace_map_items:
            return
        
        span = ctx.get(f'map_item_span:{item_index}')
        if span:
            span.set_attribute('duration', duration)
            span.__exit__(None, None, None)
    
    def on_map_item_cached(self, item_index: int, signature: str, ctx: CallbackContext) -> None:
        """Handle cached map item."""
        if not self.trace_map_items:
            return
        
        span = ctx.get(f'map_item_span:{item_index}')
        if span:
            span.set_attribute('cached', True)
            span.set_attribute('cache_signature', signature)
            span.__exit__(None, None, None)
    
    def on_map_end(self, total_duration: float, ctx: CallbackContext) -> None:
        """Close map operation span."""
        span = ctx.get('map_span')
        if span:
            span.set_attribute('total_duration', total_duration)
            span.__exit__(None, None, None)
        
        # Collect for waterfall
        start_time = ctx.get('map_start_time')
        if start_time:
            self.span_data.append({
                'name': 'map_operation',
                'start_time': start_time,
                'duration': total_duration,
                'depth': ctx.depth,
                'parent': ctx.current_pipeline_id,
                'cached': False,
                'type': 'map'
            })
    
    def get_waterfall_chart(self):
        """Generate waterfall chart from collected span data.
        
        Only works in Jupyter notebooks. Returns Plotly figure.
        
        Returns:
            Plotly figure (auto-displays in Jupyter)
        
        Raises:
            ImportError: If plotly is not installed
        """
        from .waterfall import create_waterfall_chart
        return create_waterfall_chart(self.span_data)
