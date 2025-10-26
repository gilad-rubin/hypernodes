# Callback Implementations (Pseudocode)

# Purpose

This page contains pseudocode implementations of the core callbacks to reverse-engineer the optimal callback interface and context design.

---

# ProgressCallback

```python
from tqdm import tqdm
from pipeline_system import PipelineCallback, CallbackContext

class ProgressCallback(PipelineCallback):
    def __init__(self, show_cache_hits: bool = True):
        [self.show](http://self.show)_cache_hits = show_cache_hits
        self.progress_bars = {}  # node_id -> tqdm instance
        self.pipeline_bars = {}  # pipeline_id -> tqdm instance
    
    def on_pipeline_start(self, pipeline_id: str, inputs: Dict, ctx: CallbackContext):
        # Get total node count for this pipeline
        total_nodes = ctx.get_pipeline_metadata(pipeline_id).get('total_nodes', 0)
        
        # Create indented progress bar based on depth
        indent = "  " * ctx.depth
        desc = f"{indent}Pipeline: {pipeline_id}"
        
        self.pipeline_bars[pipeline_id] = tqdm(
            total=total_nodes,
            desc=desc,
            position=ctx.depth,
            leave=True
        )
    
    def on_node_start(self, node_id: str, inputs: Dict, ctx: CallbackContext):
        # For regular nodes, just update description
        # (progress is updated in on_node_end)
        pass
    
    def on_node_end(self, node_id: str, outputs: Dict, duration: float, ctx: CallbackContext):
        # Update parent pipeline progress
        pipeline_id = ctx.current_pipeline_id
        if pipeline_id in self.pipeline_bars:
            self.pipeline_bars[pipeline_id].update(1)
            self.pipeline_bars[pipeline_id].set_postfix({
                'last': node_id,
                'time': f'{duration:.2f}s'
            })
    
    def on_node_cached(self, node_id: str, signature: str, ctx: CallbackContext):
        # Mark as cached and update progress
        pipeline_id = ctx.current_pipeline_id
        if pipeline_id in self.pipeline_bars:
            self.pipeline_bars[pipeline_id].update(1)
            if [self.show](http://self.show)_cache_hits:
                self.pipeline_bars[pipeline_id].set_postfix({
                    'last': f'{node_id} [cached]',
                    'time': '<1ms'
                })
    
    def on_nested_pipeline_start(self, parent_id: str, child_pipeline_id: str, ctx: CallbackContext):
        # Context already pushed, just wait for on_pipeline_start to be called
        pass
    
    def on_nested_pipeline_end(self, parent_id: str, child_pipeline_id: str, duration: float, ctx: CallbackContext):
        # Close the child pipeline's progress bar
        if child_pipeline_id in self.pipeline_bars:
            self.pipeline_bars[child_pipeline_id].close()
            del self.pipeline_bars[child_pipeline_id]
    
    def on_pipeline_end(self, pipeline_id: str, outputs: Dict, duration: float, ctx: CallbackContext):
        # Close progress bar for this pipeline
        if pipeline_id in self.pipeline_bars:
            self.pipeline_bars[pipeline_id].close()
            del self.pipeline_bars[pipeline_id]
    
    def on_error(self, node_id: str, error: Exception, ctx: CallbackContext):
        # Update progress bar to show error
        pipeline_id = ctx.current_pipeline_id
        if pipeline_id in self.pipeline_bars:
            self.pipeline_bars[pipeline_id].set_postfix({
                'last': f'{node_id} [ERROR]',
                'error': str(error)[:50]
            })
    
    # NEW: Map operation hooks
    def on_map_start(self, node_id: str, total_items: int, ctx: CallbackContext):
        """Called when a map operation starts"""
        indent = "  " * (ctx.depth + 1)
        desc = f"{indent}Mapping: {node_id}"
        
        self.progress_bars[f"map:{node_id}"] = tqdm(
            total=total_items,
            desc=desc,
            position=ctx.depth + 1,
            leave=False
        )
    
    def on_map_item_end(self, node_id: str, item_index: int, duration: float, ctx: CallbackContext):
        """Called after each item in a map operation completes"""
        bar_key = f"map:{node_id}"
        if bar_key in self.progress_bars:
            self.progress_bars[bar_key].update(1)
    
    def on_map_end(self, node_id: str, total_duration: float, ctx: CallbackContext):
        """Called when a map operation completes"""
        bar_key = f"map:{node_id}"
        if bar_key in self.progress_bars:
            self.progress_bars[bar_key].close()
            del self.progress_bars[bar_key]
```

---

# LoggingCallback

```python
import logging
import json
from pipeline_system import PipelineCallback, CallbackContext

class LoggingCallback(PipelineCallback):
    def __init__(self, logger: logging.Logger = None, structured: bool = True):
        self.logger = logger or logging.getLogger('pipeline_system')
        self.structured = structured
    
    def _log(self, level: str, message: str, ctx: CallbackContext, **extra):
        """Helper to log with context"""
        if self.structured:
            log_data = {
                'message': message,
                'depth': ctx.depth,
                'pipeline_path': ' > '.join(ctx.hierarchy_path),
                'current_pipeline': ctx.current_pipeline_id,
                **extra
            }
            self.logger.log(getattr(logging, level.upper()), json.dumps(log_data))
        else:
            indent = "  " * ctx.depth
            path = " > ".join(ctx.hierarchy_path)
            self.logger.log(
                getattr(logging, level.upper()),
                f"{indent}[{path}] {message}"
            )
    
    def on_pipeline_start(self, pipeline_id: str, inputs: Dict, ctx: CallbackContext):
        self._log(
            'info',
            f"Pipeline started: {pipeline_id}",
            ctx,
            pipeline_id=pipeline_id,
            input_keys=list(inputs.keys())
        )
    
    def on_node_start(self, node_id: str, inputs: Dict, ctx: CallbackContext):
        self._log(
            'debug',
            f"Node started: {node_id}",
            ctx,
            node_id=node_id,
            input_keys=list(inputs.keys())
        )
    
    def on_node_end(self, node_id: str, outputs: Dict, duration: float, ctx: CallbackContext):
        self._log(
            'debug',
            f"Node completed: {node_id} ({duration:.3f}s)",
            ctx,
            node_id=node_id,
            duration=duration,
            output_keys=list(outputs.keys())
        )
    
    def on_node_cached(self, node_id: str, signature: str, ctx: CallbackContext):
        self._log(
            'debug',
            f"Node cached: {node_id}",
            ctx,
            node_id=node_id,
            cache_signature=signature
        )
    
    def on_pipeline_end(self, pipeline_id: str, outputs: Dict, duration: float, ctx: CallbackContext):
        self._log(
            'info',
            f"Pipeline completed: {pipeline_id} ({duration:.3f}s)",
            ctx,
            pipeline_id=pipeline_id,
            duration=duration,
            output_keys=list(outputs.keys())
        )
    
    def on_error(self, node_id: str, error: Exception, ctx: CallbackContext):
        self._log(
            'error',
            f"Node failed: {node_id}",
            ctx,
            node_id=node_id,
            error_type=type(error).__name__,
            error_message=str(error)
        )
    
    def on_nested_pipeline_start(self, parent_id: str, child_pipeline_id: str, ctx: CallbackContext):
        self._log(
            'debug',
            f"Nested pipeline starting: {child_pipeline_id}",
            ctx,
            parent_id=parent_id,
            child_id=child_pipeline_id
        )
    
    def on_nested_pipeline_end(self, parent_id: str, child_pipeline_id: str, duration: float, ctx: CallbackContext):
        self._log(
            'debug',
            f"Nested pipeline completed: {child_pipeline_id} ({duration:.3f}s)",
            ctx,
            parent_id=parent_id,
            child_id=child_pipeline_id,
            duration=duration
        )
```

---

# LogfireCallback (TelemetryCallback)

```python
import logfire
from opentelemetry import context, trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from pipeline_system import PipelineCallback, CallbackContext

class LogfireCallback(PipelineCallback):
    def __init__(
        self,
        export_to: str = "cloud",
        trace_map_items: bool = True,
        sample_rate: float = 1.0,
        custom_attributes: Dict = None
    ):
        self.export_to = export_to
        self.trace_map_items = trace_map_items
        self.sample_rate = sample_rate
        self.custom_attributes = custom_attributes or {}
        
        # Initialize Logfire
        logfire.configure(send_to_logfire=(export_to == "cloud"))
        self.tracer = trace.get_tracer(__name__)
    
    def _sanitize_data(self, data: Dict, max_size: int = 1000) -> Dict:
        """Sanitize data for span attributes (remove large objects, PII, etc.)"""
        # Implementation details...
        return {k: str(v)[:max_size] for k, v in data.items()}
    
    def on_pipeline_start(self, pipeline_id: str, inputs: Dict, ctx: CallbackContext):
        # Get parent span from context (if nested)
        parent_span = ctx.get('current_span')
        
        # Create new span as child of parent (if exists)
        span = self.tracer.start_span(
            f"pipeline:{pipeline_id}",
            context=trace.set_span_in_context(parent_span) if parent_span else None
        )
        
        # Add attributes
        span.set_attribute('[pipeline.id](http://pipeline.id)', pipeline_id)
        span.set_attribute('pipeline.depth', ctx.depth)
        span.set_attribute('pipeline.path', ' > '.join(ctx.hierarchy_path))
        
        # Add custom attributes
        for key, value in self.custom_attributes.items():
            span.set_attribute(f'custom.{key}', value)
        
        # Add sanitized inputs
        sanitized_inputs = self._sanitize_data(inputs)
        for key, value in sanitized_inputs.items():
            span.set_attribute(f'input.{key}', value)
        
        # Store span in context
        ctx.set('current_span', span)
        ctx.set(f'pipeline_span:{pipeline_id}', span)
        
        # For remote execution: prepare trace context for propagation
        carrier = {}
        TraceContextTextMapPropagator().inject(carrier, context=trace.set_span_in_context(span))
        ctx.set('trace_context_carrier', carrier)
    
    def on_node_start(self, node_id: str, inputs: Dict, ctx: CallbackContext):
        # Get current span (parent)
        parent_span = ctx.get('current_span')
        
        # Create node span as child
        span = self.tracer.start_span(
            f"node:{node_id}",
            context=trace.set_span_in_context(parent_span) if parent_span else None
        )
        
        span.set_attribute('[node.id](http://node.id)', node_id)
        span.set_attribute('[pipeline.id](http://pipeline.id)', ctx.current_pipeline_id)
        span.set_attribute('depth', ctx.depth)
        
        # Add sanitized inputs
        sanitized_inputs = self._sanitize_data(inputs)
        for key, value in sanitized_inputs.items():
            span.set_attribute(f'input.{key}', value)
        
        # Store node span
        ctx.set('current_span', span)
        ctx.set(f'node_span:{node_id}', span)
    
    def on_node_end(self, node_id: str, outputs: Dict, duration: float, ctx: CallbackContext):
        # Get and close node span
        span = ctx.get(f'node_span:{node_id}')
        if span:
            span.set_attribute('duration_ms', duration * 1000)
            span.set_attribute('status', 'success')
            
            # Add sanitized outputs
            sanitized_outputs = self._sanitize_data(outputs)
            for key, value in sanitized_outputs.items():
                span.set_attribute(f'output.{key}', value)
            
            span.end()
        
        # Restore parent span (pipeline span) as current
        pipeline_span = ctx.get(f'pipeline_span:{ctx.current_pipeline_id}')
        ctx.set('current_span', pipeline_span)
    
    def on_node_cached(self, node_id: str, signature: str, ctx: CallbackContext):
        # Create a quick span for the cache hit
        parent_span = ctx.get('current_span')
        
        with self.tracer.start_as_current_span(
            f"node:{node_id}[cached]",
            context=trace.set_span_in_context(parent_span) if parent_span else None
        ) as span:
            span.set_attribute('[node.id](http://node.id)', node_id)
            span.set_attribute('cache.status', 'hit')
            span.set_attribute('cache.signature', signature)
            span.set_attribute('duration_ms', 0)
    
    def on_pipeline_end(self, pipeline_id: str, outputs: Dict, duration: float, ctx: CallbackContext):
        # Get and close pipeline span
        span = ctx.get(f'pipeline_span:{pipeline_id}')
        if span:
            span.set_attribute('duration_ms', duration * 1000)
            span.set_attribute('status', 'success')
            
            # Add sanitized outputs
            sanitized_outputs = self._sanitize_data(outputs)
            for key, value in sanitized_outputs.items():
                span.set_attribute(f'output.{key}', value)
            
            span.end()
        
        # Restore parent span as current (if nested)
        if ctx.parent_pipeline_id:
            parent_span = ctx.get(f'pipeline_span:{ctx.parent_pipeline_id}')
            ctx.set('current_span', parent_span)
    
    def on_error(self, node_id: str, error: Exception, ctx: CallbackContext):
        # Record error on current span
        span = ctx.get('current_span')
        if span:
            span.set_attribute('status', 'error')
            span.set_attribute('error.type', type(error).__name__)
            span.set_attribute('error.message', str(error))
            span.record_exception(error)
    
    def on_nested_pipeline_start(self, parent_id: str, child_pipeline_id: str, ctx: CallbackContext):
        # Context already pushed by executor
        # on_pipeline_start will be called next for the child
        pass
    
    def on_nested_pipeline_end(self, parent_id: str, child_pipeline_id: str, duration: float, ctx: CallbackContext):
        # Child pipeline span already closed in on_pipeline_end
        pass
    
    # NEW: Map operation hooks
    def on_map_start(self, node_id: str, total_items: int, ctx: CallbackContext):
        """Called when a map operation starts"""
        if not self.trace_map_items:
            return
        
        parent_span = ctx.get('current_span')
        
        # Create a span for the map operation
        span = self.tracer.start_span(
            f"map:{node_id}",
            context=trace.set_span_in_context(parent_span) if parent_span else None
        )
        span.set_attribute('[map.total](http://map.total)_items', total_items)
        span.set_attribute('[node.id](http://node.id)', node_id)
        
        ctx.set(f'map_span:{node_id}', span)
        ctx.set('current_span', span)
    
    def on_map_item_start(self, node_id: str, item_index: int, item: Any, ctx: CallbackContext):
        """Called before each item in a map operation"""
        if not self.trace_map_items:
            return
        
        # Sample based on sample_rate
        import random
        if random.random() > self.sample_rate:
            return
        
        parent_span = ctx.get(f'map_span:{node_id}')
        
        span = self.tracer.start_span(
            f"map_item:{node_id}[{item_index}]",
            context=trace.set_span_in_context(parent_span) if parent_span else None
        )
        span.set_attribute('map.item_index', item_index)
        span.set_attribute('[node.id](http://node.id)', node_id)
        
        ctx.set(f'map_item_span:{node_id}:{item_index}', span)
    
    def on_map_item_end(self, node_id: str, item_index: int, duration: float, ctx: CallbackContext):
        """Called after each item in a map operation completes"""
        span = ctx.get(f'map_item_span:{node_id}:{item_index}')
        if span:
            span.set_attribute('duration_ms', duration * 1000)
            span.end()
    
    def on_map_end(self, node_id: str, total_duration: float, ctx: CallbackContext):
        """Called when a map operation completes"""
        if not self.trace_map_items:
            return
        
        span = ctx.get(f'map_span:{node_id}')
        if span:
            span.set_attribute('duration_ms', total_duration * 1000)
            span.end()
        
        # Restore parent span
        pipeline_span = ctx.get(f'pipeline_span:{ctx.current_pipeline_id}')
        ctx.set('current_span', pipeline_span)
```

---

# Reverse-Engineered Requirements

Based on writing these implementations, here's what we need:

## CallbackContext Additions

```python
class CallbackContext:
    # ... existing methods ...
    
    def get_pipeline_metadata(self, pipeline_id: str) -> Dict:
        """Get metadata about a pipeline (e.g., total_nodes)"""
        return [self.data](http://self.data).get(f'_pipeline_metadata:{pipeline_id}', {})
    
    def set_pipeline_metadata(self, pipeline_id: str, metadata: Dict):
        """Store metadata about a pipeline (managed by executor)"""
        [self.data](http://self.data)[f'_pipeline_metadata:{pipeline_id}'] = metadata
```

## Additional Callback Methods

```python
class PipelineCallback:
    # ... existing methods ...
    
    # Map operation hooks (NEW)
    def on_map_start(self, node_id: str, total_items: int, ctx: CallbackContext):
        """Called when a map operation starts"""
        pass
    
    def on_map_item_start(self, node_id: str, item_index: int, item: Any, ctx: CallbackContext):
        """Called before processing each item in a map operation"""
        pass
    
    def on_map_item_end(self, node_id: str, item_index: int, duration: float, ctx: CallbackContext):
        """Called after processing each item in a map operation"""
        pass
    
    def on_map_item_cached(self, node_id: str, item_index: int, signature: str, ctx: CallbackContext):
        """Called when a map item is retrieved from cache"""
        pass
    
    def on_map_end(self, node_id: str, total_duration: float, ctx: CallbackContext):
        """Called when a map operation completes"""
        pass
```

## Executor Responsibilities

The executor must:

1. **Create and manage CallbackContext** for each execution
2. **Call `ctx.push_pipeline()` / `ctx.pop_pipeline()`** when entering/exiting nested pipelines
3. **Set pipeline metadata** (e.g., total_nodes) at the start via `ctx.set_pipeline_metadata()`
4. **Trigger all callback hooks** at appropriate lifecycle points
5. **Handle errors gracefully** and still call `on_error()` hooks
6. **Propagate trace context** to remote backends for distributed tracing

## Key Insights

### 1. Map Operations Need Special Hooks

Map operations have different observability needs:

- Aggregate progress (map-level)
- Per-item progress (item-level)
- Per-item tracing (optional, for debugging)
- Cache hits per item

### 2. Context Needs Pipeline Metadata

Callbacks need to know:

- Total node count (for progress bars)
- Pipeline structure (for visualization)
- Node types (regular vs map vs nested pipeline)

### 3. Span Management is Complex

The telemetry callback needs to:

- Track multiple active spans (pipeline, node, map)
- Restore parent spans after children complete
- Propagate context to remote backends
- Sample map items to reduce overhead

### 4. Remote Execution Context

For distributed tracing:

- Serialize trace context as a carrier dict
- Store in `ctx` for remote backends to access
- Remote backends must extract and reattach context

---

# Recommendations

✅ **Add map operation hooks** to the callback interface

✅ **Add `get_pipeline_metadata()` / `set_pipeline_metadata()`** to CallbackContext

✅ **Store trace context carrier** in CallbackContext for remote execution

✅ **Document executor responsibilities** for context management

✅ **Consider adding `on_map_item_cached()`** for cache hit tracking in maps