# Intelligent Callback System

# Architecture

Callbacks provide hooks into the pipeline execution lifecycle. **Note:** Caching is **core execution logic**, not a callback—the executor handles cache checking and loading, then triggers `on_node_cached` for observability.

## CallbackContext

Callbacks share state via a `CallbackContext` object passed to each callback method. The context automatically tracks hierarchical execution to enable seamless nested pipeline handling.

```python
class CallbackContext:
    """Shared state across all callbacks in a single execution"""
    def __init__(self):
        [self.data](http://self.data): Dict[str, Any] = {}
        self._hierarchy_stack: List[str] = []  # Stack of pipeline IDs
        self._depth: int = 0
    
    def set(self, key: str, value: Any):
        """Store a value for other callbacks to access"""
        [self.data](http://self.data)[key] = value
    
    def get(self, key: str, default=None) -> Any:
        """Retrieve a value set by another callback"""
        return [self.data](http://self.data).get(key, default)
    
    def push_pipeline(self, pipeline_id: str):
        """Track entering a nested pipeline (managed by executor)"""
        self._hierarchy_stack.append(pipeline_id)
        self._depth += 1
    
    def pop_pipeline(self) -> str:
        """Track exiting a nested pipeline (managed by executor)"""
        self._depth -= 1
        return self._hierarchy_stack.pop()
    
    def get_pipeline_metadata(self, pipeline_id: str) -> Dict:
        """Get metadata about a pipeline (e.g., total_nodes)"""
        return [self.data](http://self.data).get(f'_pipeline_metadata:{pipeline_id}', {})
    
    def set_pipeline_metadata(self, pipeline_id: str, metadata: Dict):
        """Store metadata about a pipeline (managed by executor)"""
        [self.data](http://self.data)[f'_pipeline_metadata:{pipeline_id}'] = metadata
    
    @property
    def current_pipeline_id(self) -> str:
        """Get the currently executing pipeline ID"""
        return self._hierarchy_stack[-1] if self._hierarchy_stack else None
    
    @property
    def parent_pipeline_id(self) -> str:
        """Get the parent pipeline ID (None if at root)"""
        return self._hierarchy_stack[-2] if len(self._hierarchy_stack) >= 2 else None
    
    @property
    def depth(self) -> int:
        """Current nesting depth (0 = root pipeline)"""
        return self._depth
    
    @property
    def hierarchy_path(self) -> List[str]:
        """Full path from root to current pipeline"""
        return self._hierarchy_stack.copy()
```

## Callback Interface

```python
from pipeline_system import Callback, CallbackContext

class PipelineCallback:
    def on_pipeline_start(self, pipeline_id: str, inputs: Dict, ctx: CallbackContext):
        """Called when pipeline execution starts"""
        pass
    
    def on_node_start(self, node_id: str, inputs: Dict, ctx: CallbackContext):
        """Called before a node executes"""
        pass
    
    def on_node_end(self, node_id: str, outputs: Dict, duration: float, ctx: CallbackContext):
        """Called after a node executes"""
        pass
    
    def on_node_cached(self, node_id: str, signature: str, ctx: CallbackContext):
        """Called when a node is skipped due to cache hit (caching is core, not a callback)"""
        pass
    
    def on_pipeline_end(self, pipeline_id: str, outputs: Dict, duration: float, ctx: CallbackContext):
        """Called when pipeline execution completes"""
        pass
    
    def on_error(self, node_id: str, error: Exception, ctx: CallbackContext):
        """Called when a node raises an error"""
        pass
    
    def on_nested_pipeline_start(self, parent_id: str, child_pipeline_id: str, ctx: CallbackContext):
        """Called when a nested pipeline starts execution
        Note: ctx.push_pipeline() is already called by executor before this hook
        """
        pass
    
    def on_nested_pipeline_end(self, parent_id: str, child_pipeline_id: str, duration: float, ctx: CallbackContext):
        """Called when a nested pipeline completes execution
        Note: ctx.pop_pipeline() is called by executor after this hook
        """
        pass
    
    # Map operation hooks
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

---

# Built-in Callbacks

## ProgressCallback

- Renders progress bars via tqdm
- Auto-detects environment: [`tqdm.rich`](http://tqdm.rich) for CLI, `tqdm.notebook` for Jupyter
- Shows parallel node execution
- Shows map operation progress
- Hierarchy-aware: indents nested pipeline progress bars

## LoggingCallback

Simple structured logging for debugging and monitoring:

- Outputs to stdout, files, or log aggregators (e.g., Logstash, CloudWatch)
- Structured JSON logs with timestamps and severity
- Hierarchy-aware: logs include `depth` and `pipeline_path` fields
- Lightweight alternative to full telemetry

**When to use:**

- Simple debugging and troubleshooting
- Text-based log analysis
- Traditional logging infrastructure
- Development and testing

## TelemetryCallback

Advanced distributed tracing and observability with OpenTelemetry:

- Hierarchical span tracking with automatic parent-child relationships
- Distributed tracing across remote backends (Coiled, Modal)
- Rich structured logs attached to spans
- Metrics and performance data
- Real-time visualization in tools like Jaeger, Zipkin, Logfire
- Automatic context propagation

**When to use:**

- Production monitoring and debugging
- Distributed/remote execution
- Performance analysis
- Complex nested pipelines
- Integration with APM tools

**Hierarchy handling:**

The TelemetryCallback automatically creates parent-child span relationships using the context stack:

```python
class TelemetryCallback:
    def on_pipeline_start(self, pipeline_id: str, inputs: Dict, ctx: CallbackContext):
        # Get parent span from context (if nested)
        parent_span = ctx.get('current_span')
        
        # Create new span as child of parent (if exists)
        span = tracer.start_span(
            f"pipeline:{pipeline_id}",
            parent=parent_span
        )
        
        # Store as current span
        ctx.set('current_span', span)
        ctx.set(f'span:{pipeline_id}', span)
    
    def on_nested_pipeline_start(self, parent_id: str, child_pipeline_id: str, ctx: CallbackContext):
        # Context already pushed by executor
        # Parent span is still in 'current_span'
        pass
    
    def on_pipeline_end(self, pipeline_id: str, outputs: Dict, duration: float, ctx: CallbackContext):
        span = ctx.get(f'span:{pipeline_id}')
        if span:
            span.set_attribute('duration', duration)
            span.end()
        
        # Restore parent span as current (if nested)
        if ctx.parent_pipeline_id:
            parent_span = ctx.get(f'span:{ctx.parent_pipeline_id}')
            ctx.set('current_span', parent_span)
```

See [Tracing & Telemetry](Tracing%20&%20Telemetry%20da0bddf3d656448e99f2b968fd8c2b49.md) for details.

## MetricsCallback

- Tracks execution time per node
- Memory usage, CPU utilization
- Exports to Prometheus, Datadog, etc.
- Can attach metrics to telemetry spans when used together

---

# Hierarchy-Aware Example

```python
class LoggingCallback:
    def on_pipeline_start(self, pipeline_id: str, inputs: Dict, ctx: CallbackContext):
        indent = "  " * ctx.depth
        path = " > ".join(ctx.hierarchy_path)
        [logger.info](http://logger.info)(f"{indent}[{path}] Pipeline started: {pipeline_id}")
    
    def on_node_start(self, node_id: str, inputs: Dict, ctx: CallbackContext):
        indent = "  " * ctx.depth
        [logger.info](http://logger.info)(f"{indent}  → Node: {node_id}")
    
    def on_node_end(self, node_id: str, outputs: Dict, duration: float, ctx: CallbackContext):
        indent = "  " * ctx.depth
        [logger.info](http://logger.info)(f"{indent}  ✓ Node: {node_id} ({duration:.2f}s)")

# Output:
# [root_pipeline] Pipeline started: root_pipeline
#   → Node: load_data
#   ✓ Node: load_data (0.5s)
#   → Node: process_data (nested pipeline)
#     [root_pipeline > process_data] Pipeline started: process_data
#       → Node: validate
#       ✓ Node: validate (0.1s)
#       → Node: transform
#       ✓ Node: transform (0.3s)
#     [root_pipeline > process_data] Pipeline ended (0.4s)
#   ✓ Node: process_data (0.4s)
```

---

# Shared Context Example

```python
class TelemetryCallback:
    def on_node_start(self, node_id: str, inputs: Dict, ctx: CallbackContext):
        parent_span = ctx.get('current_span')
        span = tracer.start_span(node_id, parent=parent_span)
        ctx.set('current_span', span)
        ctx.set(f'node_span:{node_id}', span)

class ProgressCallback:
    def on_node_start(self, node_id: str, inputs: Dict, ctx: CallbackContext):
        span = ctx.get('current_span')
        if span:
            # Report progress as span event
            span.add_event(f"Starting {node_id}")
        
        # Indent progress bar based on depth
        indent = "  " * ctx.depth
        self.progress_bars[node_id] = tqdm(desc=f"{indent}{node_id}")

class MetricsCallback:
    def on_node_end(self, node_id: str, outputs: Dict, duration: float, ctx: CallbackContext):
        # Attach metrics to telemetry span if available
        span = ctx.get(f'node_span:{node_id}')
        if span:
            span.set_attribute('execution_time', duration)
            span.set_attribute('memory_mb', self.get_memory_usage())
        
        # Also send to metrics backend
        self.metrics_client.record({
            'node_id': node_id,
            'pipeline_path': ' > '.join(ctx.hierarchy_path),
            'depth': ctx.depth,
            'duration': duration
        })
```

---

# Callback Configuration Inheritance

Callbacks follow the hierarchical configuration inheritance system. See the **Hierarchical Configuration Precedence** section in [Core Concepts](Core%20Concepts%204a4dd7402980462eb83fc2b3d5059ccc.md) for complete details.

**Callback inheritance behavior:**

```python
# Parent defines callbacks
parent = Pipeline(
    nodes=[...],
    callbacks=[ProgressCallback(), LoggingCallback(), MetricsCallback()]
)

# Child inherits all callbacks
child = Pipeline(
    nodes=[...]
    # No callbacks specified → inherits all three callbacks from parent
)

# Grandchild overrides with custom callbacks
grandchild = Pipeline(
    nodes=[...],
    callbacks=[TelemetryCallback()]  # Override: only telemetry, no progress/logging/metrics
)
```

**Callback composition patterns:**

```python
# Define callbacks at appropriate level
outer = Pipeline(
    nodes=[...],
    callbacks=[ProgressCallback(), TelemetryCallback()]  # User-facing + distributed tracing
)

inner = Pipeline(
    nodes=[...],
    callbacks=[LoggingCallback()]  # Detailed internal logs only
)
# Inner gets LoggingCallback only (overrides parent)
```

**Selective callback disabling**:

```python
# Disable callbacks for specific sub-pipeline
quiet_pipeline = Pipeline(
    nodes=[...],
    callbacks=[]  # Override: no callbacks for this sub-pipeline
)
```

**Use cases:**

- **Progress tracking**: Define ProgressCallback at top level, automatically handles nested indentation
- **Telemetry**: Add TelemetryCallback at root, automatically creates hierarchical span trees
- **Performance testing**: Disable callbacks in nested pipelines to measure pure execution time
- **Debugging**: Add LoggingCallback to specific sub-pipeline to debug without verbose logs everywhere

---

# Usage

```python
from pipeline_system import Pipeline, ProgressCallback, TelemetryCallback

pipeline = Pipeline(
    nodes=[node1, node2, nested_pipeline],
    callbacks=[ProgressCallback(), TelemetryCallback()]
)

result = [pipeline.run](http://pipeline.run)(inputs)
# Automatically shows hierarchy:
# - Progress bars indented by nesting level
# - Telemetry spans with parent-child relationships
# - All nested pipelines tracked seamlessly
```

[Callback Implementations (Pseudocode)](Intelligent%20Callback%20System%20be7cb6cd6a5f419fb949210a31497a73/Callback%20Implementations%20(Pseudocode)%209dbb5e9ffa7b4875921b0b62422bcc32.md)