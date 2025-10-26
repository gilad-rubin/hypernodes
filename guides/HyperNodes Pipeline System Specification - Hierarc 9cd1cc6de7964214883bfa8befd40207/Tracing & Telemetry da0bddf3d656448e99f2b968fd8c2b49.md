# Tracing & Telemetry

# Overview

Tracing and telemetry are **first-class citizens** in the pipeline system. Every execution is observable, traceable, and measurable by default. The system provides multiple layers of observability:

- **Structured Logging**: Capture detailed execution information with context
- **Distributed Tracing**: Track execution flow across nodes, pipelines, and remote workers
- **Real-Time Metrics**: Monitor performance, cache hits, parallelism, and resource usage
- **Visual Dashboards**: Interactive waterfall charts, timeline views, and progress tracking

---

# Architecture

Telemetry is implemented through the **Intelligent Callback System**, allowing observability tools to hook into pipeline lifecycle events without modifying pipeline code.

## Key Principles

**Non-Invasive Integration**

Telemetry is added via callbacks, not embedded in pipeline logic. Your pipeline code remains clean and testable.

**Hierarchical by Default**

Nested pipelines inherit and propagate telemetry context. When a pipeline contains other pipelines, traces maintain the full execution hierarchy automatically.

**Backend-Agnostic**

The same telemetry code works across local and remote execution. Context propagation ensures traces are stitched across process and network boundaries.

**Recursive Observability**

Every level of pipeline nesting gets its own telemetry spans, progress tracking, and visualization. A nested pipeline is treated as a node with its own observable world.

---

# Telemetry Layers

## 1. Execution Tracing

**OpenTelemetry spans** capture the execution timeline:

- Pipeline start/end spans
- Individual node execution spans
- Map operation spans (parent span + child spans per item)
- Nested pipeline spans (recursive hierarchy)

Spans include metadata:

- Input/output signatures
- Cache hit/miss status
- Execution duration
- Error information
- Backend and execution strategy

## 2. Structured Logging

Contextual logs are emitted at each lifecycle event:

- Pipeline initialization
- Node execution start/end
- Cache lookups and saves
- Error conditions
- Backend transitions

Logs are structured with consistent fields for aggregation and filtering.

## 3. Real-Time Metrics

Performance metrics tracked automatically:

- **Execution time**: Per node, per pipeline, per map item
- **Cache hit rate**: Overall and per-node statistics
- **Parallelism**: Active workers, queue depth
- **Resource usage**: Memory, CPU (when available)
- **Throughput**: Items/second for map operations

## 4. Visual Progress Tracking

See [Progress Visualization](Progress%20Visualization%20acf9de815df347f195c7eb98d79e72f8.md) for details on progress bars and status displays.

---

# Nested Pipeline Telemetry

When using **NestedPipeline** (a pipeline that contains other pipelines), telemetry maintains full hierarchical context:

**Recursive Span Hierarchy**

```
Parent Pipeline Span
├─ Node A Span
├─ Nested Pipeline Span
│  ├─ Child Node 1 Span
│  ├─ Child Node 2 Span
│  └─ Child Node 3 Span
└─ Node B Span
```

**Key Features:**

- Each nested pipeline creates a child span in the parent's trace
- Progress bars display hierarchically (see Visualization)
- Cache hits propagate up for aggregate statistics
- Errors in nested pipelines are captured with full context
- Separate backends for nested pipelines are tracked distinctly

**Example:**

```python
# Nested pipeline with its own backend
inner_pipeline = Pipeline(
    nodes=[preprocess, encode],
    backend=ModalBackend(gpu="A100")
)

outer_pipeline = Pipeline(
    nodes=[load_data, inner_pipeline, aggregate],
    backend=LocalBackend()
)

# Telemetry automatically:
# - Creates separate spans for outer and inner execution
# - Tracks inner pipeline's remote execution on Modal
# - Maintains context across local→remote→local boundaries
# - Displays hierarchical progress in real-time
```

---

# Context Propagation

## Local Execution

For local backends (threaded, multiprocess), telemetry context is propagated using OpenTelemetry's built-in context managers.

## Remote Execution

For remote backends (Modal, Coiled):

1. **Serialize context**: Active span context is serialized when dispatching remote jobs
2. **Transmit**: Context is sent along with node inputs
3. **Reattach**: Remote worker reattaches context before node execution
4. **Stream back**: Telemetry from remote execution is streamed to parent trace

This ensures **seamless distributed tracing** across hybrid local/remote pipelines.

---

# Telemetry Callbacks

## CallbackContext

All telemetry callbacks receive a `CallbackContext` object that maintains hierarchical execution state automatically. The context tracks:

- **Depth**: Nesting level of current execution
- **Hierarchy path**: Full chain of pipeline IDs from root to current
- **Shared state**: Key-value store for callbacks to coordinate (e.g., passing spans between callbacks)

This enables seamless hierarchical tracing without manual context management.

## Callback Interface

The telemetry callback implements the full `PipelineCallback` interface, including hooks for:

- Pipeline lifecycle (`on_pipeline_start`, `on_pipeline_end`)
- Node execution (`on_node_start`, `on_node_end`, `on_node_cached`)
- Nested pipelines (`on_nested_pipeline_start`, `on_nested_pipeline_end`)
- Map operations (`on_map_start`, `on_map_item_start`, `on_map_item_end`, `on_map_item_cached`, `on_map_end`)
- Error handling (`on_error`)

See [Intelligent Callback System](Intelligent%20Callback%20System%20be7cb6cd6a5f419fb949210a31497a73.md) for the complete callback interface and implementation examples.

---

# Default Telemetry Implementation: Logfire

[Logfire Integration](Tracing%20&%20Telemetry%20da0bddf3d656448e99f2b968fd8c2b49/Logfire%20Integration%2083377a692e0945698202258796352365.md)

---

# Configuration

**Enable telemetry:**

```python
from pipeline_system import Pipeline, LogfireCallback

pipeline = Pipeline(
    nodes=[...],
    callbacks=[LogfireCallback()]
)
```

**Configure export:**

```python
# Send to Logfire cloud
callback = LogfireCallback(export_to="cloud")

# Use local collector
callback = LogfireCallback(export_to="[localhost:4317](http://localhost:4317)")

# In-process only (no export)
callback = LogfireCallback(export_to=None)
```

**Selective instrumentation:**

```python
# Only trace expensive operations
callback = LogfireCallback(trace_map_items=False)

# Custom sampling
callback = LogfireCallback(sample_rate=0.1)
```

**Callback configuration inheritance:**

Telemetry callbacks follow the hierarchical configuration precedence system. When a nested pipeline doesn't define its own callbacks, it inherits them from its parent. This means:

- Define telemetry at the root level, and all nested pipelines automatically get traced
- Override callbacks in specific sub-pipelines for specialized instrumentation
- Disable callbacks selectively by setting `callbacks=[]` on a nested pipeline

See the **Callback Configuration Inheritance** section in [Intelligent Callback System](Intelligent%20Callback%20System%20be7cb6cd6a5f419fb949210a31497a73.md) and the **Hierarchical Configuration Precedence** section in [Nested Pipelines](Nested%20Pipelines%20e1f81b1aceb749ba86d9079449edf976.md) for details.

---

# Best Practices

1. **Always enable telemetry in production**: Even with sampling, traces are invaluable for debugging
2. **Use hierarchical views**: Nested pipelines benefit most from waterfall visualizations
3. **Export to persistent storage**: Traces are ephemeral; export for historical analysis
4. **Monitor cache hit rates**: Low cache hit rates indicate opportunity for optimization
5. **Profile with remote execution**: Telemetry reveals network overhead and serialization costs

---

# Integration with Other Components

- **Caching**: Cache hits/misses are tracked in telemetry spans
- **Visualization**: Progress bars and telemetry dashboards show complementary views
- **Backends**: Telemetry adapts to execution strategy (sequential, parallel, remote)
- **Error Handling**: Exceptions are captured with full trace context for debugging