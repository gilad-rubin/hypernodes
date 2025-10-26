# Implementation Plan: Tracing, Telemetry & Progress Visualization

## Overview

Implement comprehensive observability for HyperNodes with:
- **Live progress bars** for `.run()` and `.map()` operations
- **Distributed tracing** with Logfire (OpenTelemetry-based)
- **Waterfall charts** for post-hoc analysis
- **Opt-in telemetry** (default: local only, can export to Logfire cloud)

## Architecture

### Design Principles

1. **Composable Callbacks**: Separate callbacks for progress and telemetry - users compose them
2. **Hierarchical**: Automatic nesting for nested pipelines
3. **Non-invasive**: No changes to pipeline execution logic
4. **Environment-aware**: Auto-detects Jupyter vs CLI
5. **Performance**: Minimal overhead when telemetry disabled
6. **Respect Logfire Config**: TelemetryCallback uses logfire as-is, no configuration management

### Key Components

```
src/hypernodes/
├── telemetry/
│   ├── __init__.py
│   ├── progress.py          # ProgressCallback (tqdm/rich)
│   ├── tracing.py           # TelemetryCallback (Logfire)
│   └── waterfall.py         # Waterfall chart generation (Jupyter only)
└── callbacks.py             # Base classes (existing)
```

## Phase 1: Core Infrastructure

### 1.1 Dependencies

**Add to `pyproject.toml`:**

```toml
[project.optional-dependencies]
telemetry = [
    "logfire>=2.0.0",
    "tqdm>=4.67.1",
    "rich>=13.0.0",
    "plotly>=5.18.0",  # For waterfall charts
]
```

### 1.2 Environment Detection

Create `src/hypernodes/telemetry/environment.py`:

```python
def is_jupyter() -> bool:
    """Detect if running in Jupyter notebook."""
    try:
        get_ipython().__class__.__name__
        return True
    except NameError:
        return False
```

## Phase 2: Progress Visualization

### 2.1 ProgressCallback

**File:** `src/hypernodes/telemetry/progress.py`

**Features:**
- Hierarchical progress bars with indentation based on `ctx.depth`
- Auto-select `tqdm.notebook` vs `tqdm.rich`
- Track: node execution, map operations, cache hits
- Show: progress %, time elapsed, items/sec (for maps)

**Key Implementation:**

```python
class ProgressCallback(PipelineCallback):
    def __init__(self, enable=True):
        self.bars = {}  # node_id -> tqdm instance
        self.enable = enable
        self.use_notebook = is_jupyter()
    
    def on_pipeline_start(self, pipeline_id, inputs, ctx):
        indent = "  " * ctx.depth
        total_nodes = ctx.get_pipeline_metadata(pipeline_id).get('total_nodes', 0)
        bar = self._create_bar(
            desc=f"{indent}Pipeline {pipeline_id}",
            total=total_nodes
        )
        ctx.set(f'progress_bar:{pipeline_id}', bar)
    
    def on_node_start(self, node_id, inputs, ctx):
        indent = "  " * (ctx.depth + 1)
        bar = self._create_bar(desc=f"{indent}├─ {node_id}")
        ctx.set(f'progress_bar:{node_id}', bar)
    
    def on_node_end(self, node_id, outputs, duration, ctx):
        bar = ctx.get(f'progress_bar:{node_id}')
        if bar:
            bar.set_description(f"{bar.desc} ✓ ({duration:.2f}s)")
            bar.close()
        # Update parent pipeline bar
        pipeline_bar = ctx.get(f'progress_bar:{ctx.current_pipeline_id}')
        if pipeline_bar:
            pipeline_bar.update(1)
    
    def on_map_start(self, total_items, ctx):
        bar = self._create_bar(
            desc=f"{'  ' * (ctx.depth + 1)}Maps",
            total=total_items
        )
        ctx.set('map_progress_bar', bar)
```

## Phase 3: Telemetry & Tracing

### 3.1 TelemetryCallback with Logfire

**File:** `src/hypernodes/telemetry/tracing.py`

**Features:**
- OpenTelemetry spans via Logfire
- Hierarchical span tree (parent-child)
- Context propagation (for future remote backends)
- **Uses existing logfire configuration** (no configuration management)
- Capture: inputs, outputs, duration, cache hits, errors

**Configuration:**

```python
class TelemetryCallback(PipelineCallback):
    """Uses logfire for tracing. User must call logfire.configure() first.
    
    Example:
        import logfire
        logfire.configure()  # or logfire.configure(send_to_logfire=False) for local only
        
        pipeline = Pipeline(nodes=[...], callbacks=[TelemetryCallback()])
    """
    
    def __init__(
        self,
        trace_map_items: bool = True,
        logfire_instance: Optional['logfire.Logfire'] = None
    ):
        """Initialize telemetry callback.
        
        Args:
            trace_map_items: Whether to create spans for individual map items
            logfire_instance: Optional logfire instance (uses default if None)
        """
        import logfire
        self.logfire = logfire_instance or logfire
        self.trace_map_items = trace_map_items
        self.span_data = []  # For waterfall charts
    
    def on_pipeline_start(self, pipeline_id, inputs, ctx):
        # Create span (logfire automatically handles parent context)
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
    
    def on_node_start(self, node_id, inputs, ctx):
        # Logfire automatically handles parent from context
        span = self.logfire.span(
            f'node:{node_id}',
            node_id=node_id,
            depth=ctx.depth
        ).__enter__()
        
        ctx.set(f'span:{node_id}', span)
        ctx.set(f'span_start:{node_id}', time.time())
    
    def on_node_end(self, node_id, outputs, duration, ctx):
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
                'cached': False
            })
    
    def on_node_cached(self, node_id, signature, ctx):
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
                'cached': True
            })
```

## Phase 4: Waterfall Visualization

### 4.1 Span Data Collection

Add to `TelemetryCallback`:

```python
class TelemetryCallback:
    def __init__(self, ...):
        ...
        self.span_data = []  # Collect for waterfall
    
    def on_node_end(self, node_id, outputs, duration, ctx):
        ...
        # Collect data for waterfall
        self.span_data.append({
            'name': node_id,
            'start_time': time.time() - duration,
            'duration': duration,
            'depth': ctx.depth,
            'parent': ctx.current_pipeline_id,
            'cached': False
        })
    
    def get_waterfall_data(self):
        """Get data for waterfall chart generation."""
        return self.span_data
```

### 4.2 Waterfall Chart Generator (Jupyter Only)

**File:** `src/hypernodes/telemetry/waterfall.py`

```python
import plotly.graph_objects as go
from typing import List, Dict

def create_waterfall_chart(span_data: List[Dict]) -> go.Figure:
    """Create Gantt-style waterfall chart from span data.
    
    Only works in Jupyter notebooks. Displays interactive Plotly chart.
    """
    if not span_data:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No span data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Normalize start times
    min_time = min(s['start_time'] for s in span_data)
    
    fig = go.Figure()
    
    for span in span_data:
        y_pos = span['depth']
        start = (span['start_time'] - min_time) * 1000  # ms
        duration = span['duration'] * 1000  # ms
        
        color = 'lightgreen' if span.get('cached') else 'lightblue'
        
        fig.add_trace(go.Bar(
            name=span['name'],
            y=[y_pos],
            x=[duration],
            base=start,
            orientation='h',
            marker=dict(color=color),
            hovertemplate=f"{span['name']}<br>Duration: {duration:.2f}ms<extra></extra>"
        ))
    
    fig.update_layout(
        title="Pipeline Execution Waterfall",
        xaxis_title="Time (ms)",
        yaxis_title="Depth",
        barmode='overlay',
        showlegend=False,
        height=max(300, len(span_data) * 30)  # Dynamic height
    )
    
    return fig


class TelemetryCallback(PipelineCallback):
    # ... (see above)
    
    def get_waterfall_chart(self) -> go.Figure:
        """Generate waterfall chart from collected span data.
        
        Returns:
            Plotly figure (auto-displays in Jupyter)
        """
        return create_waterfall_chart(self.span_data)
```

## Phase 5: Usage Examples

### 5.1 User Composition Pattern

Users compose callbacks themselves - no CombinedCallback needed:

```python
import logfire
from hypernodes import Pipeline
from hypernodes.telemetry import ProgressCallback, TelemetryCallback

# Configure logfire (user controls all settings)
logfire.configure()  # Uses .env, defaults to send_to_logfire=True if token present
# OR: logfire.configure(send_to_logfire=False)  # Local only
# OR: logfire.configure(send_to_logfire='if-token-present')  # Smart default

# Create callbacks
progress = ProgressCallback()
telemetry = TelemetryCallback()

# Compose them
pipeline = Pipeline(
    nodes=[...],
    callbacks=[progress, telemetry]  # User decides which to include
)

result = pipeline.run(inputs={...})

# Generate waterfall (in Jupyter)
chart = telemetry.get_waterfall_chart()
chart  # Auto-displays in Jupyter

### 5.2 Common Usage Patterns

**Pattern 1: Progress Only**
```python
from hypernodes.telemetry import ProgressCallback

pipeline = Pipeline(
    nodes=[...],
    callbacks=[ProgressCallback()]
)
```

**Pattern 2: Telemetry Only (Local)**
```python
import logfire
from hypernodes.telemetry import TelemetryCallback

logfire.configure(send_to_logfire=False)  # Local only
pipeline = Pipeline(
    nodes=[...],
    callbacks=[TelemetryCallback()]
)
```

**Pattern 3: Both Progress + Telemetry (Cloud)**
```python
import logfire
from hypernodes.telemetry import ProgressCallback, TelemetryCallback

logfire.configure()  # Uses LOGFIRE_TOKEN from .env if present

pipeline = Pipeline(
    nodes=[...],
    callbacks=[ProgressCallback(), TelemetryCallback()]
)
result = pipeline.run(inputs={...})

# In Jupyter: Generate waterfall
telemetry = pipeline.callbacks[1]  # Or keep reference
chart = telemetry.get_waterfall_chart()
chart  # Auto-displays
```

## Phase 6: Test Suite

### 6.1 Test Files

**Create:**
- `tests/test_telemetry_progress.py` - Progress bar tests
- `tests/test_telemetry_tracing.py` - Logfire tracing tests
- `tests/test_telemetry_waterfall.py` - Waterfall chart tests
- `tests/test_telemetry_integration.py` - Full integration tests

### 6.2 Key Test Cases

1. **Progress bars**:
   - Single pipeline with multiple nodes
   - Nested pipelines (hierarchical display)
   - Map operations (items/sec, cache hits)
   - Jupyter vs CLI environment

2. **Tracing**:
   - Span hierarchy matches execution order
   - Parent-child relationships correct
   - Cache hits recorded
   - Errors captured
   - Export to Logfire cloud (mock)

3. **Waterfall**:
   - Chart generation from span data
   - Parallel nodes visualized correctly
   - Nested pipelines shown hierarchically

4. **Integration**:
   - CombinedCallback works correctly
   - No conflicts between progress and tracing
   - Performance overhead < 5%

## Phase 7: Documentation

### 7.1 User Guide

**Create:** `docs/guides/TELEMETRY_GUIDE.md`

**Contents:**
- Quick start
- Configuration options
- Live progress bars
- Logfire integration
- Waterfall charts
- Performance tips

### 7.2 Example Notebook

**Create:** `notebooks/telemetry_showcase.ipynb`

**Demonstrate:**
- Basic progress bars
- Nested pipelines with progress
- Map operations with live stats
- Telemetry export to Logfire
- Waterfall chart generation

## Key Decisions Made

1. **No logfire configuration**: TelemetryCallback uses logfire as-is. User calls `logfire.configure()` with their settings.
2. **No CombinedCallback**: Users compose callbacks themselves: `callbacks=[ProgressCallback(), TelemetryCallback()]`
3. **Waterfall is Jupyter-only**: Returns Plotly figure that auto-displays in notebooks
4. **No .env handling**: Let logfire handle all environment variables and configuration

**Ready to proceed with implementation!**

## Implementation Order

1. Phase 1: Dependencies + environment detection
2. Phase 2: ProgressCallback  
3. Phase 3: TelemetryCallback (respects logfire config)
4. Phase 4: Waterfall visualization (Jupyter only)
5. Phase 5: Usage examples
6. Phase 6: Test suite
7. Phase 7: Documentation

## Success Criteria

- [ ] Live progress bars for single and map operations
- [ ] Hierarchical display for nested pipelines
- [ ] Logfire integration (opt-in)
- [ ] Waterfall charts for post-hoc analysis
- [ ] < 5% performance overhead
- [ ] All tests passing
- [ ] Complete documentation
- [ ] Working Jupyter notebook example

## Next Steps

After approval, begin Phase 1 implementation.
