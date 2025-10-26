# Telemetry & Observability Guide

Comprehensive guide to using HyperNodes telemetry features for observability and performance analysis.

## Overview

HyperNodes provides two telemetry callbacks:

- **ProgressCallback**: Live progress bars (tqdm/rich-based)
- **TelemetryCallback**: Distributed tracing with Logfire (OpenTelemetry)

Both work seamlessly with nested pipelines and respect the hierarchical structure.

## Installation

```bash
# Install telemetry dependencies
pip install 'hypernodes[telemetry]'

# Or install individually
pip install logfire tqdm rich plotly
```

## Quick Start

### Progress Bars Only

```python
from hypernodes import node, Pipeline
from hypernodes.telemetry import ProgressCallback

@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

@node(output_name="result")
def add_one(doubled: int) -> int:
    return doubled + 1

pipeline = Pipeline(
    nodes=[double, add_one],
    callbacks=[ProgressCallback()]
)

result = pipeline.run(inputs={"x": 5})
# Shows live progress bars in terminal
```

### Telemetry (Local Only)

```python
import logfire
from hypernodes import node, Pipeline
from hypernodes.telemetry import TelemetryCallback

# Configure logfire (local only, no cloud export)
logfire.configure(send_to_logfire=False)

pipeline = Pipeline(
    nodes=[double, add_one],
    callbacks=[TelemetryCallback()]
)

result = pipeline.run(inputs={"x": 5})
# Creates OpenTelemetry spans locally
```

### Both Progress + Telemetry

```python
import logfire
from hypernodes import Pipeline
from hypernodes.telemetry import ProgressCallback, TelemetryCallback

# Configure logfire
logfire.configure(send_to_logfire=False)

# Compose callbacks
pipeline = Pipeline(
    nodes=[double, add_one],
    callbacks=[ProgressCallback(), TelemetryCallback()]
)

result = pipeline.run(inputs={"x": 5})
# Shows progress bars AND creates telemetry spans
```

## ProgressCallback

### Features

- **Auto-detection**: Uses `tqdm.notebook` in Jupyter, `tqdm.rich` in CLI
- **Hierarchical display**: Indented by nesting depth
- **Per-node tracking**: Shows time elapsed for each node
- **Map operations**: Displays items/sec, cache hits, progress
- **Cache indicators**: ⚡ symbol for cached nodes

### Example Output

```
Pipeline: main_pipeline ━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:42
  ├─ clean_text ✓ (0.05s)
  ├─ extract_features ✓ (0.20s)
  └─ train_model ✓ (0.17s)
```

### Options

```python
# Disable for testing
progress = ProgressCallback(enable=False)

# Use in pipeline
pipeline = Pipeline(nodes=[...], callbacks=[progress])
```

## TelemetryCallback

### Features

- **OpenTelemetry spans**: Uses Logfire for distributed tracing
- **Hierarchical spans**: Automatic parent-child relationships
- **Context propagation**: Ready for future remote backends
- **Span data collection**: For waterfall charts
- **Rich attributes**: Captures inputs, outputs, duration, cache hits, errors

### Configuration

TelemetryCallback **uses your existing logfire configuration**. Configure logfire before creating the callback:

```python
import logfire

# Option 1: Local only (no cloud export)
logfire.configure(send_to_logfire=False)

# Option 2: Send to Logfire cloud (requires LOGFIRE_TOKEN in .env)
logfire.configure()  # Auto-detects token

# Option 3: Conditional (send if token present)
logfire.configure(send_to_logfire='if-token-present')
```

### Environment Variables

Create a `.env` file:

```bash
# Optional: Logfire token for cloud export
LOGFIRE_TOKEN=your_token_here

# Optional: Service name
LOGFIRE_SERVICE_NAME=my-pipeline-service

# Optional: Environment
LOGFIRE_ENVIRONMENT=production
```

Logfire automatically loads from `.env` when you call `logfire.configure()`.

### Options

```python
telemetry = TelemetryCallback(
    trace_map_items=True,  # Create spans for individual map items
    logfire_instance=None  # Use custom logfire instance (optional)
)
```

### Viewing Traces

**Local Console:**
```python
logfire.configure(send_to_logfire=False, console=True)
```

**Logfire Cloud:**
1. Sign up at https://logfire.pydantic.dev
2. Get your token
3. Add to `.env`: `LOGFIRE_TOKEN=your_token`
4. Run: `logfire.configure()`
5. View traces at https://logfire.pydantic.dev

## Waterfall Charts (Jupyter Only)

Visualize execution timeline with interactive Plotly charts.

### Basic Usage

```python
from hypernodes.telemetry import TelemetryCallback
import logfire

logfire.configure(send_to_logfire=False)

telemetry = TelemetryCallback()
pipeline = Pipeline(nodes=[...], callbacks=[telemetry])

result = pipeline.run(inputs={...})

# In Jupyter notebook:
chart = telemetry.get_waterfall_chart()
chart  # Auto-displays interactive chart
```

### Features

- **Gantt-style visualization**: Horizontal bars showing execution timeline
- **Parallel execution**: See which nodes ran concurrently
- **Nested pipelines**: Hierarchical display
- **Cache hits**: Green bars for cached nodes
- **Interactive**: Hover for details (duration, depth, type)

### Chart Types

- **Blue bars**: Regular nodes
- **Orange bars**: Pipelines
- **Light green bars**: Map operations
- **Green bars**: Cached operations (⚡)

## Common Patterns

### Pattern 1: Development (Progress Only)

```python
from hypernodes.telemetry import ProgressCallback

pipeline = Pipeline(
    nodes=[...],
    callbacks=[ProgressCallback()]
)
```

**Use when:**
- Developing locally
- Want quick feedback on progress
- Don't need detailed tracing

### Pattern 2: Production (Telemetry Only)

```python
import logfire
from hypernodes.telemetry import TelemetryCallback

logfire.configure()  # Uses LOGFIRE_TOKEN from .env

pipeline = Pipeline(
    nodes=[...],
    callbacks=[TelemetryCallback()]
)
```

**Use when:**
- Running in production
- Need distributed tracing
- Want performance monitoring
- Don't need console output

### Pattern 3: Analysis (Both + Waterfall)

```python
import logfire
from hypernodes.telemetry import ProgressCallback, TelemetryCallback

logfire.configure(send_to_logfire=False)

telemetry = TelemetryCallback()
pipeline = Pipeline(
    nodes=[...],
    callbacks=[ProgressCallback(), telemetry]
)

result = pipeline.run(inputs={...})

# Generate waterfall for analysis
chart = telemetry.get_waterfall_chart()
chart
```

**Use when:**
- Performance optimization
- Debugging issues
- Understanding execution flow
- Jupyter notebook analysis

### Pattern 4: Testing (Disabled)

```python
from hypernodes.telemetry import ProgressCallback

progress = ProgressCallback(enable=False)

pipeline = Pipeline(
    nodes=[...],
    callbacks=[progress]
)
```

**Use when:**
- Running tests
- Don't want console output
- Still want callback structure

## Nested Pipelines

Telemetry automatically handles nested pipelines:

```python
import logfire
from hypernodes import Pipeline
from hypernodes.telemetry import ProgressCallback, TelemetryCallback

logfire.configure(send_to_logfire=False)

# Inner pipeline
inner = Pipeline(nodes=[preprocess, encode])

# Outer pipeline (contains inner)
outer = Pipeline(
    nodes=[load_data, inner, aggregate],
    callbacks=[ProgressCallback(), TelemetryCallback()]
)

result = outer.run(inputs={...})
```

**Output:**
```
Pipeline: outer ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:48
  ├─ load_data ✓ (0.02s)
  ├─ Pipeline: inner ━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:43
  │  ├─ preprocess ✓ (0.08s)
  │  └─ encode ✓ (0.35s)
  └─ aggregate ✓ (0.03s)
```

**Telemetry:**
- Creates hierarchical span tree
- Parent-child relationships automatic
- Context propagates correctly
- Waterfall shows nesting

## Map Operations

Special handling for `.map()` operations:

```python
from hypernodes.telemetry import ProgressCallback, TelemetryCallback
import logfire

logfire.configure(send_to_logfire=False)

telemetry = TelemetryCallback(trace_map_items=True)  # Trace each item
pipeline = Pipeline(
    nodes=[...],
    callbacks=[ProgressCallback(), telemetry]
)

results = pipeline.map(items=data)
```

**Progress output:**
```
Map [5.2 items/s, 27.6% cached] ━━━━━━━━ 847/1000 0:02:34
```

**Options:**
- `trace_map_items=True`: Create span for each item (detailed but verbose)
- `trace_map_items=False`: Only parent map span (default, recommended)

## Performance Tips

1. **Disable in production if not needed**: Progress bars have overhead
   ```python
   progress = ProgressCallback(enable=False)
   ```

2. **Don't trace every map item**: Use `trace_map_items=False` for large maps
   ```python
   telemetry = TelemetryCallback(trace_map_items=False)
   ```

3. **Use sampling for high-volume**: Configure logfire sampling
   ```python
   logfire.configure(
       sampling=logfire.SamplingOptions(head=0.1)  # 10% sampling
   )
   ```

4. **Local console for dev, cloud for prod**: Separate configurations
   ```python
   # Dev
   logfire.configure(send_to_logfire=False, console=True)
   
   # Prod
   logfire.configure(send_to_logfire='if-token-present', console=False)
   ```

## Troubleshooting

### "logfire is not installed"

```bash
pip install 'hypernodes[telemetry]'
```

### "plotly is not installed"

```bash
pip install 'hypernodes[telemetry]'
# or
pip install plotly
```

### Progress bars not showing

- Check if running in Jupyter vs CLI
- Verify `enable=True` (default)
- Try without rich: `from tqdm import tqdm` directly

### Telemetry not working

- Check logfire configuration: `logfire.configure()`
- Verify no errors in console
- Check if `send_to_logfire=False` for local testing

### Waterfall chart empty

- Ensure you're in Jupyter notebook
- Check `telemetry.span_data` has entries
- Verify pipeline actually executed

## Next Steps

- **Read the specs**: See full specifications in `docs/`
- **Try examples**: Check `examples/` directory
- **Run tests**: `pytest tests/test_telemetry_*`
- **Explore Logfire**: https://logfire.pydantic.dev/docs

## API Reference

### ProgressCallback

```python
ProgressCallback(enable: bool = True)
```

### TelemetryCallback

```python
TelemetryCallback(
    trace_map_items: bool = True,
    logfire_instance: Optional[logfire.Logfire] = None
)
```

**Methods:**
- `get_waterfall_chart()` → Plotly Figure

### Waterfall Charts

```python
create_waterfall_chart(span_data: List[Dict]) → go.Figure
```
