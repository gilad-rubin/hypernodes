# Telemetry & Tracing

- Progress bars: `ProgressCallback`
- Distributed tracing: `TelemetryCallback` (Logfire / OpenTelemetry)

```python
from hypernodes import Pipeline
from hypernodes.telemetry import ProgressCallback, TelemetryCallback

pipeline = Pipeline(nodes=[...], callbacks=[ProgressCallback(), TelemetryCallback()])
result = pipeline.run(inputs={...})
```

In notebooks, generate waterfall charts:

```python
telemetry = pipeline.callbacks[-1]
fig = telemetry.get_waterfall_chart()
fig  # displays in Jupyter
```
