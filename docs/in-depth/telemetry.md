# Telemetry & Tracing

- Progress bars: `ProgressCallback`
- Distributed tracing: `TelemetryCallback` (Logfire / OpenTelemetry)

```python
from hypernodes import Pipeline, SequentialEngine
from hypernodes.telemetry import ProgressCallback, TelemetryCallback

engine = SequentialEngine(callbacks=[ProgressCallback(), TelemetryCallback()])
pipeline = Pipeline(nodes=[...], engine=engine)
result = pipeline.run(inputs={...})
```

In notebooks, generate waterfall charts:

```python
telemetry = engine.callbacks[-1]  # Get telemetry callback from engine
fig = telemetry.get_waterfall_chart()
fig  # displays in Jupyter
```
