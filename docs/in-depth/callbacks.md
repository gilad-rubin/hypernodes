# Callbacks

Implement custom behavior by subclassing `PipelineCallback` and wiring into the engine via `engine=SequentialEngine(callbacks=[...])`.

Lifecycle hooks include:
- `on_pipeline_start/end`
- `on_node_start/end`
- `on_node_cached`
- Nested pipeline hooks
- Map operation hooks
