# Progress Visualization

# Progress Tracking

The pipeline system provides real-time visualization of execution progress, showing:

- Overall pipeline progress
- Individual node execution status
- Parallel execution visualization
- Cache hit statistics
- Performance metrics

---

# Single Pipeline Execution

**Running a pipeline with [`pipeline.run](http://pipeline.run)()`:**

```
Processing Pipeline ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:42
  ├─ clean_text ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:05
  ├─ extract_features ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:20
  └─ train_model ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:17
```

**Features:**

- Hierarchical display showing nested pipelines
- Time elapsed for each node
- Overall pipeline completion percentage
- Clear indication of which nodes are running

---

# Parallel Node Execution

**When multiple nodes can run in parallel:**

```
Processing Pipeline ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:42
  ├─ extract_feature_a ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:15
  ├─ extract_feature_b ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:18  
  ├─ extract_feature_c ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:12
  └─ combine ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:05
```

**Execution behavior:**

1. `extract_feature_a`, `extract_feature_b`, `extract_feature_c` run concurrently
2. Progress bars update independently
3. `combine` waits for all three to complete
4. Total time is determined by the slowest parallel branch

---

# Map Operations

**Running [`pipeline.map](http://pipeline.map)()` over a collection:**

```
Processing items [map] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 847/1000 0:02:34
  Workers: 10/10 active
  Rate: 5.2 items/sec
  Cache hits: 234 (27.6%)
  Failed: 3
```

**Metrics shown:**

- **Progress**: Current item / Total items
- **Workers**: Number of parallel workers active
- **Rate**: Items processed per second
- **Cache hits**: Percentage of items loaded from cache
- **Failed**: Number of items that encountered errors

---

# Cache Hit Visualization

**When re-running a pipeline with cached nodes:**

```
Processing Pipeline ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:08
  ├─ load_data ⚡ CACHED ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
  ├─ preprocess ⚡ CACHED ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
  └─ train_model ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:08
```

**Cache indicators:**

- ⚡ **CACHED**: Node output loaded from cache (instant)
- Elapsed time shows 0:00:00 for cached nodes
- Only changed nodes are re-executed
- Total time dramatically reduced

---

# Nested Pipeline Visualization

**Nested pipelines are treated as nodes with their own recursive world.** When a pipeline contains another pipeline (either directly or using `.as_node()`), all visualization, progress tracking, telemetry, and tracing features work hierarchically and recursively.

## Hierarchical Display

**When using `.as_node()` with `map_over`:**

```jsx
Processing Batch Pipeline ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:01:23
  └─ encode_corpus [maps over corpus: 100 items] ━━━━━━━━ 100% 0:01:23
      ├─ clean_text ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:12
      └─ encode ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:01:11
```

## Visualization Depth Control

Use the `depth` parameter to control how deeply nested pipelines are expanded:

```python
# Default: Show nested pipelines as single nodes
encode_and_index.visualize()  # depth=1
```

**Output (depth=1):**

```jsx
corpus ──▶ [encode_corpus] ──▶ encoded_corpus ──▶ [build_index] ──▶ index
```

**Expanded view (depth=2):**

```python
encode_and_index.visualize(depth=2)
```

**Output (depth=2):**

```jsx
corpus ──▶ [encode_corpus (maps over corpus)]
           ├─ passage ──▶ [clean_text] ──▶ cleaned_text
           ├─ cleaned_text ──▶ [encode_text] ──▶ embedding  
           └─ passage, embedding ──▶ [pack_encoded] ──▶ encoded_passage
         ──▶ encoded_corpus ──▶ [build_index] ──▶ index
```

**Fully expanded (depth=-1):**

```python
encode_and_index.visualize(depth=-1)  # Expand all levels
```

## Recursive Features

Each nested pipeline maintains its own:

- **Progress bars**: Child pipelines show their own progress hierarchy
- **Cache hit tracking**: Cache statistics propagate up to parent
- **Backend execution**: Nested pipelines can define their own backend
- **Telemetry spans**: Full trace hierarchy maintained across levels
- **Error context**: Failures in nested pipelines include full stack

## Example: Nested Pipeline with Separate Backend

```python
# Inner pipeline runs on GPU cluster
inner_pipeline = Pipeline(
    nodes=[preprocess, encode],
    backend=ModalBackend(gpu="A100"),
    callbacks=[LogfireCallback()]
)

# Outer pipeline runs locally
outer_pipeline = Pipeline(
    nodes=[load_data, inner_pipeline, aggregate],
    backend=LocalBackend(),
    callbacks=[LogfireCallback()]
)

# Visualization shows both local and remote execution
result = outer_[pipeline.run](http://pipeline.run)(data=dataset)
```

**Progress output:**

```jsx
Processing Pipeline [Local] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:48
  ├─ load_data ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:02
  ├─ inner_pipeline [Modal, GPU: A100] ━━━━━━━━━━━━━━━━ 100% 0:00:43
  │  ├─ preprocess ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:08
  │  └─ encode ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:35
  └─ aggregate ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:03
```

**Key visualization features:**

- **Backend indicators**: Shows where each level executes (Local, Modal, Coiled)
- **Resource tags**: Displays GPU type, memory, etc. for remote backends
- **Indentation**: Clear visual hierarchy of nested execution
- **Independent timing**: Each level tracks its own duration
- **Aggregate metrics**: Parent shows total time including all children
- **Aggregate metrics**: Parent shows total time including all children

## Configuration Inheritance Visualization

Progress bars display the effective configuration at each level, showing both overrides and inherited settings:

```jsx
Outer Pipeline [Local, RedisCache] ━━━━━━━━━━━━━━━━━━━━ 100%
  ├─ preprocess [Local, RedisCache] ━━━━━━━━━━━━━━━━━━ 100%
  ├─ gpu_pipeline [Modal GPU, RedisCache] ━━━━━━━━━━━━ 100%
  │  ├─ encode [Modal GPU, RedisCache] ━━━━━━━━━━━━━━━ 100%
  │  └─ transform [Modal GPU, RedisCache] ━━━━━━━━━━━━ 100%
  └─ postprocess [Local, RedisCache] ━━━━━━━━━━━━━━━━━ 100%
```

**Configuration annotations:**

- `[Local]` - LocalBackend (inherited or explicit)
- `[Modal GPU]` - ModalBackend override
- `[RedisCache]` - Cache backend (inherited)
- `[No Cache]` - Cache explicitly disabled

This makes it easy to see the effective configuration at each level and spot overrides at a glance.

See the **Hierarchical Configuration Precedence** section in [Core Concepts](Core%20Concepts%204a4dd7402980462eb83fc2b3d5059ccc.md) for details on how configuration inheritance works

## Recursive Map Operations

When a nested pipeline uses `map()`, visualization shows:

```jsx
Processing Batch [Local] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:02:15
  └─ process_items [map over 50 items] ━━━━━━━━━━━━━━━━ 100% 0:02:15
      └─ inner_pipeline [Modal] [map over 50 items] ━━━━ 100% 0:02:12
          ├─ extract ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:01:10
          └─ transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:01:02
      
      Workers: 10/10 active
      Rate: 4.2 items/sec
      Cache hits: 12 (24%)
```

## Telemetry Integration

See [Tracing & Telemetry](Tracing%20&%20Telemetry%20da0bddf3d656448e99f2b968fd8c2b49.md) for details on how telemetry spans maintain hierarchical context across nested pipelines, including:

- Span parent-child relationships
- Context propagation across remote boundaries
- Waterfall visualizations of nested execution
- Cache hit tracking at each level

**Nested display:**

- Shows parent pipeline and nested pipeline hierarchy
- Indicates map operations with item count
- Indentation shows nesting level
- Aggregate timing for all iterations
- Backend and resource information for each level

---

# Jupyter Notebook Integration

**Auto-detection of environment:**

The visualization system automatically detects the execution environment:

- **Terminal/CLI**: Uses [`tqdm.rich`](http://tqdm.rich) for rich text formatting
- **Jupyter Notebook**: Uses `tqdm.notebook` with HTML widgets
- **Plain output**: Falls back to simple text progress for non-interactive environments

**Jupyter-specific features:**

- Interactive progress bars with real-time updates
- Collapsible nested pipeline displays
- Color-coded status indicators
- Hover tooltips with detailed metrics