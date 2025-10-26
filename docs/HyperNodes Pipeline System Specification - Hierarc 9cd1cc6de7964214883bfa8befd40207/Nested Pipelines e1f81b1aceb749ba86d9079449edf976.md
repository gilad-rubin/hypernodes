# Nested Pipelines

# Nested Pipelines

**Pipelines are nodes as well.** A pipeline can contain other pipelines, enabling hierarchical composition and modular architecture. Nested pipelines are treated as nodes with their own recursive world—they support all features independently:

- **Separate backends**: Each nested pipeline can define its own execution backend
- **Independent caching**: Cache hits tracked at each level
- **Hierarchical visualization**: Progress bars, telemetry, and traces display recursively
- **Isolated callbacks**: Nested pipelines can have their own callback configuration

## Hierarchical Configuration Precedence

Nested pipelines inherit configuration from their parent pipeline by default, but can override any aspect independently. This creates a flexible hierarchy where configuration cascades down until explicitly overridden.

**Configuration inheritance applies to:**

- **Backend**: Execution backend (local, remote, etc.)
- **Caching**: Cache configuration and storage
- **Callbacks**: Event handlers and hooks
- **Telemetry**: Logging and monitoring configuration
- **Timeout settings**: Execution time limits

**Precedence rules:**

1. **If a nested pipeline explicitly defines configuration**, it overrides the parent's configuration for that aspect
2. **If a nested pipeline does not define configuration**, it inherits from its parent
3. **Inheritance is recursive**: deeply nested pipelines inherit through the full chain until an override is encountered

**Example: Selective override**

```python
# Parent pipeline with full configuration
parent = Pipeline(
    nodes=[load_data, process_data, save_results],
    backend=LocalBackend(),
    cache=RedisCache(host="[localhost](http://localhost)"),
    callbacks=[logging_callback, metrics_callback]
)

# Child pipeline inherits everything except backend
child = Pipeline(
    nodes=[transform, validate],
    backend=ModalBackend(gpu="A100")  # Override: use remote GPU
    # Inherits: cache from parent, callbacks from parent
)

# Grandchild inherits from child (which already overrode backend)
grandchild = Pipeline(
    nodes=[encode, embed],
    cache=None  # Override: disable caching
    # Inherits: backend from child (Modal GPU), callbacks from parent
)
```

**Example: Full inheritance**

```python
# Outer pipeline defines all configuration
outer = Pipeline(
    nodes=[preprocess, inner_pipeline, postprocess],
    backend=LocalBackend(),
    cache=DiskCache(path="/tmp/cache"),
    callbacks=[progress_callback]
)

# Inner pipeline has no configuration
inner_pipeline = Pipeline(
    nodes=[step1, step2, step3]
    # No backend → inherits LocalBackend from outer
    # No cache → inherits DiskCache from outer
    # No callbacks → inherits progress_callback from outer
)
```

**Example: Mixed inheritance chain**

```python
level_1 = Pipeline(
    nodes=[...],
    backend=LocalBackend(),
    cache=RedisCache(),
    timeout=300
)

level_2 = Pipeline(
    nodes=[...],
    backend=ModalBackend()  # Override backend
    # Inherits: cache=RedisCache, timeout=300
)

level_3 = Pipeline(
    nodes=[...],
    cache=None  # Override cache (disable)
    # Inherits: backend=ModalBackend (from level_2), timeout=300 (from level_1)
)

# Final configuration for level_3:
# - backend: ModalBackend (overridden at level_2)
# - cache: None (overridden at level_3)
# - timeout: 300 (inherited from level_1)
```

**Benefits:**

- **Consistent defaults**: Define configuration once at the top level
- **Selective optimization**: Override only where needed (e.g., GPU for specific sub-pipeline)
- **Simplified refactoring**: Change parent configuration affects all children unless overridden
- **Clear configuration flow**: Easy to trace where settings come from

**Visualization shows inheritance:**

```
Outer Pipeline [Local, RedisCache] ━━━━━━━━━━━━━━━━━━━━ 100%
  ├─ preprocess [Local, RedisCache] ━━━━━━━━━━━━━━━━━━ 100%
  ├─ gpu_pipeline [Modal GPU, RedisCache] ━━━━━━━━━━━━ 100%
  │  ├─ encode [Modal GPU, RedisCache] ━━━━━━━━━━━━━━━ 100%
  │  └─ transform [Modal GPU, RedisCache] ━━━━━━━━━━━━ 100%
  └─ postprocess [Local, RedisCache] ━━━━━━━━━━━━━━━━━ 100%
```

See [Backends](Backends%207ba2913775254dec81a496ec0e3a27e5.md) and [**Simple, testable code**: Your pipeline stays unit-testable. Test with one input, debug easily.](https://www.notion.so/Simple-testable-code-Your-pipeline-stays-unit-testable-Test-with-one-input-debug-easily-f1b57e34ded040e4bd2f7bb7a6325a56?pvs=21) for configuration details.

## Creating Nested Pipelines

```python
# Define inner pipeline
preprocessing = Pipeline(
    nodes=[clean_text, tokenize, normalize],
    backend=LocalBackend()
)

# Use inner pipeline as a node in outer pipeline
main_pipeline = Pipeline(
    nodes=[load_data, preprocessing, train_model],
    backend=LocalBackend()
)

# Nested pipeline is treated as a function node
result = main_[pipeline.run](http://pipeline.run)(inputs={"data_path": "data.csv"})
```

## Separate Backends for Nested Pipelines

Nested pipelines can execute on different backends, enabling hybrid local/remote workflows:

```python
# Inner pipeline runs on GPU cluster
gpu_pipeline = Pipeline(
    nodes=[encode_text, embed_image],
    backend=ModalBackend(gpu="A100", memory="256 GB")
)

# Outer pipeline runs locally
local_pipeline = Pipeline(
    nodes=[load_files, gpu_pipeline, save_results],
    backend=LocalBackend()
)

# Execution:
# 1. load_files runs locally
# 2. gpu_pipeline dispatched to Modal with A100 GPU
# 3. save_results runs locally with results from remote execution
result = local_[pipeline.run](http://pipeline.run)(inputs={"file_path": "input.txt"})
```

## Recursive Caching

Each level of nesting has independent cache behavior:

```python
# First run: Both levels execute fully
result = outer_[pipeline.run](http://pipeline.run)(inputs={...})

# Second run with same inputs: Both levels cache hit
result = outer_[pipeline.run](http://pipeline.run)(inputs={...})  # Instant return

# Third run with changed inner input: Outer cache hit, inner re-runs
result = outer_[pipeline.run](http://pipeline.run)(inputs={"inner_param": "new_value"})
```

## Hierarchical Visualization

Nested pipelines display with full hierarchy in progress bars and telemetry:

```
Outer Pipeline [Local] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:48
  ├─ load_data ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:02
  ├─ inner_pipeline [Modal, GPU: A100] ━━━━━━━━━━━━━━━━━ 100% 0:00:43
  │  ├─ preprocess ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:08
  │  └─ encode ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:35
  └─ aggregate ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:03
```

See [Progress Visualization](Progress%20Visualization%20acf9de815df347f195c7eb98d79e72f8.md) and [Tracing & Telemetry](Tracing%20&%20Telemetry%20da0bddf3d656448e99f2b968fd8c2b49.md) for details on hierarchical observability.

## Nested Map Operations

Nested pipelines work seamlessly with `map()`:

```python
# Inner pipeline processes individual items
item_pipeline = Pipeline(
    nodes=[parse, validate, transform],
    backend=ModalBackend()
)

# Outer pipeline maps over collection
batch_pipeline = Pipeline(
    nodes=[load_batch, item_pipeline, aggregate_results],
    backend=LocalBackend()
)

# The inner pipeline is mapped over each item in the batch
# Each item execution is independent, cached separately, and tracked hierarchically
results = batch_[pipeline.map](http://pipeline.map)(inputs={"batch_id": [1, 2, 3, 4, 5]}, map_over=["batch_id"])
```

## Use Cases

**Hybrid Execution**

Run preprocessing locally, expensive computation remotely, postprocessing locally.

**Modular Development**

Build and test sub-pipelines independently, then compose them into larger workflows.

**Resource Optimization**

Use different backends for different computational requirements (CPU vs. GPU, local vs. cloud).

**Reusable Components**

Define common sub-pipelines once, reuse them across multiple parent pipelines.

## Using Pipelines as Nodes with `.as_node()`

When using a pipeline as a node in another pipeline, you often need to adapt its interface—renaming parameters, mapping outputs, or defining internal mapping behavior. The `.as_node()` method provides a clean way to configure these adaptations.

### Basic Usage

**Without configuration** (names must match exactly):

```python
# Inner pipeline expects "passage", outer pipeline provides "passage"
inner_pipeline = Pipeline(nodes=[clean_text, encode_text])
outer_pipeline = Pipeline(nodes=[load_data, inner_pipeline, save_results])

result = outer_[pipeline.run](http://pipeline.run)(inputs={"passage": "Hello World"})
```

**With configuration** (adapt the interface):

```python
# Inner pipeline expects "passage", outer pipeline calls it "document"
adapted_pipeline = inner_[pipeline.as](http://pipeline.as)_node(
    input_mapping={"document": "passage"},
    output_mapping={"encoded_passage": "encoded_document"}
)

outer_pipeline = Pipeline(nodes=[load_data, adapted_pipeline, save_results])
result = outer_[pipeline.run](http://pipeline.run)(inputs={"document": "Hello World"})
```

### Parameters

```python
adapted = [pipeline.as](http://pipeline.as)_node(
    input_mapping={"outer_param_name": "inner_param_name"},
    output_mapping={"inner_output_name": "outer_output_name"},
    map_over=["param_name"]  # Optional: specify which parameters to map over
)
```

**`input_mapping`**: Maps outer pipeline parameter names to inner pipeline parameter names.

- Format: `{outer_name: inner_name}`
- Direction: **outer → inner** (how inputs flow INTO the pipeline)
- Example: `{"corpus": "passage"}` means the outer pipeline's `corpus` parameter provides values for the inner pipeline's `passage` parameter

**`output_mapping`**: Maps inner pipeline output names to outer pipeline output names.

- Format: `{inner_name: outer_name}`
- Direction: **inner → outer** (how outputs flow OUT of the pipeline)
- Example: `{"encoded_passage": "encoded_corpus"}` means the inner pipeline's `encoded_passage` output becomes `encoded_corpus` in the outer pipeline

**`map_over`**: Specifies that the pipeline should map over a list parameter (from the outer pipeline's perspective).

- The inner pipeline still operates on single items
- The outer pipeline sees it as processing a list
- Caching happens per-item independently

### Direction Rationale

> ⚠️ **Direction matters!** `input_mapping` goes `{outer: inner}` while `output_mapping` goes `{inner: outer}`.
> 

> 
> 

> **Why different directions?**
> 

> - `input_mapping`: You're connecting wires FROM the outer context TO the inner pipeline. Think: "Where do the inner pipeline's inputs come from?" → `{outer: inner}`
> 

> - `output_mapping`: You're routing results FROM the inner pipeline TO the outer context. Think: "Where do the inner pipeline's outputs go?" → `{inner: outer}`
> 

> 
> 

> This asymmetry is intentional and matches the flow of data through the node boundary.
> 

### Example 1: Renaming Inputs Only

```python
@node(output_name="cleaned")
def clean_text(passage: str) -> str:
    return passage.strip().lower()

inner = Pipeline(nodes=[clean_text])

# Outer pipeline uses "document" instead of "passage"
adapted = [inner.as](http://inner.as)_node(input_mapping={"document": "passage"})

outer = Pipeline(nodes=[load_doc, adapted, save_result])
result = [outer.run](http://outer.run)(inputs={"document": "  Hello World  "})
```

### Example 2: Renaming Outputs Only

```python
@node(output_name="result")
def process(data: str) -> str:
    return data.upper()

inner = Pipeline(nodes=[process])

# Outer pipeline wants the output named "processed_data"
adapted = [inner.as](http://inner.as)_node(output_mapping={"result": "processed_data"})

outer = Pipeline(nodes=[load, adapted, save])
result = [outer.run](http://outer.run)(inputs={"data": "hello"})
```

### Example 3: Internal Mapping with Renaming

This is the most powerful pattern: a pipeline that internally maps over items but appears as a single-input, list-output node to the outer pipeline.

```python
# Inner pipeline: processes ONE passage
@node(output_name="cleaned_text")
def clean_text(passage: Passage) -> str:
    return passage.text.strip().lower()

@node(output_name="embedding")
def encode_text(encoder: Encoder, cleaned_text: str) -> Vector:
    return encoder.encode(cleaned_text)

@node(output_name="encoded_passage")
def pack_encoded(passage: Passage, embedding: Vector) -> EncodedPassage:
    return EncodedPassage(pid=
```

**What happens internally:**

1. `encode_corpus` receives `corpus` (a list) from outer pipeline
2. Maps it to `passage` for each item via `input_mapping`
3. Executes `single_encode` for each passage independently (with caching)
4. Collects `encoded_passage` outputs from each execution
5. Renames to `encoded_corpus` via `output_mapping`
6. `build_index` receives `encoded_corpus` as a list

**From the outer pipeline's perspective:**

- `encode_corpus` is just a node that takes `corpus: List[Passage]` and produces `encoded_corpus: List[EncodedPassage]`
- The mapping is completely encapsulated
- Each item in the corpus is cached independently

### Visualization

When visualizing pipelines with `.as_node()`:

```python
encode_and_index.visualize()  # Shows encode_corpus as single node
encode_and_index.visualize(depth=2)  # Expands to show internal pipeline
```

**Default (depth=1):**

```
corpus ──▶ [encode_corpus] ──▶ encoded_corpus ──▶ [build_index] ──▶ index
```

**Expanded (depth=2):**

```
corpus ──▶ [encode_corpus (maps over corpus)]
           ├─ passage ──▶ [clean_text] ──▶ cleaned_text
           ├─ cleaned_text ──▶ [encode_text] ──▶ embedding  
           └─ passage, embedding ──▶ [pack_encoded] ──▶ encoded_passage
         ──▶ encoded_corpus ──▶ [build_index] ──▶ index
```

### Namespace Management

When using `.as_node()` with `output_mapping`, the outer pipeline only sees the mapped names. This helps avoid naming collisions:

```python
# Two pipelines with same output name
pipeline_a = Pipeline(...).as_node(output_mapping={"result": "result_a"})
pipeline_b = Pipeline(...).as_node(output_mapping={"result": "result_b"})

outer = Pipeline(nodes=[pipeline_a, pipeline_b, combine])
# No collision: outer pipeline sees "result_a" and "result_b"
```

### Best Practices

1. **Use `.as_node()` when building reusable components** that will be used in multiple contexts with different naming conventions
2. **Use `map_over` to encapsulate mapping complexity** when you want the outer pipeline to stay clean
3. **Always document the expected interface** in docstrings when creating adapted pipelines
4. **Use descriptive names** for mapped parameters to make data flow obvious
5. **Avoid deep nesting** of mappings—if you need to chain multiple renames, consider refactoring