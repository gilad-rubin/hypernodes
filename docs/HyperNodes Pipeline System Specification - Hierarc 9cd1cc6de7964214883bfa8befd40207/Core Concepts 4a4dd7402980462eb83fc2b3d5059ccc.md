# Core Concepts

# Functions

**Functions** are the atomic units of computation. Each function:

- Takes named inputs (function parameters)
- Produces a named output (declared in `@node` decorator)
- Declares dependencies implicitly through parameter names

```python
@node(output_name="cleaned_text")
def clean_text(passage: str) -> str:
    return passage.strip().lower()

@node(output_name="word_count")
def count_words(cleaned_text: str) -> int:
    return len(cleaned_text.split())

pipeline = Pipeline(nodes[clean_text, count_words])

result = [pipeline.run](http://pipeline.run)(inputs={"passage": "  Hello World  "})
assert result == {"cleaned_text": "hello world", "word_count": 2}
```

---

# Pipelines

**Pipelines** are directed acyclic graphs (DAGs) of functions. The pipeline automatically:

- Resolves dependencies based on parameter names matching output names
- Executes functions in the correct order
- Can visualize the DAG

```python
from pipeline_system import Pipeline

# Build pipeline from functions (default backend is LocalBackend with sequential execution)
encode_pipeline = Pipeline(nodes[clean_text, encode_text, pack_encoded])

# Visualize the DAG
encode_pipeline.visualize()
```

The pipeline automatically figures out execution order:

1. `clean_text` runs first (takes `passage` input)
2. `encode_text` runs next (needs `cleaned_text` from step 1)
3. `pack_encoded` runs last (needs `passage` and `embedding` from step 2)
    
    

## Execution Strategy

**Execution strategy is determined by the Backend.** The pipeline DAG defines *what* to run and in what order; the Backend determines *how* nodes are scheduled and executed.

**Example:**

```python
@node(output_name="a")
def step_a(input: Data) -> Result:
    return process_a(input)

@node(output_name="b")
def step_b(input: Data) -> Result:
    return process_b(input)

@node(output_name="c")
def step_c(a: Result, b: Result) -> Final:
    return combine(a, b)
```

In this pipeline, `step_a` and `step_b` can run independently (both only depend on `input`), while `step_c` depends on both results.

**The Backend decides when and how these execute:**

- **Eager execution** (default): `step_a` and `step_b` start immediately in parallel when inputs are available; `step_c` starts as soon as both complete
- **Lazy execution** (optional): Nodes only execute when pulled by downstream dependencies
- **Batch-aware execution** (optional): Optimize scheduling for batch workloads

See [Backends](Backends%207ba2913775254dec81a496ec0e3a27e5.md) for execution strategy configuration options.

---

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
    nodes[load_data, process_data, save_results],
    backend=LocalBackend(),
    cache=RedisCache(host="[localhost](http://localhost)"),
    callbacks=[logging_callback, metrics_callback]
)

# Child pipeline inherits everything except backend
child = Pipeline(
    nodes[transform, validate],
    backend=ModalBackend(gpu="A100")  # Override: use remote GPU
    # Inherits: cache from parent, callbacks from parent
)

# Grandchild inherits from child (which already overrode backend)
grandchild = Pipeline(
    nodes[encode, embed],
    cache=None  # Override: disable caching
    # Inherits: backend from child (Modal GPU), callbacks from parent
)
```

**Example: Full inheritance**

```python
# Outer pipeline defines all configuration
outer = Pipeline(
    nodes[preprocess, inner_pipeline, postprocess],
    backend=LocalBackend(),
    cache=DiskCache(path="/tmp/cache"),
    callbacks=[progress_callback]
)

# Inner pipeline has no configuration
inner_pipeline = Pipeline(
    nodes[step1, step2, step3]
    # No backend → inherits LocalBackend from outer
    # No cache → inherits DiskCache from outer
    # No callbacks → inherits progress_callback from outer
)
```

**Example: Mixed inheritance chain**

```python
level_1 = Pipeline(
    nodes[...],
    backend=LocalBackend(),
    cache=RedisCache(),
    timeout=300
)

level_2 = Pipeline(
    nodes[...],
    backend=ModalBackend()  # Override backend
    # Inherits: cache=RedisCache, timeout=300
)

level_3 = Pipeline(
    nodes[...],
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

```jsx
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
    nodes[clean_text, tokenize, normalize],
    backend=LocalBackend()
)

# Use inner pipeline as a node in outer pipeline
main_pipeline = Pipeline(
    nodes[load_data, preprocessing, train_model],
    backend=LocalBackend()
)

# Nested pipeline is treated as a function node
result = main_[pipeline.run](http://pipeline.run)(inputs={"data": raw_data})
```

## Separate Backends for Nested Pipelines

Nested pipelines can execute on different backends, enabling hybrid local/remote workflows:

```python
# Inner pipeline runs on GPU cluster
gpu_pipeline = Pipeline(
    nodes[encode_text, embed_image],
    backend=ModalBackend(gpu="A100", memory="256 GB")
)

# Outer pipeline runs locally
local_pipeline = Pipeline(
    nodes[load_files, gpu_pipeline, save_results],
    backend=LocalBackend()
)

# Execution:
# 1. load_files runs locally
# 2. gpu_pipeline dispatched to Modal with A100 GPU
# 3. save_results runs locally with results from remote execution
result = local_[pipeline.run](http://pipeline.run)(inputs={"files": file_list})
```

## Recursive Caching

Each level of nesting has independent cache behavior:

```python
# First run: Both levels execute fully
result = outer_[pipeline.run](http://pipeline.run)(inputs={"data": x})

# Second run with same data: Full cache hit at outer level
result = outer_[pipeline.run](http://pipeline.run)(inputs={"data": x})
```

## Hierarchical Visualization

Nested pipelines display with full hierarchy in progress bars and telemetry:

```jsx
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
    nodes[parse, validate, transform],
    backend=ModalBackend()
)

# Outer pipeline maps over collection
batch_pipeline = Pipeline(
    nodes[load_batch, item_pipeline, aggregate_results],
    backend=LocalBackend()
)

# The inner pipeline is mapped over each item in the batch
# Each item execution is independent, cached separately, and tracked hierarchically
results = batch_[pipeline.run](http://pipeline.run)(inputs={"batch_id": 123})
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

Define common sub-pipelines once, reuse them across multiple parent pipelines.

## Using Pipelines as Nodes with `.as_node()`

When using a pipeline as a node in another pipeline, you often need to adapt its interface—renaming parameters, mapping outputs, or defining internal mapping behavior. The `.as_node()` method provides a clean way to configure these adaptations.

### Basic Usage

**Without configuration** (names must match exactly):

```python
# Inner pipeline expects "passage", outer pipeline provides "passage"
inner_pipeline = Pipeline(nodes[clean_text, encode_text])
outer_pipeline = Pipeline(nodes[load_data, inner_pipeline, save_results])

result = outer_[pipeline.run](http://pipeline.run)(inputs={"passage": "Hello"})
```

**With configuration** (adapt the interface):

```python
# Inner pipeline expects "passage", outer pipeline calls it "document"
adapted_pipeline = inner_[pipeline.as](http://pipeline.as)_node(
    input_mapping={"document": "passage"}
)
outer_pipeline = Pipeline(nodes[load_data, adapted_pipeline, save_results])

result = outer_[pipeline.run](http://pipeline.run)(inputs={"document": "Hello"})  # Works!
```

### Parameters

```python
adapted = [pipeline.as](http://pipeline.as)_node(
    input_mapping: dict = None,
    output_mapping: dict = None,
    map_over: str | list[str] = None
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

inner = Pipeline(nodes[clean_text])

# Outer pipeline uses "document" instead of "passage"
adapted = [inner.as](http://inner.as)_node(input_mapping={"document": "passage"})
outer = Pipeline(nodes[adapted])

result = [outer.run](http://outer.run)(inputs={"document": "  HELLO  "})
assert result == {"cleaned": "hello"}
```

### Example 2: Renaming Outputs Only

```python
@node(output_name="result")
def process(data: str) -> str:
    return data.upper()

inner = Pipeline(nodes[process])

# Outer pipeline wants the output named "processed_data"
adapted = [inner.as](http://inner.as)_node(output_mapping={"result": "processed_data"})
outer = Pipeline(nodes[adapted])

result = [outer.run](http://outer.run)(inputs={"data": "hello"})
assert result == {"processed_data": "HELLO"}
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
    return EncodedPassage(pid=[passage.pid](http://passage.pid), embedding=embedding)

single_encode = Pipeline(nodes[clean_text, encode_text, pack_encoded])

# Outer pipeline: processes a CORPUS (list of passages)
encode_corpus = single_[encode.as](http://encode.as)_node(
    input_mapping={"corpus": "passage"},      # outer → inner
    output_mapping={"encoded_passage": "encoded_corpus"},  # inner → outer
    map_over="corpus"  # Map over corpus, treat each item as "passage"
)

@node(output_name="index")
def build_index(indexer: Indexer, encoded_corpus: List[EncodedPassage]) -> Index:
    return indexer.index(encoded_corpus)

encode_and_index = Pipeline(nodes[encode_corpus, build_index])

# Usage
result = encode_and_[index.run](http://index.run)(inputs={
    "corpus": [Passage(pid="1", text="Hello"), Passage(pid="2", text="World")],
    "encoder": my_encoder,
    "indexer": my_indexer
})
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

```jsx
corpus ──▶ [encode_corpus] ──▶ encoded_corpus ──▶ [build_index] ──▶ index
```

**Expanded (depth=2):**

```jsx
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

outer = Pipeline(nodes[pipeline_a, pipeline_b, combine])
# No collision: outer pipeline sees "result_a" and "result_b"
```

### Best Practices

1. **Use `.as_node()` when building reusable components** that will be used in multiple contexts with different naming conventions
2. **Use `map_over` to encapsulate mapping complexity** when you want the outer pipeline to stay clean
3. **Always document the expected interface** in docstrings when creating adapted pipelines
4. **Use descriptive names** for mapped parameters to make data flow obvious
5. **Avoid deep nesting** of mappings—if you need to chain multiple renames, consider refactoring

---

# Running Pipelines

## The Core Workflow: Test with One, Scale to Many

**A key design principle**: Build and test your pipeline with a single input, then scale to many inputs without changing code.

```python
# Step 1: Build your pipeline
encode_pipeline = Pipeline(nodes[clean_text, encode_text, pack_encoded])

# Visualize the DAG
encode_pipeline.visualize()

# Step 2: Test with a single input
result = encode_[pipeline.run](http://pipeline.run)(inputs={"passage": Passage(pid="1", text="Hello")})

# Step 3: Scale to many inputs
results = encode_[pipeline.map](http://pipeline.map)(
    inputs={"passage": [Passage(pid="1", text="Hello"), Passage(pid="2", text="World")]},
    map_over="passage"
)
```

**Benefits:**

1. **Simple, testable code**: Your pipeline stays unit-testable. Test with one input, debug easily.
2. **Intelligent caching**: Each item in `map()` is cached independently. If you already ran the pipeline on "Hello", it won't re-run when you process ["Hello", "World", "Foo"].
3. **No code changes**: The exact same pipeline definition works for single inputs and batch processing.

---

## [pipeline.run](http://pipeline.run)()

Execute a pipeline with a single set of inputs:

```python
result = encode_[pipeline.run](http://pipeline.run)(inputs={"passage": my_passage, "encoder": my_encoder})
```

**Key features:**

- Executes the entire DAG with provided inputs
- Automatically resolves dependencies between functions
- Returns a dictionary with all output names as keys
- Caches intermediate results for future runs
- Perfect for testing, debugging, and single-item processing

---

## [pipeline.map](http://pipeline.map)

Execute a pipeline over a collection of inputs:

```python
results = encode_[pipeline.map](http://pipeline.map)(
    inputs={"passage": [p1, p2, p3], "encoder": my_encoder},
    map_over="passage"
)
```

**Key features:**

- `inputs` is a dictionary where keys are parameter names (first positional argument)
- For parameters in `map_over`, values must be lists
- For parameters NOT in `map_over`, values are single constants used for all items
- `map_over` specifies which parameter(s) vary across items
- `map_mode` parameter controls how multiple varying inputs are combined (default: `"zip"`)
- Processes items in parallel (when backend supports it)
- Each item execution is independent and cached separately
- Returns a dict where each output is a list
- Progress tracking shows overall completion and cache hit rate
- Failed items can be retried without reprocessing successful ones
- Cache hits across runs: if you ran the pipeline on "Hello" before, it's instantly retrieved

### Map Modes

**Zip mode (default):** Process corresponding items together. Most common use case.

```python
# Process (id=1, text="a"), (id=2, text="b"), (id=3, text="c")
results = [pipeline.map](http://pipeline.map)(
    inputs={"id": [1, 2, 3], "text": ["a", "b", "c"]},
    map_over=["id", "text"],
    map_mode="zip"  # Default
)
```

**Product mode:** Create all combinations. Less common, useful for parameter sweeps.

```python
# Process all combinations of x and y
results = [pipeline.map](http://pipeline.map)(
    inputs={"x": [1, 2], "y": [10, 20]},
    map_over=["x", "y"],
    map_mode="product"  # Runs 4 times: (1,10), (1,20), (2,10), (2,20)
)
```

**Example: Adding new items to a batch**

```python
# First run: Process 100 passages
results_1 = encode_[pipeline.map](http://pipeline.map)(
    inputs={"passage": passages_1_to_100},
    map_over="passage"
)

# Later: Add 50 new passages
# Only the 50 new items are processed, 100 cached items retrieved instantly
results_2 = encode_[pipeline.map](http://pipeline.map)(
    inputs={"passage": passages_1_to_150},
    map_over="passage"
)
```

---

## pipeline.visualize()

Visualize the pipeline DAG:

```python
encode_pipeline.visualize()
```

This displays a graph showing:

- Functions as nodes
- Data flow between functions
- Input and output names
- Execution order