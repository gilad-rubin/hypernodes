# Backend → Executor Refactoring: Design Discussion

## Current Progress Summary

### ✅ Completed (Tasks 1-4):

**1. Created Executor Base Classes**
- `executors/base.py` - Abstract `Executor` class with `run()` and `map()` interface
- `executors/__init__.py` - Dynamic exports based on available dependencies

**2. LocalExecutor Complete (~1600 lines)**
- `executors/local.py` - Full LocalBackend → LocalExecutor migration
- Includes `PipelineExecutionEngine` for reusable execution patterns
- All execution modes: sequential/async/threaded/parallel
- All map modes with intelligent worker management

**3. DaftExecutor Complete**
- `executors/daft.py` - DaftBackend → DaftExecutor migration
- Updated imports and class names
- Framework executor that receives entire pipeline graph

**4. ModalExecutor Complete** *(Pending removal - see discussion below)*
- `executors/modal.py` - Created but may not be needed
- Remote function serialization approach

---

## Open Questions & Design Decisions

### 1. Modal Executor - Remove or Keep?

**User Feedback:**
> "We don't need a modal executor at all. I want to actually remove modal stuff from this codebase. If someone wants to run this on modal, they can just do it by running the script."

**Decision:**
- ❌ **Remove `ModalExecutor`** from the codebase
- ✅ Keep guidance in `guides/modal_functions.md` showing how to use Modal
- Users can wrap pipeline execution in Modal functions themselves
- Pattern: Define everything inside the Modal function to avoid serialization issues

**Action Items:**
- Delete `executors/modal.py`
- Remove Modal from `executors/__init__.py`
- Update documentation to show Modal usage pattern
- Maybe add example script showing Modal integration

---

### 2. Executor Interface - Simplifying Signatures

**Current Signatures:**
```python
def run(
    self,
    pipeline: Pipeline,
    inputs: Dict[str, Any],
    ctx: Optional[CallbackContext] = None,
    output_name: Union[str, List[str], None] = None,
) -> Dict[str, Any]:
    ...

def map(
    self,
    pipeline: Pipeline,
    items: List[Dict[str, Any]],
    inputs: Dict[str, Any],
    ctx: Optional[CallbackContext] = None,
    output_name: Union[str, List[str], None] = None,
) -> List[Dict[str, Any]]:
    ...
```

**User Feedback:**
> "Yeah, `run` and `map` should have exactly the same signature: `pipeline`, `inputs`, `output_name`, `context`. I think it's there because of the callbacks and nested pipelines, but if it's possible, I prefer to hide it from the user because I don't expect anyone to put the context there."

> "The only difference I think between `run` and `map` is this `map` should have `map_over` and from that it can construct the items."

**Proposed New Signatures:**
```python
# Option A: Keep context but make it more internal
def run(
    self,
    pipeline: Pipeline,
    inputs: Dict[str, Any],
    output_name: Union[str, List[str], None] = None,
    _ctx: Optional[CallbackContext] = None,  # Private, not in docs
) -> Dict[str, Any]:
    ...

def map(
    self,
    pipeline: Pipeline,
    inputs: Dict[str, Any],
    map_over: Union[str, List[str]],
    output_name: Union[str, List[str], None] = None,
    _ctx: Optional[CallbackContext] = None,  # Private, not in docs
) -> List[Dict[str, Any]]:
    ...
```

**Questions:**
- Should we move `ctx` to private parameter `_ctx`?
- Should `map()` construct items from `inputs` + `map_over` instead of receiving pre-constructed `items`?
- Does this break compatibility with current `Pipeline.map()` implementation?

---

### 3. Framework vs Node-Level Executors

**User Feedback:**
> "The framework executor receives the entire pipeline graph, is responsible for everything. By the way, I don't know, maybe it does make sense that the framework executor should handle caching and callbacks on its own. That's actually a really interesting point because if I'm using Daft, I'm assuming it should have some caching mechanism and callbacks that might interfere with our own callbacks."

**Current Architecture:**
- **Framework Executors** (like Daft):
  - Receive entire pipeline + graph
  - Convert to framework-native representation
  - Handle execution holistically
  - Currently still use HyperNodes cache/callbacks

- **Node-Level Executors** (like LocalExecutor):
  - Execute pipeline node-by-node
  - Fire callbacks per node
  - Handle caching per node
  - Support different parallelism strategies

**Key Question:** Should framework executors handle their own caching/callbacks?

**Options:**

**Option A: Framework Executors Own Cache/Callbacks**
```python
class DaftExecutor(Executor):
    def run(self, pipeline, inputs, output_name=None, _ctx=None):
        # Convert to Daft DataFrame
        # Use Daft's native caching (if any)
        # Don't fire HyperNodes callbacks
        # Return results
```

**Pros:**
- Clean separation - framework handles everything
- No callback interference
- Framework optimizations aren't blocked

**Cons:**
- Lose HyperNodes telemetry/progress tracking
- Inconsistent behavior between executors
- Users can't use same cache across executors

**Option B: Framework Executors Use HyperNodes Cache/Callbacks**
```python
class DaftExecutor(Executor):
    def run(self, pipeline, inputs, output_name=None, _ctx=None):
        # Convert to Daft DataFrame
        # Still use pipeline.effective_cache
        # Still fire callbacks
        # Return results
```

**Pros:**
- Consistent telemetry across all executors
- Unified caching layer
- Users get progress tracking

**Cons:**
- May interfere with framework optimizations
- Callback overhead on framework operations
- Framework caching might conflict

**Option C: Make It Configurable**
```python
class DaftExecutor(Executor):
    def __init__(
        self,
        use_hypernodes_cache: bool = False,
        use_hypernodes_callbacks: bool = False,
    ):
        ...
```

**Pros:**
- User controls the trade-off
- Flexibility for different use cases

**Cons:**
- More complexity
- Users need to understand the implications

**Discussion Needed:**
- Which option makes the most sense?
- What's the user expectation when using Daft?
- How important is unified telemetry?

---

### 4. Parallelism Control - Node-Level vs Map-Level

**User Feedback:**
> "I do want to distinguish between one pipeline running the nodes within a pipeline. Let's say I have two really, really heavy computations as the first two nodes that are independent. I might want to parallelize them or run them asynchronously."

> "Also I would like to offer that for `.map` operations, whether it's on the outer pipeline or the inner ones."

**Current LocalExecutor Configuration:**
```python
LocalExecutor(
    node_execution="sequential",  # How to execute nodes within pipeline.run()
    map_execution="sequential",   # How to execute items in pipeline.map()
    max_workers=4,
)
```

**Use Case Example:**
```python
# Outer pipeline with heavy independent nodes
outer = Pipeline(
    nodes=[heavy_node_1, heavy_node_2, process],
    executor=LocalExecutor(node_execution="parallel")  # Parallelize heavy nodes
)

# Inner pipeline with map operation
inner = Pipeline(
    nodes=[load, transform, save],
    executor=LocalExecutor(map_execution="threaded")  # Parallel map items
)

# Nested: outer contains inner as a node with map_over
outer_with_inner = Pipeline(
    nodes=[prepare, inner.as_node(map_over="items"), aggregate]
)
```

**Nested Parallelism Challenge:**
- Outer pipeline: `node_execution="parallel"` (2 workers)
- Inner pipeline: `map_execution="parallel"` (10 items, 10 workers)
- Total potential workers: 2 × 10 = 20 concurrent processes!

**Current Solution:**
```python
def _calculate_effective_workers(self, num_items: int, map_depth: int) -> int:
    if map_depth == 0:
        return min(self.max_workers, num_items)  # Full workers
    elif map_depth == 1:
        return min(int(self.max_workers**0.5) or 1, num_items)  # Sqrt
    else:
        return 1  # Sequential for deeper nesting
```

**Questions:**
- Is this worker reduction strategy good enough?
- Should we expose more control to users?
- Should we just document the behavior and let users manage it?

**User Statement:**
> "I don't really know what's the best practice here. I don't know how to optimize it. We're not trying to optimize anything. Just allow the user to define these things."

**Proposed:** Keep current behavior, document it clearly, let users configure if needed.

---

### 5. Cache/Callback Materialization with Modal

**User Concern:**
> "I'm just thinking about this use case where I want to run this all on modal and I'm trying to when I'm running things on modal I don't want to materialize the cache and the callbacks."

**Current Pattern (from guides/modal_functions.md):**
```python
@app.function()
def run_pipeline(inputs: dict, cache_path: str = "./cache"):
    # Instantiate cache/callbacks ON MODAL from config
    cache = DiskCache(path=cache_path)
    callbacks = [ProgressCallback()]

    pipeline = Pipeline(nodes=[...], cache=cache, callbacks=callbacks)
    return pipeline.run(inputs=inputs)

# Jupyter: call directly
result = run_pipeline.remote({"x": 5}, cache_path="./modal_cache")
```

**Problem with Nested Pipelines:**
```python
# Define locally
inner = Pipeline(
    nodes=[...],
    cache=DiskCache("./local_cache"),  # ❌ This cache is materialized locally
)

outer = Pipeline(
    nodes=[prepare, inner.as_node(map_over="items"), save],
)

# Send to Modal
@app.function()
def run_outer(inputs):
    # outer is serialized with inner's local cache already materialized!
    return outer.run(inputs)
```

**Possible Solutions:**

**Option 1: Guidance Only**
- Document the pattern clearly
- Tell users: "Define everything inside the Modal function"
- No code changes needed

**Option 2: Lazy Cache/Callback Instantiation**
```python
class Pipeline:
    def __init__(
        self,
        nodes,
        cache_factory: Optional[Callable[[], Cache]] = None,  # Factory pattern
        callbacks_factory: Optional[Callable[[], List[Callback]]] = None,
    ):
        self.cache_factory = cache_factory
        self._cache = None

    @property
    def cache(self):
        if self._cache is None and self.cache_factory:
            self._cache = self.cache_factory()
        return self._cache
```

**User Previous Thought:**
> "I thought of creating lazy materializing cache and callbacks, but actually that's too complicated."

**Decision:** Probably stick with **Option 1 - Guidance Only** since:
- Modal use case is now removed from core
- Factory pattern adds complexity
- Clear documentation can solve this
- Users can structure their code appropriately

---

### 6. Naming Clarification

**User Feedback:**
> "It's a little bit confusing. I don't think you got it completely, so local executor might not be a good name."

**Current Names:**
- `LocalExecutor` - Node-by-node execution with parallelism control
- `DaftExecutor` - Framework executor

**Possible Alternative Names:**

For "LocalExecutor":
- `NodeExecutor` - Emphasizes node-by-node execution
- `StandardExecutor` - Default/standard way to run
- `HyperNodesExecutor` - Emphasizes it uses HyperNodes native execution
- `SequentialExecutor` - But this conflicts with the fact it supports parallel modes
- `IncrementalExecutor` - Executes incrementally node-by-node

For "DaftExecutor":
- `DaftFrameworkExecutor` - More explicit
- Just `DaftExecutor` - Current, probably fine

**Discussion Needed:**
- What name best captures the distinction?
- Should we emphasize "node-by-node" vs "holistic"?
- Or emphasize "HyperNodes-native" vs "framework-native"?

---

## Summary of Decisions Needed

### High Priority:
1. **Remove Modal Executor?** → User says yes, remove it
2. **Simplify signatures?** → Hide `ctx` as `_ctx`, change `map(items, ...)` to `map(map_over, ...)`?
3. **Framework executors and cache/callbacks** → Should Daft handle its own or use HyperNodes?

### Medium Priority:
4. **Nested parallelism** → Keep current worker reduction? Just document?
5. **Naming** → Rename `LocalExecutor` to something clearer?

### Low Priority:
6. **Modal guidance** → Just documentation, no lazy materialization?

---

## Next Steps

Based on decisions above:
1. Remove Modal executor if confirmed
2. Update executor signatures if agreed
3. Clarify framework executor behavior
4. Update Pipeline class to use new naming/interface
5. Update all tests
6. Run full test suite

---

## Questions for User

1. **Modal Executor**: Confirmed we should delete it entirely?

2. **Executor Signatures**: Should we move forward with:
   ```python
   def run(pipeline, inputs, output_name=None, _ctx=None)
   def map(pipeline, inputs, map_over, output_name=None, _ctx=None)
   ```

3. **Framework Executors**: Should Daft use HyperNodes cache/callbacks or handle its own?

4. **Naming**: Any preference for renaming `LocalExecutor`? Or keep it?

5. **Nested Parallelism**: Current worker reduction strategy OK, or need more control?

6. **Anything else** I misunderstood or missed?
