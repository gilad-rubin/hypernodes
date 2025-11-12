SOLID Refactoring Plan: HyperNodes Engine Architecture
Executive Summary
Goal: Refactor the execution engine to follow SOLID principles while eliminating ~77% code duplication. Current State:
backend.py (~2000 lines) + executors/local.py (~1565 lines) = 3565 lines
Massive code duplication between files and within files
God classes with too many responsibilities
Violation of Open/Closed Principle (if-statement dispatching)
Target State:
4 new files totaling ~750 lines
Single Responsibility: Each class has one job
Open/Closed: Add new executors without modifying existing code
Dependency Inversion: Uniform executor interface
Leverage NetworkX for graph operations
Architecture Overview
Core Principle: Uniform Executor Interface
Everything uses the same interface as concurrent.futures:
future = executor.submit(fn, *args, **kwargs)
result = future.result()
This works for:
SequentialExecutor (custom adapter)
AsyncExecutor (custom adapter)
ThreadPoolExecutor (stdlib)
ProcessPoolExecutor (stdlib)
Key Components
┌─────────────────────────────────────────────────────────────┐
│ User Code                                                    │
│ pipeline = Pipeline(nodes=[...])                            │
│ engine = HypernodesEngine(                                  │
│     node_executor="threaded",                               │
│     map_executor=ThreadPoolExecutor(max_workers=4)          │
│ )                                                            │
│ pipeline.with_engine(engine)                                │
│ results = pipeline.run(inputs={...})                        │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ HypernodesEngine (engine.py)                                │
│ - Resolves executor strings to instances                    │
│ - Creates orchestrators for node & map execution            │
│ - Handles map depth tracking & executor downgrading         │
│ - Delegates to PipelineOrchestrator                         │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ PipelineOrchestrator (orchestrator.py)                      │
│ - Contains ALL common execution logic                       │
│ - Setup: context, callbacks, NetworkX operations            │
│ - Loop: Find ready nodes using NetworkX graph               │
│ - Execute: executor.submit(execute_single_node, ...)        │
│ - Accumulate: Merge results into available_values           │
│ - Cleanup: Filter outputs, trigger callbacks                │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ execute_single_node (node_execution.py)                     │
│ - Handles both Node and PipelineNode                        │
│ - Computes signatures using cache helpers                   │
│ - Checks cache, executes if miss, stores result             │
│ - Triggers callbacks at appropriate lifecycle points        │
└─────────────────────────────────────────────────────────────┘
File Structure & Responsibilities
1. engine.py (~300 lines)
Responsibilities:
Define Engine ABC with run() and map() interface
Implement HypernodesEngine with:
Executor resolution (string → instance)
Depth-based executor downgrading for nested maps
Orchestrator composition
Map item preparation and result transposition
Key Classes:
Engine(ABC) - Base interface
HypernodesEngine(Engine) - Main implementation
Important Methods:
_resolve_executor(executor: Union[str, Executor]) -> Executor
_get_executor_for_depth(base_executor, depth: int) -> Executor
_prepare_items(inputs, map_over, map_mode) -> List[Dict]
_transpose_results(results: List[Dict]) -> Dict[str, List]
2. orchestrator.py (~150 lines)
Responsibilities:
Common pipeline execution logic (setup → loop → cleanup)
Use NetworkX for graph operations
Submit nodes to executor using uniform interface
Accumulate results and manage available values
Key Class:
PipelineOrchestrator - Single responsibility: orchestrate execution
NetworkX Usage:
# Find ready nodes efficiently
ready_nodes = [
    node for node in pending_nodes
    if all(
        pred in completed_nodes or isinstance(pred, str)
        for pred in pipeline.graph.predecessors(node)
    )
]

# Alternative: Use in-degree
in_degrees = dict(pipeline.graph.in_degree(pending_nodes))
ready_nodes = [node for node, degree in in_degrees.items() if degree == 0]
3. executor_adapters.py (~100 lines)
Responsibilities:
Provide adapters for non-concurrent.futures executors
Make sequential and async execution look like .submit() interface
Key Classes:
SequentialExecutor - Immediate synchronous execution
AsyncExecutor - Asyncio-based concurrent execution
Default Worker Counts:
DEFAULT_WORKERS = {
    "sequential": 1,
    "async": 100,  # High concurrency for I/O
    "threaded": os.cpu_count() or 4,
    "parallel": os.cpu_count() or 4,
}
4. node_execution.py (~200 lines)
Responsibilities:
Execute single nodes (Node or PipelineNode)
Compute signatures for caching
Manage cache get/put operations
Trigger node-level callbacks
Key Functions:
execute_single_node(node, available_values, pipeline, callbacks, ctx, node_signatures) -> (result, signature)
compute_node_signature(node, inputs, node_signatures) -> str
compute_pipeline_node_signature(pipeline_node, inputs, node_signatures) -> str
_execute_pipeline_node(pipeline_node, inputs, pipeline, callbacks, ctx) -> Dict
_get_node_id(node) -> str (utility)
Important Design Considerations
1. NetworkX Graph Operations
Current Usage (keep):
pipeline.graph - NetworkX DiGraph of dependencies
pipeline.execution_order - Topological sort
pipeline._compute_required_nodes(output_name) - Uses nx.ancestors()
Enhanced Usage (new):
# In PipelineOrchestrator.execute():

# Option A: Use predecessors to find ready nodes
ready_nodes = []
for node in pending_nodes:
    predecessors = pipeline.graph.predecessors(node)
    # Check if all non-input predecessors are available
    if all(
        pred in completed_nodes or isinstance(pred, str)  # str = input parameter
        for pred in predecessors
    ):
        ready_nodes.append(node)

# Option B: Use in-degree (more efficient for large graphs)
subgraph = pipeline.graph.subgraph(pending_nodes)
in_degrees = dict(subgraph.in_degree())
ready_nodes = [node for node, degree in in_degrees.items() if degree == 0]
Benefits:
Leverages NetworkX's optimized graph algorithms
Cleaner than manual parameter checking
More maintainable and testable
2. Nested Map Depth Tracking
Problem: Nested maps can explode concurrency
Outer map: 100 items with 16 workers
Inner map: 50 items with 16 workers
Total: 100 × 16 × 50 × 16 = 1,280,000 concurrent tasks! 🔥
Solution: Track depth in context and downgrade executors
# In HypernodesEngine.map():
map_depth = ctx.get("_map_depth", 0)

if map_depth == 0:
    # Top-level: full parallelism (e.g., 16 workers)
    effective_executor = self.map_executor
elif map_depth == 1:
    # First nesting: sqrt(workers) (e.g., 4 workers)
    effective_executor = self._downgrade_executor(self.map_executor)
else:
    # Deeper nesting: sequential (1 worker)
    effective_executor = SequentialExecutor()

ctx.set("_map_depth", map_depth + 1)
Context Flow:
Outer pipeline.map() sets _map_depth = 1
Context passed to execute_pipeline_item → node_orchestrator.execute()
When PipelineNode with map_over executes, it calls pipeline.map(_ctx=ctx)
Inner pipeline.map() reads _map_depth = 1, increments to 2
Executor is downgraded appropriately
3. PipelineNode with map_over
How it works:
# User code
inner = Pipeline(nodes=[process_doc])
outer = Pipeline(nodes=[
    load_docs,
    inner.as_node(map_over="documents"),  # ← Creates PipelineNode
    save_results
])

# Execution flow:
# 1. load_docs produces {"documents": [doc1, doc2, ...]}
# 2. PipelineNode.__call__ is invoked with documents=[...]
# 3. Detects self.map_over is set
# 4. Calls self.pipeline.map(inputs={...}, map_over="documents", _ctx=exec_ctx)
# 5. Inner pipeline.map uses map_executor (with depth adjustment!)
# 6. Each item processed using node_executor
Key insight: PipelineNode.__call__ (in pipeline.py) already handles this correctly. The engine just needs to:
Execute PipelineNode like any other node
Pass _exec_ctx so nested maps can track depth
Let PipelineNode handle the internal map delegation
4. Executor Lifecycle Management
Rules:
If user passes an executor instance → don't close it (they own it)
If engine creates executor from string → close it when done
Temporary executors (depth downgrading) → always close
class HypernodesEngine:
    def _resolve_executor(self, executor):
        if isinstance(executor, str):
            # We create it, we own it
            if executor == "threaded":
                return ThreadPoolExecutor(max_workers=DEFAULT_WORKERS["threaded"])
            # ... etc
        else:
            # User provided it, they own it
            return executor
    
    def shutdown(self):
        """Cleanup owned resources."""
        # Only shutdown if we created it
        if isinstance(self._base_node_executor_spec, str):
            self.node_executor.shutdown(wait=True)
        if isinstance(self._base_map_executor_spec, str):
            self.map_executor.shutdown(wait=True)
5. Signature Computation for PipelineNodes
Challenge: PipelineNode wraps a pipeline - how to compute its signature? Solution:
def compute_pipeline_node_signature(pipeline_node, inputs, node_signatures):
    """Compute signature for a PipelineNode.
    
    Signature = hash(inner_pipeline_structure + inputs + dependencies)
    """
    inner_pipeline = pipeline_node.pipeline
    
    # Hash the inner pipeline structure (all node functions)
    inner_code_hashes = []
    for inner_node in inner_pipeline.execution_order:
        if hasattr(inner_node, "pipeline"):
            # Nested PipelineNode - use its pipeline ID
            inner_code_hashes.append(inner_node.pipeline.id)
        elif hasattr(inner_node, "func"):
            # Regular node - hash its function
            inner_code_hashes.append(hash_code(inner_node.func))
    
    code_hash = hashlib.sha256("::".join(inner_code_hashes).encode()).hexdigest()
    inputs_hash = hash_inputs(inputs)
    
    # Dependencies
    deps_signatures = [
        node_signatures[param]
        for param in pipeline_node.root_args
        if param in node_signatures
    ]
    deps_hash = ":".join(sorted(deps_signatures))
    
    return compute_signature(code_hash, inputs_hash, deps_hash)
Implementation Phases & Tests
Phase 1: Foundation - Executor Adapters
Files to create:
src/hypernodes/executor_adapters.py
Implementation:
class SequentialExecutor:
    def submit(self, fn, *args, **kwargs): ...
    def shutdown(self, wait=True): pass

class AsyncExecutor:
    def submit(self, fn, *args, **kwargs): ...
    def shutdown(self, wait=True): pass

DEFAULT_WORKERS = {...}
Tests (tests/test_executor_adapters.py):
def test_sequential_executor_submit():
    """Test SequentialExecutor executes immediately."""
    executor = SequentialExecutor()
    future = executor.submit(lambda x: x * 2, 5)
    assert future.result() == 10

def test_sequential_executor_exception():
    """Test SequentialExecutor handles exceptions."""
    executor = SequentialExecutor()
    future = executor.submit(lambda: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        future.result()

def test_async_executor_submit():
    """Test AsyncExecutor runs async functions."""
    executor = AsyncExecutor()
    async def async_fn(x):
        await asyncio.sleep(0.01)
        return x * 2
    future = executor.submit(async_fn, 5)
    assert future.result() == 10

def test_executor_interface_compatibility():
    """Test all executors have compatible interface."""
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    
    executors = [
        SequentialExecutor(),
        AsyncExecutor(),
        ThreadPoolExecutor(max_workers=2),
        ProcessPoolExecutor(max_workers=2),
    ]
    
    for executor in executors:
        assert hasattr(executor, 'submit')
        assert hasattr(executor, 'shutdown')
        future = executor.submit(lambda x: x * 2, 5)
        assert future.result() == 10
    
    # Cleanup
    for ex in executors[2:]:
        ex.shutdown(wait=True)
Phase 2: Node Execution Logic
Files to create:
src/hypernodes/node_execution.py
Implementation:
def execute_single_node(node, available_values, pipeline, callbacks, ctx, node_signatures):
    """Execute one node with caching and callbacks."""
    # 1. Get inputs
    # 2. Compute signature
    # 3. Check cache
    # 4. If miss: execute + callbacks
    # 5. Store in cache
    # 6. Return (result, signature)

def compute_node_signature(node, inputs, node_signatures): ...
def compute_pipeline_node_signature(pipeline_node, inputs, node_signatures): ...
def _execute_pipeline_node(pipeline_node, inputs, pipeline, callbacks, ctx): ...
def _get_node_id(node): ...
Tests (tests/test_node_execution.py):
def test_execute_single_node_basic():
    """Test executing a simple node."""
    @node(output_name="result")
    def add_one(x: int) -> int:
        return x + 1
    
    pipeline = Pipeline(nodes=[add_one])
    ctx = CallbackContext()
    
    result, signature = execute_single_node(
        add_one,
        {"x": 5},
        pipeline,
        [],
        ctx,
        {}
    )
    
    assert result == 6
    assert isinstance(signature, str)
    assert len(signature) == 64  # SHA256 hex

def test_execute_single_node_with_cache():
    """Test node execution with caching."""
    call_count = 0
    
    @node(output_name="result", cache=True)
    def expensive(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2
    
    cache = DiskCache(".test_cache")
    pipeline = Pipeline(nodes=[expensive], cache=cache)
    ctx = CallbackContext()
    
    # First execution
    result1, sig1 = execute_single_node(expensive, {"x": 5}, pipeline, [], ctx, {})
    assert result1 == 10
    assert call_count == 1
    
    # Second execution (should hit cache)
    result2, sig2 = execute_single_node(expensive, {"x": 5}, pipeline, [], ctx, {})
    assert result2 == 10
    assert call_count == 1  # Not called again!
    assert sig1 == sig2
    
    # Cleanup
    cache.clear()

def test_execute_pipeline_node():
    """Test executing a PipelineNode (nested pipeline)."""
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    inner = Pipeline(nodes=[double])
    pipeline_node = inner.as_node()
    
    outer = Pipeline(nodes=[pipeline_node])
    ctx = CallbackContext()
    
    result, signature = execute_single_node(
        pipeline_node,
        {"x": 5},
        outer,
        [],
        ctx,
        {}
    )
    
    assert result == {"doubled": 10}

def test_execute_pipeline_node_with_map_over():
    """Test PipelineNode with map_over parameter."""
    @node(output_name="processed")
    def process(item: str) -> str:
        return item.upper()
    
    inner = Pipeline(nodes=[process])
    pipeline_node = inner.as_node(map_over="items")
    
    outer = Pipeline(nodes=[pipeline_node])
    ctx = CallbackContext()
    
    result, signature = execute_single_node(
        pipeline_node,
        {"items": ["a", "b", "c"]},
        outer,
        [],
        ctx,
        {}
    )
    
    assert result == {"processed": ["A", "B", "C"]}

def test_node_signature_changes_with_inputs():
    """Test signatures change when inputs change."""
    @node(output_name="result")
    def identity(x: int) -> int:
        return x
    
    pipeline = Pipeline(nodes=[identity])
    ctx = CallbackContext()
    
    _, sig1 = execute_single_node(identity, {"x": 1}, pipeline, [], ctx, {})
    _, sig2 = execute_single_node(identity, {"x": 2}, pipeline, [], ctx, {})
    
    assert sig1 != sig2

def test_node_callbacks_triggered():
    """Test callbacks are triggered during execution."""
    events = []
    
    class TestCallback(PipelineCallback):
        def on_node_start(self, node_id, inputs, ctx):
            events.append(("start", node_id, inputs))
        
        def on_node_end(self, node_id, outputs, duration, ctx):
            events.append(("end", node_id, outputs))
        
        def on_node_cached(self, node_id, signature, ctx):
            events.append(("cached", node_id))
    
    @node(output_name="result")
    def add_one(x: int) -> int:
        return x + 1
    
    callback = TestCallback()
    pipeline = Pipeline(nodes=[add_one], callbacks=[callback])
    ctx = CallbackContext()
    
    execute_single_node(add_one, {"x": 5}, pipeline, [callback], ctx, {})
    
    assert len(events) == 2
    assert events[0] == ("start", "add_one", {"x": 5})
    assert events[1][0] == "end"
    assert events[1][1] == "add_one"
Phase 3: Orchestrator
Files to create:
src/hypernodes/orchestrator.py
Implementation:
class PipelineOrchestrator:
    def __init__(self, executor): ...
    
    def execute(self, pipeline, inputs, ctx, output_name):
        """Common execution logic using NetworkX and executor.submit()."""
        # Setup phase
        # Execution loop with NetworkX
        # Cleanup phase
Tests (tests/test_orchestrator.py):
def test_orchestrator_sequential():
    """Test orchestrator with sequential executor."""
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="result")
    def add_ten(doubled: int) -> int:
        return doubled + 10
    
    pipeline = Pipeline(nodes=[double, add_ten])
    executor = SequentialExecutor()
    orchestrator = PipelineOrchestrator(executor)
    
    result = orchestrator.execute(pipeline, {"x": 5}, None, None)
    
    assert result == {"doubled": 10, "result": 20}

def test_orchestrator_threaded():
    """Test orchestrator with ThreadPoolExecutor."""
    from concurrent.futures import ThreadPoolExecutor
    import time
    
    @node(output_name="a")
    def slow_a(x: int) -> int:
        time.sleep(0.1)
        return x * 2
    
    @node(output_name="b")
    def slow_b(x: int) -> int:
        time.sleep(0.1)
        return x * 3
    
    @node(output_name="result")
    def combine(a: int, b: int) -> int:
        return a + b
    
    pipeline = Pipeline(nodes=[slow_a, slow_b, combine])
    executor = ThreadPoolExecutor(max_workers=2)
    orchestrator = PipelineOrchestrator(executor)
    
    start = time.time()
    result = orchestrator.execute(pipeline, {"x": 5}, None, None)
    duration = time.time() - start
    
    assert result == {"a": 10, "b": 15, "result": 25}
    # Should execute a and b in parallel (~0.1s), then combine (~0.1s total)
    assert duration < 0.15  # Would be 0.2s if sequential
    
    executor.shutdown(wait=True)

def test_orchestrator_output_name_filtering():
    """Test orchestrator respects output_name parameter."""
    @node(output_name="a")
    def compute_a(x: int) -> int:
        return x * 2
    
    @node(output_name="b")
    def compute_b(a: int) -> int:
        return a + 10
    
    @node(output_name="c")
    def compute_c(b: int) -> int:
        return b * 3
    
    pipeline = Pipeline(nodes=[compute_a, compute_b, compute_c])
    executor = SequentialExecutor()
    orchestrator = PipelineOrchestrator(executor)
    
    # Request only "b" - should not compute "c"
    result = orchestrator.execute(pipeline, {"x": 5}, None, "b")
    
    assert result == {"b": 20}
    assert "c" not in result

def test_orchestrator_with_cache():
    """Test orchestrator leverages caching."""
    call_counts = {"a": 0, "b": 0}
    
    @node(output_name="a", cache=True)
    def compute_a(x: int) -> int:
        call_counts["a"] += 1
        return x * 2
    
    @node(output_name="b", cache=True)
    def compute_b(a: int) -> int:
        call_counts["b"] += 1
        return a + 10
    
    cache = DiskCache(".test_cache")
    pipeline = Pipeline(nodes=[compute_a, compute_b], cache=cache)
    executor = SequentialExecutor()
    orchestrator = PipelineOrchestrator(executor)
    
    # First run
    result1 = orchestrator.execute(pipeline, {"x": 5}, None, None)
    assert result1 == {"a": 10, "b": 20}
    assert call_counts == {"a": 1, "b": 1}
    
    # Second run (should use cache)
    result2 = orchestrator.execute(pipeline, {"x": 5}, None, None)
    assert result2 == {"a": 10, "b": 20}
    assert call_counts == {"a": 1, "b": 1}  # Not called again!
    
    # Cleanup
    cache.clear()

def test_orchestrator_callbacks():
    """Test orchestrator triggers callbacks."""
    events = []
    
    class TestCallback(PipelineCallback):
        def on_pipeline_start(self, pipeline_id, inputs, ctx):
            events.append("pipeline_start")
        
        def on_pipeline_end(self, pipeline_id, outputs, duration, ctx):
            events.append("pipeline_end")
        
        def on_node_start(self, node_id, inputs, ctx):
            events.append(f"node_start:{node_id}")
    
    @node(output_name="result")
    def identity(x: int) -> int:
        return x
    
    callback = TestCallback()
    pipeline = Pipeline(nodes=[identity], callbacks=[callback])
    executor = SequentialExecutor()
    orchestrator = PipelineOrchestrator(executor)
    
    orchestrator.execute(pipeline, {"x": 5}, None, None)
    
    assert "pipeline_start" in events
    assert "node_start:identity" in events
    assert "pipeline_end" in events
Phase 4: Engine Implementation
Files to create:
src/hypernodes/engine.py
Implementation:
class Engine(ABC):
    @abstractmethod
    def run(self, pipeline, inputs, output_name, _ctx): ...
    
    @abstractmethod
    def map(self, pipeline, inputs, map_over, map_mode, output_name, _ctx): ...

class HypernodesEngine(Engine):
    def __init__(self, node_executor, map_executor): ...
    def _resolve_executor(self, executor): ...
    def _get_executor_for_depth(self, base_executor, depth): ...
    def run(self, pipeline, inputs, output_name, _ctx): ...
    def map(self, pipeline, inputs, map_over, map_mode, output_name, _ctx): ...
Tests (tests/test_engine.py):
def test_engine_basic_run():
    """Test basic pipeline execution."""
    @node(output_name="result")
    def double(x: int) -> int:
        return x * 2
    
    pipeline = Pipeline(nodes=[double])
    engine = HypernodesEngine()
    pipeline.with_engine(engine)
    
    result = pipeline.run(inputs={"x": 5})
    assert result == {"result": 10}

def test_engine_string_executor():
    """Test engine resolves string executor specs."""
    @node(output_name="result")
    def identity(x: int) -> int:
        return x
    
    engine = HypernodesEngine(node_executor="sequential", map_executor="threaded")
    assert isinstance(engine.node_executor, SequentialExecutor)
    from concurrent.futures import ThreadPoolExecutor
    assert isinstance(engine.map_executor, ThreadPoolExecutor)

def test_engine_custom_executor():
    """Test engine accepts custom executor instances."""
    from concurrent.futures import ThreadPoolExecutor
    
    custom_executor = ThreadPoolExecutor(max_workers=8)
    engine = HypernodesEngine(node_executor=custom_executor)
    
    assert engine.node_executor is custom_executor
    
    custom_executor.shutdown(wait=True)

def test_engine_map_basic():
    """Test basic map operation."""
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    pipeline = Pipeline(nodes=[double])
    engine = HypernodesEngine()
    pipeline.with_engine(engine)
    
    result = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
    assert result == {"doubled": [2, 4, 6]}

def test_engine_map_depth_tracking():
    """Test map depth is tracked in context."""
    @node(output_name="processed")
    def process(item: str) -> str:
        return item.upper()
    
    inner = Pipeline(nodes=[process])
    
    # Create nested pipeline with map_over
    outer_node = inner.as_node(map_over="items")
    
    @node(output_name="items")
    def create_items(x: int) -> List[str]:
        return [f"item{i}" for i in range(x)]
    
    outer = Pipeline(nodes=[create_items, outer_node])
    engine = HypernodesEngine(map_executor="sequential")
    outer.with_engine(engine)
    
    # Execute
    result = outer.run(inputs={"x": 3})
    assert result == {"processed": ["ITEM0", "ITEM1", "ITEM2"]}

def test_engine_map_executor_downgrade():
    """Test executor is downgraded at deeper nesting levels."""
    from concurrent.futures import ThreadPoolExecutor
    
    engine = HypernodesEngine(map_executor=ThreadPoolExecutor(max_workers=16))
    
    # Depth 0: full executor
    exec_d0 = engine._get_executor_for_depth(engine.map_executor, 0)
    assert isinstance(exec_d0, ThreadPoolExecutor)
    assert exec_d0._max_workers == 16
    
    # Depth 1: reduced workers
    exec_d1 = engine._get_executor_for_depth(engine.map_executor, 1)
    assert isinstance(exec_d1, ThreadPoolExecutor)
    assert exec_d1._max_workers == 4  # sqrt(16)
    
    # Depth 2: sequential
    exec_d2 = engine._get_executor_for_depth(engine.map_executor, 2)
    assert isinstance(exec_d2, SequentialExecutor)
    
    # Cleanup
    engine.map_executor.shutdown(wait=True)
    exec_d1.shutdown(wait=True)

def test_engine_different_node_and_map_executors():
    """Test using different executors for nodes vs maps."""
    call_log = []
    
    @node(output_name="result")
    def track_execution(x: int) -> int:
        call_log.append(("node", x))
        return x * 2
    
    pipeline = Pipeline(nodes=[track_execution])
    
    # Sequential for nodes, threaded for maps
    engine = HypernodesEngine(
        node_executor="sequential",
        map_executor=ThreadPoolExecutor(max_workers=4)
    )
    pipeline.with_engine(engine)
    
    # Map should use threaded executor
    result = pipeline.map(inputs={"x": [1, 2, 3, 4]}, map_over="x")
    
    assert result == {"result": [2, 4, 6, 8]}
    assert len(call_log) == 4
    
    engine.map_executor.shutdown(wait=True)
Phase 5: Integration Tests
Tests (tests/test_engine_integration.py):
def test_nested_pipeline_with_map_over():
    """Test complex nested pipeline with map_over."""
    # Inner pipeline: process a single document
    @node(output_name="cleaned")
    def clean_text(doc: str) -> str:
        return doc.strip().lower()
    
    @node(output_name="word_count")
    def count_words(cleaned: str) -> int:
        return len(cleaned.split())
    
    inner = Pipeline(nodes=[clean_text, count_words])
    
    # Outer pipeline: process multiple documents
    @node(output_name="documents")
    def load_docs() -> List[str]:
        return ["  Hello World  ", "  Test Document  ", "  One Two Three  "]
    
    # Wrap inner pipeline with map_over
    process_docs = inner.as_node(
        input_mapping={"doc": "doc"},
        map_over="documents"
    )
    
    @node(output_name="total")
    def sum_counts(word_count: List[int]) -> int:
        return sum(word_count)
    
    outer = Pipeline(nodes=[load_docs, process_docs, sum_counts])
    engine = HypernodesEngine(
        node_executor="sequential",
        map_executor=ThreadPoolExecutor(max_workers=2)
    )
    outer.with_engine(engine)
    
    result = outer.run(inputs={})
    
    assert result["word_count"] == [2, 2, 3]  # word counts for each doc
    assert result["total"] == 7
    
    engine.map_executor.shutdown(wait=True)

def test_deeply_nested_maps():
    """Test triple-nested map operations."""
    # Level 3: process a single character
    @node(output_name="upper")
    def uppercase(char: str) -> str:
        return char.upper()
    
    level3 = Pipeline(nodes=[uppercase])
    
    # Level 2: process a single word
    level2_node = level3.as_node(map_over="word")
    
    @node(output_name="word")
    def split_word(word: str) -> List[str]:
        return list(word)
    
    level2 = Pipeline(nodes=[split_word, level2_node])
    
    # Level 1: process multiple sentences
    level1_node = level2.as_node(map_over="sentences")
    
    @node(output_name="sentences")
    def create_sentences() -> List[str]:
        return ["hello", "world"]
    
    level1 = Pipeline(nodes=[create_sentences, level1_node])
    
    # Engine with aggressive downgrading
    engine = HypernodesEngine(
        map_executor=ThreadPoolExecutor(max_workers=16)
    )
    level1.with_engine(engine)
    
    result = level1.run(inputs={})
    
    # Should produce uppercase characters for each word
    assert result["upper"] == [
        ["H", "E", "L", "L", "O"],
        ["W", "O", "R", "L", "D"]
    ]
    
    engine.map_executor.shutdown(wait=True)

def test_performance_comparison():
    """Compare performance across different executors."""
    import time
    
    @node(output_name="result")
    def cpu_intensive(x: int) -> int:
        # Simulate CPU work
        sum([i ** 2 for i in range(10000)])
        return x * 2
    
    pipeline = Pipeline(nodes=[cpu_intensive])
    items = list(range(20))
    
    # Sequential
    engine_seq = HypernodesEngine(map_executor="sequential")
    pipeline.with_engine(engine_seq)
    start = time.time()
    pipeline.map(inputs={"x": items}, map_over="x")
    seq_time = time.time() - start
    
    # Parallel
    from concurrent.futures import ProcessPoolExecutor
    engine_par = HypernodesEngine(map_executor=ProcessPoolExecutor(max_workers=4))
    pipeline.with_engine(engine_par)
    start = time.time()
    pipeline.map(inputs={"x": items}, map_over="x")
    par_time = time.time() - start
    
    # Parallel should be significantly faster
    assert par_time < seq_time * 0.5
    
    engine_par.map_executor.shutdown(wait=True)

def test_caching_across_maps():
    """Test caching works correctly in map operations."""
    call_count = 0
    
    @node(output_name="result", cache=True)
    def expensive(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2
    
    cache = DiskCache(".test_cache")
    pipeline = Pipeline(nodes=[expensive], cache=cache)
    engine = HypernodesEngine()
    pipeline.with_engine(engine)
    
    # First map
    result1 = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
    assert result1 == {"result": [2, 4, 6]}
    assert call_count == 3
    
    # Second map with same inputs (should use cache)
    result2 = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
    assert result2 == {"result": [2, 4, 6]}
    assert call_count == 3  # Not called again!
    
    # Third map with partial overlap
    result3 = pipeline.map(inputs={"x": [2, 3, 4]}, map_over="x")
    assert result3 == {"result": [4, 6, 8]}
    assert call_count == 4  # Only called for new item (4)
    
    # Cleanup
    cache.clear()
Migration Strategy
Step 1: Create New Files (No Breaking Changes)
Create engine.py, orchestrator.py, executor_adapters.py, node_execution.py
Keep existing backend.py and executors/local.py unchanged
Users can opt-in to new engine
Step 2: Update Pipeline
Add with_engine() method alongside with_backend()
Support both Backend and Engine interfaces during migration
Update effective_backend → effective_engine (with fallback)
Step 3: Deprecation Warnings
Add deprecation warnings to LocalBackend and HyperNodesEngine (old)
Guide users to new HypernodesEngine (new)
Step 4: Remove Old Code
After 2-3 releases, remove backend.py and executors/ folder
Keep only ModalBackend (move to integrations)
Success Metrics
✅ Code Reduction: 3565 → 750 lines (77% reduction)
✅ SOLID Compliance: All principles followed
✅ Test Coverage: >90% for all new code
✅ Performance: No regression, potential improvement with better NetworkX usage
✅ Backward Compatibility: Smooth migration path