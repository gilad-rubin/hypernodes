# Runners API Specification

## Runner Compatibility Matrix

| Feature | `Runner` | `AsyncRunner` | `DaftRunner` |
|---------|----------|---------------|--------------|
| DAG execution | ✅ | ✅ | ✅ |
| Cycles | ✅ | ✅ | ❌ |
| `@branch` gates | ✅ | ✅ | ❌ |
| `@route` gates | ✅ | ✅ | ❌ |
| `InterruptNode` | ❌ | ✅ | ❌ |
| `.iter()` streaming | ❌ | ✅ | ❌ |
| `.map()` batch | ✅ | ✅ | ✅ |
| Async nodes | ❌ | ✅ | ✅ |
| Distributed execution | ❌ | ❌ | ✅ |

---

## Runner (Synchronous)

### Constructor

```python
class Runner:
    def __init__(
        self,
        *,
        cache: Cache | None = None,
        callbacks: list[Callback] | None = None,
    ) -> None:
        """
        Create synchronous runner.
        
        Args:
            cache: Cache backend (e.g., DiskCache, MemoryCache).
            callbacks: Observability callbacks.
        """
```

### run()

```python
def run(
    self,
    graph: Graph,
    inputs: dict[str, Any],
    *,
    select: list[str] | None = None,
    session_id: str | None = None,
    max_iterations: int = 1000,
) -> dict[str, Any]:
    """
    Execute graph synchronously.
    
    Args:
        graph: Graph to execute.
        inputs: Input values. For cycles, determines starting point.
        select: Output names to return. Default: all leaf outputs.
        session_id: Group related runs (for logging/tracing).
        max_iterations: Maximum iterations before InfiniteLoopError.
    
    Returns:
        Dict mapping output names to values.
    
    Raises:
        GraphConfigError: Graph structure invalid.
        ConflictError: Parallel producers conflict.
        MissingInputError: Required input not provided.
        InfiniteLoopError: Exceeded max_iterations.
        IncompatibleRunnerError: Graph has async nodes.
    
    Example:
        runner = Runner(cache=DiskCache("./cache"))
        result = runner.run(graph, inputs={"query": "hello"})
        print(result["response"])
    """
```

### map()

```python
def map(
    self,
    graph: Graph,
    inputs: dict[str, Any],
    *,
    map_over: str | list[str],
    select: list[str] | None = None,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Execute graph for each item in mapped parameter(s).
    
    Args:
        graph: Graph to execute.
        inputs: Input values. map_over params should be lists.
        map_over: Parameter name(s) to iterate over.
        select: Outputs to return per item.
        session_id: Group all runs under one session.
    
    Returns:
        List of output dicts, one per input item.
    
    Example:
        results = runner.map(
            graph,
            inputs={"queries": ["q1", "q2", "q3"], "config": shared_config},
            map_over="queries",
        )
        # results = [{"response": "r1"}, {"response": "r2"}, {"response": "r3"}]
    """
```

---

## AsyncRunner (Asynchronous)

### Constructor

```python
class AsyncRunner:
    def __init__(
        self,
        *,
        cache: Cache | None = None,
        callbacks: list[Callback] | None = None,
    ) -> None:
        """
        Create asynchronous runner.
        
        Args:
            cache: Cache backend.
            callbacks: Observability callbacks.
        """
```

### run()

```python
async def run(
    self,
    graph: Graph,
    inputs: dict[str, Any],
    *,
    select: list[str] | None = None,
    session_id: str | None = None,
    max_iterations: int = 1000,
    checkpoint: bytes | None = None,
) -> RunResult:
    """
    Execute graph asynchronously.
    
    Args:
        graph: Graph to execute.
        inputs: Input values.
        select: Outputs to return.
        session_id: Session identifier.
        max_iterations: Max iterations.
        checkpoint: Resume from saved state (for InterruptNode).
    
    Returns:
        RunResult with outputs and optional checkpoint.
    
    Example:
        runner = AsyncRunner()
        result = await runner.run(graph, inputs={"query": "hello"})
        
        if result.interrupted:
            # Handle interrupt, get user input
            result = await runner.run(
                graph,
                inputs={"user_decision": decision},
                checkpoint=result.checkpoint,
            )
    """
```

### RunResult

```python
@dataclass
class RunResult:
    outputs: dict[str, Any]      # Output values
    interrupted: bool            # True if stopped at InterruptNode
    checkpoint: bytes | None     # State for resume (if interrupted)
    run_id: str                  # Unique run identifier
    interrupt_name: str | None   # Name of interrupt (if interrupted)
    interrupt_value: Any | None  # Value to show user (if interrupted)
```

### iter()

```python
async def iter(
    self,
    graph: Graph,
    inputs: dict[str, Any],
    *,
    session_id: str | None = None,
    checkpoint: bytes | None = None,
) -> AsyncIterator[Event]:
    """
    Execute graph and yield events.
    
    Args:
        graph: Graph to execute.
        inputs: Input values.
        session_id: Session identifier.
        checkpoint: Resume from saved state.
    
    Yields:
        Event objects as they occur.
    
    Event types:
        - RunStartEvent: Execution beginning
        - NodeStartEvent: Node starting
        - NodeEndEvent: Node completed
        - StreamingChunkEvent: Token from generator
        - CacheHitEvent: Cache hit occurred
        - RouteDecisionEvent: Gate made decision
        - InterruptEvent: Paused for human input
        - RunEndEvent: Execution complete
    
    Example:
        async for event in runner.iter(graph, inputs=inputs):
            if isinstance(event, StreamingChunkEvent):
                print(event.chunk, end="", flush=True)
            elif isinstance(event, InterruptEvent):
                # Handle human-in-the-loop
                break
    """
```

### map()

```python
async def map(
    self,
    graph: Graph,
    inputs: dict[str, Any],
    *,
    map_over: str | list[str],
    select: list[str] | None = None,
    concurrency: int = 10,
) -> list[dict[str, Any]]:
    """
    Execute graph for each item with controlled concurrency.
    
    Args:
        graph: Graph to execute.
        inputs: Input values.
        map_over: Parameter(s) to iterate.
        select: Outputs to return.
        concurrency: Max concurrent executions.
    
    Returns:
        List of output dicts.
    """
```

---

## DaftRunner (Distributed)

### Constructor

```python
class DaftRunner:
    def __init__(
        self,
        *,
        cache: Cache | None = None,
    ) -> None:
        """
        Create distributed runner using Daft.
        
        Args:
            cache: Cache backend. Note: MemoryCache is per-worker.
        
        Note:
            DaftRunner only supports DAG graphs.
            Cycles, gates, and interrupts are not supported.
        """
```

### map()

```python
def map(
    self,
    graph: Graph,
    inputs: dict[str, Any],
    *,
    map_over: str | list[str],
    select: list[str] | None = None,
) -> "daft.DataFrame":
    """
    Execute graph distributed using Daft DataFrames.
    
    Args:
        graph: Must be DAG (no cycles).
        inputs: Input values.
        map_over: Parameter(s) to distribute.
        select: Outputs to return.
    
    Returns:
        Daft DataFrame with results.
    
    Raises:
        IncompatibleRunnerError: If graph has cycles/gates/interrupts.
    
    Example:
        runner = DaftRunner()
        df = runner.map(
            graph,
            inputs={"texts": large_text_list},
            map_over="texts",
        )
        results = df.collect()  # Trigger execution
    """
```

### Compatibility Validation

```python
def _validate_graph(self, graph: Graph) -> None:
    """Ensure graph is compatible with distributed execution."""
    
    if graph.has_cycles:
        raise IncompatibleRunnerError(
            "This graph has cycles, but DaftRunner doesn't support cycles.\n\n"
            f"The problem: DaftRunner uses Daft DataFrames for distributed\n"
            f"execution, which requires a DAG structure.\n\n"
            f"Cycles found: {graph.cycles}\n\n"
            f"How to fix:\n"
            f"  Option A: Use Runner or AsyncRunner instead\n"
            f"            → runner = AsyncRunner(cache=...)\n"
            f"  Option B: Restructure as a DAG"
        )
    
    if graph.gates:
        raise IncompatibleRunnerError(
            "This graph has gates (@route/@branch), but DaftRunner doesn't support gates.\n\n"
            f"Gates found: {[g.name for g in graph.gates]}\n\n"
            f"How to fix:\n"
            f"  Use Runner or AsyncRunner instead"
        )
    
    if graph.interrupt_nodes:
        raise IncompatibleRunnerError(
            "This graph has InterruptNodes, but DaftRunner doesn't support interrupts.\n\n"
            f"How to fix:\n"
            f"  Use AsyncRunner for human-in-the-loop workflows"
        )
```

---

## Event Types

### RunStartEvent

```python
@dataclass
class RunStartEvent:
    run_id: str
    session_id: str | None
    inputs: dict[str, Any]
    timestamp: float
```

### NodeStartEvent

```python
@dataclass
class NodeStartEvent:
    run_id: str
    node_name: str
    inputs: dict[str, Any]
    timestamp: float
```

### NodeEndEvent

```python
@dataclass
class NodeEndEvent:
    run_id: str
    node_name: str
    outputs: Any
    duration_ms: float
    cached: bool
    timestamp: float
```

### StreamingChunkEvent

```python
@dataclass
class StreamingChunkEvent:
    run_id: str
    node_name: str
    chunk: str | Any
    chunk_index: int
    timestamp: float
```

### CacheHitEvent

```python
@dataclass
class CacheHitEvent:
    run_id: str
    node_name: str
    timestamp: float
```

### RouteDecisionEvent

```python
@dataclass
class RouteDecisionEvent:
    run_id: str
    gate_name: str
    decision: str  # Target node name or "END"
    timestamp: float
```

### InterruptEvent

```python
@dataclass
class InterruptEvent:
    run_id: str
    interrupt_name: str
    value: Any              # Value to show user
    response_param: str     # Where to put response
    checkpoint: bytes       # State for resume
    timestamp: float
```

### RunEndEvent

```python
@dataclass
class RunEndEvent:
    run_id: str
    outputs: dict[str, Any]
    duration_ms: float
    iterations: int
    timestamp: float
```

---

## Identity Model

```python
# session_id: User-provided, groups related runs
# run_id: Framework-generated, identifies single execution

result = await runner.run(
    graph,
    inputs={...},
    session_id="conversation-123",  # User provides
)
# result.run_id → "run-abc-456"  # Framework generates
```

Use cases:
- `session_id`: Group multi-turn conversation runs
- `run_id`: Trace/debug specific execution
