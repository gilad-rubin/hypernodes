# Execution & Runtime Types

**Reference for execution state, results, events, and observability types.**

These types represent the runtime layer of HyperNodes - what happens when graphs execute.

---

## Overview

### The Execution Model

When a graph runs, it progresses through three conceptual layers:

1. **Structure** (Graph + Nodes) - What to execute
2. **State** (GraphState) - What has been executed and current values
3. **Results** (GraphResult/RunResult + Events) - What was produced

```
Graph Definition  →  Runtime State  →  Results & Events
  (structure)         (versioned)        (observable)
```

### Key Concepts

**Versioned State**: Every value has a version number that increments on update. This enables:
- Staleness detection (when to re-execute nodes)
- Cycle support (accumulators in loops)
- Checkpointing (pause/resume workflows)

**Event Streaming**: Runners emit events during execution for:
- Real-time UI updates (`NodeStartEvent`, `StreamingChunkEvent`)
- Observability/logging (`NodeEndEvent`, `RouteDecisionEvent`)
- Human-in-the-loop (`InterruptEvent`)

**Layered Architecture**: Events flow through pluggable layers:
- **UI Protocol** - WebSocket streaming to frontends
- **Observability** - Logging, tracing (Langfuse, Logfire)
- **Durability** - Checkpoint persistence (Redis, PostgreSQL, SQLite)

---

## Quick Navigation

| Type | Purpose | Usage |
|------|---------|-------|
| [GraphState](#graphstate) | Runtime value storage | Internal to execution |
| [GraphResult](#graphresult) | Execution results | Returned by `.run()`, nested graphs |
| [RunResult](#runresult) | Async execution result | Returned by `AsyncRunner.run()` |
| [Event Types](#event-types) | Streaming events | Yielded by `AsyncRunner.iter()` |

**See also:**
- [Node Types](node-types.md) - Building blocks (includes InterruptNode)
- [Graph Types](graph-types.md) - Structure and composition
- [Runners API](runners.md) - Execution guide

---

## Three-Layer Architecture

HyperNodes uses a **unified event stream** that flows through pluggable layers. The core execution engine produces events; layers consume what they need.

```
                    ┌─────────────────────────┐
                    │   Core Execution        │
                    │   (Runners)             │
                    └───────────┬─────────────┘
                                │ Events (with span hierarchy)
                    ┌───────────▼─────────────┐
                    │   Event Stream          │
                    └─┬──────────┬──────────┬─┘
                      │          │          │
        ┌─────────────▼──┐  ┌────▼─────┐  ┌▼────────────┐
        │  UI Protocol   │  │ Event    │  │ Durability  │
        │  (WebSocket)   │  │Processors│  │ (Checkpoint)│
        └────────────────┘  └──────────┘  └─────────────┘
```

### Layer Details

| Layer | Purpose | Example | Protocol |
| --- | --- | --- | --- |
| **UI Protocol** | Real-time streaming to frontends | AG-UI compatible streaming | Events → WebSocket |
| **Observability** | Logging, tracing, analytics | Langfuse, Logfire integration | Events → EventProcessor |
| **Durability** | Checkpoint persistence | Redis, PostgreSQL, SQLite | Checkpointer interface |

**Key principle:** Layers consume a unified event stream. The core produces events; layers subscribe to what they need. See [Observability](observability.md) for the `EventProcessor` interface and integration patterns.

### Event Flow Example

```python
# Runner produces events via .iter()
async for event in runner.iter(graph, inputs={...}):
    # UI layer consumes streaming chunks
    if isinstance(event, StreamingChunkEvent):
        await websocket.send(event.chunk)

    # Observability layer logs all events
    logger.info(f"{event.node_name}: {event.duration_ms}ms")

    # Durability layer saves checkpoints
    if isinstance(event, InterruptEvent):
        await db.save_checkpoint(event.checkpoint)

# Or use EventProcessor for push-based observability
runner = AsyncRunner(event_processors=[LangfuseProcessor()])
result = await runner.run(graph, inputs={...})  # Events sent to processors
```

---

## GraphState

### Purpose

**Runtime storage for value versions and execution history.** Used internally by runners to track what's been computed and when to re-execute nodes.

### Class Definition

```python
@dataclass
class GraphState:
    """Runtime value storage with versioning."""

    values: dict[str, Any]
    """Current values by name."""

    versions: dict[str, int]
    """Version number for each value (increments on update)."""

    node_executions: dict[str, NodeExecution]
    """Last execution record per node."""

    history: list[NodeExecution]
    """Chronological execution history."""
```

### Properties

```python
@property
def value_names(self) -> set[str]:
    """All value names currently in state."""
    return set(self.values.keys())

def get(self, name: str, default=None) -> Any:
    """Get value by name."""
    return self.values.get(name, default)

def has(self, name: str) -> bool:
    """Check if value exists."""
    return name in self.values

def version(self, name: str) -> int:
    """Get version number for a value."""
    return self.versions.get(name, 0)
```

### Methods

```python
def set(self, name: str, value: Any) -> GraphState:
    """Set value and increment version (returns new state)."""

def is_stale(self, node: HyperNode) -> bool:
    """Check if node needs to re-execute based on input versions."""

def checkpoint(self) -> bytes:
    """Serialize state for persistence."""

@staticmethod
def from_checkpoint(data: bytes) -> GraphState:
    """Restore state from checkpoint."""
```

### Example (Internal Use)

```python
# Created and managed by runners, not user-facing
state = GraphState(
    values={"x": 5, "doubled": 10},
    versions={"x": 0, "doubled": 1},
    node_executions={...},
    history=[...]
)

# Check staleness
if state.is_stale(add_ten_node):
    # Re-execute node
    ...
```

---

## GraphResult

### Purpose

**Results from graph execution, including nested graph outputs.** Provides dict-like access to outputs.

### Class Definition

```python
@dataclass
class GraphResult:
    """Result from graph execution."""

    outputs: dict[str, Any | "GraphResult"]
    """Output values and nested GraphResults."""

    status: Literal["complete", "interrupted", "error"]
    """Execution status."""

    history: list[NodeExecution] | None = None
    """Execution history (if tracking enabled)."""
```

### Methods

```python
def __getitem__(self, key: str) -> Any | GraphResult:
    """Dict-like access to outputs."""
    return self.outputs[key]

def keys(self):
    """Get output names."""
    return self.outputs.keys()

def items(self):
    """Get output name-value pairs."""
    return self.outputs.items()

def __contains__(self, key: str):
    """Check if output exists."""
    return key in self.outputs
```

### Example

```python
# Nested graph execution
inner = Graph(nodes=[embed, retrieve], name="rag")
outer = Graph(nodes=[preprocess, inner.as_node(), postprocess])

result = runner.run(outer, inputs={"query": "hello"})

# Access results
result["final"]                    # Top-level output
result["rag"]                      # Nested GraphResult
result["rag"]["embedding"]         # Nested graph output
result.status                      # "complete"

# Dict-like access
assert "rag" in result
for name, value in result.items():
    print(f"{name}: {value}")
```

---

## RunResult

### Purpose

**Extended result from AsyncRunner with interrupt/checkpoint support.** Returned by `AsyncRunner.run()`.

### Class Definition

```python
@dataclass
class RunResult:
    """Result from async graph execution."""

    outputs: dict[str, Any]
    """Output values."""

    interrupted: bool
    """True if stopped at InterruptNode."""

    checkpoint: bytes | None
    """Serialized state for resume (if interrupted)."""

    run_id: str
    """Unique identifier for this execution."""

    interrupt_name: str | None = None
    """Name of interrupt point (if interrupted)."""

    interrupt_value: Any | None = None
    """Value to show user (if interrupted)."""
```

### Example

```python
from hypernodes import AsyncRunner

runner = AsyncRunner()
result = await runner.run(graph, inputs={...})

if result.interrupted:
    # Show prompt to user
    prompt = result.interrupt_value
    print(f"Paused at: {result.interrupt_name}")

    # Get user's response
    response = await get_user_input(prompt)

    # Resume
    result = await runner.run(
        graph,
        inputs={result.interrupt_name: response},
        checkpoint=result.checkpoint,
    )

# Access outputs
print(result.outputs["answer"])
```

---

## Event Types

Events are emitted by `AsyncRunner.iter()` for real-time observability and UI updates. Events can also be consumed via `EventProcessor` for push-based integrations.

All events include **span hierarchy fields** for nested graph support:

```python
# Common fields on all events
run_id: str              # Unique per .run() invocation
span_id: str             # Unique per node execution
parent_span_id: str | None  # Links to parent span (None for root nodes)
timestamp: float         # Unix timestamp
```

The `span_id` → `parent_span_id` relationship forms a tree, enabling observability tools to visualize nested graph execution. See [Observability](observability.md) for details.

### NodeStartEvent

```python
@dataclass
class NodeStartEvent:
    run_id: str
    span_id: str
    parent_span_id: str | None
    node_name: str
    inputs: dict[str, Any]
    timestamp: float
```

### NodeEndEvent

```python
@dataclass
class NodeEndEvent:
    run_id: str
    span_id: str
    parent_span_id: str | None
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
    span_id: str
    parent_span_id: str | None
    node_name: str
    chunk: str | Any
    chunk_index: int
    timestamp: float
```

### InterruptEvent

```python
@dataclass
class InterruptEvent:
    run_id: str
    span_id: str
    parent_span_id: str | None
    interrupt_name: str
    value: Any              # Value to show user
    response_param: str     # Where to write response
    checkpoint: bytes       # State for resume
    timestamp: float
```

### RouteDecisionEvent

```python
@dataclass
class RouteDecisionEvent:
    run_id: str
    span_id: str
    parent_span_id: str | None
    gate_name: str
    decision: str  # Target node name or "END"
    timestamp: float
```

### NodeErrorEvent

```python
@dataclass
class NodeErrorEvent:
    run_id: str
    span_id: str
    parent_span_id: str | None
    node_name: str
    error: Exception        # The exception that was raised
    error_type: str         # Class name, e.g., "ValueError"
    timestamp: float
```

**Note:** After `NodeErrorEvent`, execution may continue (if error is handled) or terminate. `RunEndEvent` is still emitted with error status. Processors' `shutdown()` is always called.

### Example

```python
async with runner.iter(graph, inputs={...}) as run:
    async for event in run:
        match event:
            case NodeStartEvent(node_name=name):
                print(f"Starting: {name}")

            case NodeEndEvent(node_name=name, duration_ms=ms):
                print(f"Finished: {name} in {ms}ms")

            case StreamingChunkEvent(chunk=chunk):
                print(chunk, end="")

            case InterruptEvent(interrupt_name=name, value=prompt):
                print(f"Paused at: {name}")
                # Handle interrupt
```

---

## Interrupt Handling with AsyncRunner

**AsyncRunner supports interrupts in `.run()` and `.iter()`, but NOT in `.map()`.**

### Using `.run()` - Pause and Resume

```python
result = await runner.run(graph, inputs={"query": "hello"})

if result.interrupted:
    # Execution paused at InterruptNode
    prompt = result.interrupt_value

    # Get user response (your application logic)
    response = await get_user_response(prompt)

    # Resume with checkpoint
    result = await runner.run(
        graph,
        inputs={result.interrupt_name: response},
        checkpoint=result.checkpoint,
    )

# Now complete
assert not result.interrupted
print(result.outputs["answer"])
```

### Using `.iter()` - Handle Inline

```python
async with runner.iter(graph, inputs={"query": "hello"}) as run:
    async for event in run:
        match event:
            case StreamingChunkEvent(chunk=chunk):
                print(chunk, end="")

            case InterruptEvent(value=prompt, response_param=target):
                # Handle interrupt inline
                response = await get_user_response(prompt)
                run.respond(target, response)
                # Iteration continues automatically

            case NodeEndEvent(node_name=name):
                print(f"Completed: {name}")

    # After iteration, result is available
    print(run.result.outputs)
```

### Using `.run()` with Handlers

Pass handlers per-call for automatic interrupt resolution:

```python
result = await runner.run(
    graph,
    inputs={"query": "hello"},
    interrupt_handlers={
        "approval": handle_approval,
        "topic_selection": handle_topic,
    },
)

# If all interrupts have handlers → runs to completion
# If some handlers missing → returns interrupted at first unhandled
```

Handler signature:

```python
async def handle_approval(prompt: ApprovalPrompt) -> ApprovalResponse:
    """
    Receives: The value from InterruptNode's input_param
    Returns: The value to write to InterruptNode's response_param
    """
    choice = await show_dialog(prompt.message, prompt.options)
    return ApprovalResponse(choice=choice)
```

### `.map()` Does Not Support Interrupts

Batch processing with `.map()` cannot handle interrupts:

```python
# This will raise an error at validation time
if graph.has_interrupts:
    raise IncompatibleRunnerError(
        "Graph has interrupts but .map() doesn't support them.\n"
        "Use .run() or .iter() for graphs with interrupts."
    )
```

Rationale: Each batch item would potentially pause at different points, making the execution model complex and confusing. Use `.run()` in a loop if you need batch processing with interrupts.

---

## Common Patterns

### Working with Nested Results

```python
def extract_all_values(result: GraphResult, prefix="") -> dict[str, Any]:
    """Recursively extract all values from nested results."""
    flat = {}

    for key, value in result.items():
        full_key = f"{prefix}{key}" if prefix else key

        if isinstance(value, GraphResult):
            # Recurse into nested result
            nested = extract_all_values(value, f"{full_key}/")
            flat.update(nested)
        else:
            flat[full_key] = value

    return flat
```

### Event Filtering and Routing

```python
async def process_events(graph: Graph):
    """Route events to different handlers based on type."""
    async with runner.iter(graph, inputs={...}) as run:
        async for event in run:
            match event:
                case StreamingChunkEvent():
                    await ui_layer.handle_chunk(event)

                case NodeEndEvent():
                    await observability_layer.log_execution(event)

                case InterruptEvent():
                    await durability_layer.save_checkpoint(event)
                    await ui_layer.prompt_user(event)
```

### Checkpoint Persistence

```python
# Save checkpoint to database
async def save_checkpoint(event: InterruptEvent):
    await db.execute(
        "INSERT INTO checkpoints (run_id, interrupt_name, data) VALUES (?, ?, ?)",
        (event.run_id, event.interrupt_name, event.checkpoint)
    )

# Resume from checkpoint
async def resume_execution(run_id: str, user_response: Any):
    checkpoint_data = await db.fetch_one(
        "SELECT data, interrupt_name FROM checkpoints WHERE run_id = ?",
        (run_id,)
    )

    result = await runner.run(
        graph,
        inputs={checkpoint_data["interrupt_name"]: user_response},
        checkpoint=checkpoint_data["data"],
    )
    return result
```

### Multi-Layer Event Consumer

```python
class EventRouter:
    """Route events to multiple layers simultaneously."""

    def __init__(self):
        self.ui_layer = WebSocketLayer()
        self.observability_layer = LogfuseLayer()
        self.durability_layer = PostgresCheckpointer()

    async def consume(self, graph: Graph, inputs: dict):
        async with runner.iter(graph, inputs=inputs) as run:
            async for event in run:
                # All layers receive all events - they filter what they need
                await asyncio.gather(
                    self.ui_layer.handle(event),
                    self.observability_layer.handle(event),
                    self.durability_layer.handle(event),
                )

        return run.result
```

---

## Type Hierarchy

```
Graph (structure definition)
├── InputSpec (input parameter specification, returned by .inputs)
└── GraphState (runtime values)
    └── GraphResult (execution results)
        └── RunResult (async execution with interrupts)

Event Hierarchy (all include span_id, parent_span_id for hierarchy):
├── RunStartEvent
├── RunEndEvent
├── NodeStartEvent
├── NodeEndEvent
├── NodeErrorEvent
├── StreamingChunkEvent
├── CacheHitEvent
├── InterruptEvent
└── RouteDecisionEvent

Observability:
├── EventProcessor (base interface)
└── TypedEventProcessor (convenience class with typed methods)
```

**See also:**
- [Observability](observability.md) - EventProcessor interface and integration patterns
- [Node Types](node-types.md#type-hierarchy-summary) - Complete node hierarchy
- [Graph Types](graph-types.md#runner-compatibility) - Runner compatibility matrix
