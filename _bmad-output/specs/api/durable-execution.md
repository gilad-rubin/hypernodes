# Durable Execution

**From development save points to production-grade crash recovery.**

> **Type Reference:** For all type definitions (`RunResult`, `RunStatus`, `PauseReason`, `Workflow`, `Step`, etc.), see [Execution Types](execution-types.md). This document focuses on usage patterns.

---

## Overview

HyperNodes provides two paths to durability:

1. **Built-in Checkpointer** — Manual resume, simple production, development
2. **DBOS Runners** — Automatic crash recovery, production infrastructure

Both paths keep your graph code **pure and portable**. The graph never imports durability-specific code.

```
┌─────────────────────────────────────────────────────────────────────┐
│  YOUR CODE                                                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  HYPERNODES LAYER                                           │   │
│  │                                                             │   │
│  │  @node, Graph, InterruptNode, generators (yield)           │   │
│  │                                                             │   │
│  │  → Pure, portable, runner-agnostic                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  RUNNER + DURABILITY LAYER                                  │   │
│  │                                                             │   │
│  │  AsyncRunner + Checkpointer    OR    DBOSAsyncRunner       │   │
│  │  (manual resume)                     (automatic recovery)   │   │
│  │                                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Design Principles

1. **Graph code stays pure** — No durability imports in nodes
2. **Explicit over implicit** — No magic defaults, user knows what they're opting into
3. **Declarative nodes, not ambient side effects** — `InterruptNode` in graph, not `interrupt()` in function
4. **Generators for streaming** — Use `yield` for progress and streaming, not special functions
5. **Clear upgrade path** — Start simple, add durability when needed
6. **Outputs ARE state** — No separate state schema; node outputs are the state (see [State Model](state-model.md))

---

## Selective Persistence

By default, all node outputs are checkpointed. Use the `persist` parameter to control what's saved.

### Why Selective Persistence?

Not all outputs need to survive crashes:

| Output Type | Example | Should Persist? |
|-------------|---------|-----------------|
| Conversation history | `messages` | ✅ Yes - can't reconstruct |
| Final answers | `answer` | ✅ Yes - user expects this |
| Embeddings | `embedding` | ❌ No - can regenerate |
| Intermediate docs | `retrieved_docs` | ❌ No - can refetch |

### Configuration

**Graph-level (allowlist):**

```python
graph = Graph(
    nodes=[embed, retrieve, generate],
    persist=["messages", "answer"],  # Only these are checkpointed
)
```

**Node-level (override):**

```python
@node(output_name="embedding", persist=False)  # Never checkpoint
def embed(text: str) -> list[float]:
    return model.embed(text)

@node(output_name="answer", persist=True)  # Always checkpoint
def generate(docs: list[str]) -> str:
    return llm.generate(docs)
```

### Resolution Order

```
1. Node-level persist=True/False  → Explicit override, always wins
2. Graph-level persist=[...]      → Allowlist of output names
3. Default (no persist specified) → All outputs checkpointed
```

### Semantics

| `persist` | On Crash/Resume | Storage | DBOS Mapping |
|-----------|-----------------|---------|--------------|
| `True` | Load from checkpoint | Saved to DB | `@DBOS.step` |
| `False` | Re-execute node | Not saved | Regular function call |

### Example

```python
@node(output_name="embedding", persist=False)  # Large, reproducible
def embed(text: str) -> list[float]:
    return model.embed(text)

@node(output_name="docs")  # Follows graph policy
def retrieve(embedding: list[float]) -> list[str]:
    return db.search(embedding)

@node(output_name="answer")  # Follows graph policy
def generate(docs: list[str], messages: list) -> str:
    return llm.generate(docs, messages)

graph = Graph(
    nodes=[embed, retrieve, generate],
    persist=["messages", "answer"],
)

# What gets checkpointed:
# ❌ embedding - node says persist=False
# ❌ docs      - not in graph's persist list
# ✅ messages  - in persist list (passed as input, returned by another node)
# ✅ answer    - in persist list
```

### Resume Behavior

On crash and resume:

1. **Persisted outputs** → Loaded from checkpoint, node skipped
2. **Non-persisted outputs** → Node re-executes to reconstruct value

```
Original run:
  embed("hello") → [0.1, 0.2, ...]  ← NOT saved (persist=False)
  generate(...) → "answer"          ← SAVED
  💥 CRASH

Resume:
  embed("hello") → [0.1, 0.2, ...]  ← Re-executed
  generate(...) → (loaded)          ← Loaded from checkpoint
  ✅ Complete
```

### Important Notes

1. **Non-determinism is OK** — If `persist=False` nodes produce slightly different outputs on resume (e.g., embedding model updates), that's expected. Users working with AI understand non-determinism.

2. **Don't use `persist=False` for context-dependent code** — If a node reads from external context that might change (current user, system time), it should be persisted.

3. **Default is safe** — When in doubt, let outputs be checkpointed (the default).

---

## Quick Comparison

| Need | Solution | Install |
|------|----------|---------|
| Development save points | `AsyncRunner(checkpointer=SqliteCheckpointer(...))` | `pip install hypernodes` |
| Simple production (manual resume) | Same as above | `pip install hypernodes` |
| Automatic crash recovery | `DBOSAsyncRunner()` | `pip install hypernodes[dbos]` |
| Durable queues, scheduling | `DBOSAsyncRunner` + DBOS features | `pip install hypernodes[dbos]` |

---

## Path 1: Built-in Checkpointer

For development and simple production use cases where manual resume is acceptable.

### Basic Usage

```python
from hypernodes import Graph, node, AsyncRunner
from hypernodes.checkpointers import SqliteCheckpointer

@node(output_name="result")
async def fetch(query: str) -> dict:
    return await api.search(query)

@node(output_name="summary")
async def summarize(result: dict) -> str:
    return await llm.summarize(result)

graph = Graph(nodes=[fetch, summarize])

# Add checkpointing
runner = AsyncRunner(checkpointer=SqliteCheckpointer("./dev.db"))

result = await runner.run(
    graph,
    inputs={"query": "hello"},
    workflow_id="session-123",  # Required with checkpointer
)
```

### Manual Resume

The checkpointer stores state internally, keyed by `workflow_id`. You don't pass checkpoint bytes around.

```python
# First run — interrupted or crashed
result = await runner.run(
    graph,
    inputs={"query": "hello"},
    workflow_id="session-123",
)

# Resume from where we left off
# Checkpointer loads state internally by workflow_id
result = await runner.run(
    graph,
    inputs={"query": "hello"},
    workflow_id="session-123",
    resume=True,  # Checkpointer loads latest state for this workflow_id
)
```

### Human-in-the-Loop with InterruptNode

```python
from hypernodes import Graph, node, InterruptNode

@node(output_name="draft")
async def generate(prompt: str) -> str:
    return await llm.generate(prompt)

# Declarative pause point — visible in graph structure
approval = InterruptNode(
    name="approval",
    input_param="draft",       # Value shown to user
    response_param="decision", # Where user response goes
)

@node(output_name="final")
def finalize(draft: str, decision: str) -> str:
    if decision == "approve":
        return draft
    return f"REJECTED: {draft}"

graph = Graph(nodes=[generate, approval, finalize])

# First run — pauses at approval
result = await runner.run(
    graph,
    inputs={"prompt": "Write a poem"},
    workflow_id="poem-456",
)

if result.pause:
    print(f"Draft: {result.pause.value}")
    # Wait for user...

# Later — resume with user's decision
# Pass the response using the response_param name
result = await runner.run(
    graph,
    inputs={"decision": "approve"},  # Uses response_param name
    workflow_id="poem-456",
    resume=True,
)
```

### Streaming with Generators

Use `yield` for streaming — no special functions needed:

```python
@node(output_name="response")
async def stream_llm(prompt: str):
    """Generator node that streams tokens."""
    async for chunk in llm.stream(prompt):
        yield chunk
    # Final value is the concatenation of all chunks
```

**In `.run()` mode with EventProcessor (push-based):**

```python
class StreamingHandler(TypedEventProcessor):
    def on_streaming_chunk(self, event: StreamingChunkEvent) -> None:
        print(event.chunk, end="", flush=True)

runner = AsyncRunner(
    checkpointer=SqliteCheckpointer("./dev.db"),
    event_processors=[StreamingHandler()],
)
result = await runner.run(graph, inputs={...}, workflow_id="123")
# Chunks printed as they arrive, final value in result.outputs
```

**In `.iter()` mode (pull-based):**

```python
async for event in runner.iter(graph, inputs={...}, workflow_id="123"):
    match event:
        case StreamingChunkEvent(chunk=chunk):
            print(chunk, end="", flush=True)
        case InterruptEvent(value=draft):
            # Handle interrupt
            break
```

### Progress Reporting

Yield progress objects for UI updates:

```python
from dataclasses import dataclass

@dataclass
class Progress:
    percent: float
    message: str

@node(output_name="result")
async def process_with_progress(items: list):
    results = []
    for i, item in enumerate(items):
        yield Progress(i / len(items), f"Processing {i+1}/{len(items)}")
        results.append(await process_item(item))
    yield Progress(1.0, "Complete")
    # Return final value by yielding it last
    yield results
```

Filter by type in your event handler:

```python
class ProgressHandler(TypedEventProcessor):
    def on_streaming_chunk(self, event: StreamingChunkEvent) -> None:
        if isinstance(event.chunk, Progress):
            update_progress_bar(event.chunk.percent, event.chunk.message)
```

### Checkpointer Interface

```python
from abc import ABC, abstractmethod
from hypernodes.types import Workflow, Step  # See execution-types.md

class Checkpointer(ABC):
    """Base class for checkpoint storage.

    For type definitions (Workflow, Step, StepResult), see execution-types.md.
    """

    @abstractmethod
    async def save(self, workflow_id: str, workflow: Workflow) -> None:
        """Save workflow state."""
        ...

    @abstractmethod
    async def load(self, workflow_id: str) -> Workflow | None:
        """Load latest state for workflow, or None if not found."""
        ...

    @abstractmethod
    async def list_workflows(self) -> list[Workflow]:
        """List all stored workflows."""
        ...

    @abstractmethod
    async def get_history(self, workflow_id: str) -> list[Step]:
        """Get step execution history for a workflow."""
        ...
```

**See [Execution Types](execution-types.md#persistence-types)** for `Workflow`, `Step`, `StepResult`, and status enum definitions.

### Checkpointer Capabilities

| Capability | Supported |
|------------|:---------:|
| Resume from latest checkpoint | ✅ |
| Get current state | ✅ |
| List workflows | ✅ |
| Step history | ✅ |
| Automatic crash recovery | ❌ |
| Resume from specific checkpoint | ❌ |
| Workflow forking (time travel) | ❌ |
| Durable sleep | ❌ |
| Durable queues | ❌ |

For advanced features, use DBOS.

---

## Path 2: DBOS Runners

For production workloads requiring automatic crash recovery.

### What DBOS Provides

DBOS is a **library** that runs in your process and checkpoints to Postgres/SQLite using pickle serialization. If your process crashes and restarts, DBOS automatically recovers pending workflows.

```
┌─────────────────────────────────────────────────────────────────────┐
│  DBOS RECOVERY (automatic)                                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Your process runs graph                                         │
│     step_1() → checkpoint ✓                                         │
│     step_2() → checkpoint ✓                                         │
│     step_3() → 💥 CRASH                                             │
│                                                                     │
│  2. Process restarts                                                │
│     DBOS.launch()  ← Triggers automatic recovery                    │
│                                                                     │
│  3. DBOS replays from checkpoints                                   │
│     step_1() → cached ⚡ (skip)                                      │
│     step_2() → cached ⚡ (skip)                                      │
│     step_3() → execute → checkpoint ✓                               │
│     done! ✓                                                         │
│                                                                     │
│  No resume=True needed — DBOS handles it!                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### How `persist` Maps to DBOS

HyperNodes maps the `persist` parameter to DBOS primitives:

| `persist` | DBOS Mapping | On Recovery |
|-----------|--------------|-------------|
| `True` (default) | `@DBOS.step` wrapper | Output loaded from DB |
| `False` | Regular function call | Function re-executes |

```python
@DBOS.workflow()
async def graph_workflow(inputs: dict) -> dict:
    # persist=True → wrapped as DBOS step
    answer = await generate_step(inputs["prompt"])  # @DBOS.step

    # persist=False → regular function call
    embedding = embed(inputs["text"])  # NOT a step, re-runs on recovery

    return {"answer": answer, "embedding": embedding}
```

This follows DBOS's own recommendation:
> "Skip the decorator if durability isn't needed, so you avoid the extra DB checkpoint write."

When using `.get_dbos_workflow()` for advanced DBOS features, the same mapping applies.

### Basic Usage

```python
from hypernodes import Graph, node
from hypernodes.runners import DBOSAsyncRunner

@node(output_name="result")
async def fetch(query: str) -> dict:
    return await api.search(query)

@node(output_name="summary")
async def summarize(result: dict) -> str:
    return await llm.summarize(result)

graph = Graph(nodes=[fetch, summarize])

# DBOS runner — automatic durability
runner = DBOSAsyncRunner()  # SQLite by default (zero config)

result = await runner.run(
    graph,
    inputs={"query": "hello"},
    workflow_id="order-123",  # Required for DBOS
)
```

### With Postgres (Production)

```python
runner = DBOSAsyncRunner(database_url="postgresql://user:pass@host/db")
```

### Human-in-the-Loop with DBOS

`InterruptNode` maps to DBOS's `recv()`/`send()` messaging pattern:

```python
from hypernodes import Graph, node, InterruptNode

@node(output_name="draft")
async def generate(prompt: str) -> str:
    return await llm.generate(prompt)

approval = InterruptNode(
    name="approval",
    input_param="draft",
    response_param="decision",
)

@node(output_name="final")
def finalize(draft: str, decision: str) -> str:
    return draft if decision == "approve" else "REJECTED"

graph = Graph(nodes=[generate, approval, finalize])
runner = DBOSAsyncRunner()

# First run — workflow pauses at InterruptNode
# Under the hood: DBOS.recv("approval") waits for signal
result = await runner.run(
    graph,
    inputs={"prompt": "Write a poem"},
    workflow_id="poem-456",
)

print(result.pause is not None)  # True
print(result.pause.value)        # The draft
```

**Resuming with DBOS uses `send()` from external system:**

```python
from dbos import DBOS

# From webhook, API endpoint, or external process:
DBOS.send(
    destination_id="poem-456",  # workflow_id
    message={"decision": "approve"},
    topic="approval",  # InterruptNode name
)
# Workflow automatically continues — no runner.run() call needed!
```

This is fundamentally different from the checkpointer approach:
- **Checkpointer:** You call `runner.run(..., resume=True)`
- **DBOS:** External system calls `DBOS.send()`, workflow auto-resumes

### Streaming with DBOS

**Important:** `.iter()` is not recommended with DBOS due to limitations in how DBOS wraps workflows. Use `EventProcessor` (push-based) instead:

```python
class StreamingHandler(TypedEventProcessor):
    def on_streaming_chunk(self, event: StreamingChunkEvent) -> None:
        print(event.chunk, end="", flush=True)

    def on_interrupt(self, event: InterruptEvent) -> None:
        # Persist interrupt for external handling
        save_pending_approval(event.workflow_id, event.value)

runner = DBOSAsyncRunner(event_processors=[StreamingHandler()])
result = await runner.run(graph, inputs={...}, workflow_id="123")
```

This aligns with how other frameworks (like Pydantic AI) integrate with DBOS.

### DBOS Runner Capabilities

```python
@dataclass
class DBOSRunnerCapabilities(RunnerCapabilities):
    supports_cycles: bool = True
    supports_gates: bool = True
    supports_interrupts: bool = True
    supports_async_nodes: bool = True
    supports_streaming: bool = True  # Via EventProcessor only
    supports_distributed: bool = False
    returns_coroutine: bool = True

    # DBOS-specific
    supports_automatic_recovery: bool = True
    supports_workflow_fork: bool = True
    supports_durable_sleep: bool = True
    supports_durable_queues: bool = True
```

---

## Parallel Execution

HyperNodes executes nodes in **batches** (supersteps). Nodes within the same batch run concurrently.

### The Challenge

```
Batch 0: [fetch, embed, retrieve]  ← 3 async nodes run in parallel
         ↓
         Process crashes after fetch and embed complete
         ↓
         On resume: How to know which completed?
```

### Solution: Pre-register All Steps

When a batch starts, we create `StepSnapshot` records for ALL nodes in the batch with `status="pending"`. As each completes, we update its status individually.

```python
# Execution flow (simplified)
async def execute_batch(batch: list[HyperNode], workflow_id: str, start_index: int):
    # 1. Pre-register all steps as "pending"
    step_indices = []
    for i, node in enumerate(batch):
        step_index = start_index + i
        step = StepSnapshot(
            workflow_id=workflow_id,
            step_index=step_index,
            node_name=node.name,
            node_type="node",
            batch_index=batch.index,
            status="pending",
        )
        await checkpointer.save_step(step)
        step_indices.append(step_index)

    # 2. Execute all in parallel with individual checkpointing
    async def execute_one(node: HyperNode, step_index: int):
        # Check if already completed (resume case)
        snapshot = await checkpointer.load_step(workflow_id, step_index)
        if snapshot and snapshot.status == "completed":
            result = await checkpointer.load_result(workflow_id, step_index)
            return result.outputs  # Skip, use cached result

        # Update to running
        await checkpointer.update_step(workflow_id, step_index, status="running")

        try:
            outputs = await node.execute(inputs)
            # Save result separately
            await checkpointer.save_result(StepResult(
                workflow_id=workflow_id,
                step_index=step_index,
                outputs=outputs,
            ))
            # Update snapshot status
            await checkpointer.update_step(
                workflow_id, step_index,
                status="completed",
                completed_at=datetime.utcnow(),
            )
            return outputs
        except Exception as e:
            await checkpointer.save_result(StepResult(
                workflow_id=workflow_id,
                step_index=step_index,
                error=str(e),
            ))
            await checkpointer.update_step(
                workflow_id, step_index,
                status="failed",
            )
            raise

    # 3. Run all concurrently
    results = await asyncio.gather(*[
        execute_one(node, idx)
        for node, idx in zip(batch, step_indices)
    ])

    return results
```

### Key Principle: Deterministic Scheduling

From [Temporal](https://docs.temporal.io/workflows) and [DBOS](https://docs.dbos.dev/architecture):

> "Workflows must be deterministic... the order of *starting* steps must be the same on replay."

**Parallel nodes are identified by their scheduling order, not completion order.**

```
Batch 0 starts:
  → step_index=0: fetch   (scheduled first)
  → step_index=1: embed   (scheduled second)
  → step_index=2: retrieve (scheduled third)

Completion order may vary:
  → embed completes first (step_index=1 → "completed")
  → fetch completes second (step_index=0 → "completed")
  → CRASH before retrieve completes

On resume:
  → step_index=0: fetch → status="completed" → skip
  → step_index=1: embed → status="completed" → skip
  → step_index=2: retrieve → status="running" → re-execute
```

### Parallel Nodes Are Steps, Not Child Workflows

**Important distinction:**

| Concept | What It Is | Checkpoint Model |
|---------|-----------|------------------|
| **Parallel nodes** | Multiple nodes in same batch | Steps within current workflow |
| **Nested graph** | GraphNode containing subgraph | Child workflow |

Parallel nodes do NOT become child workflows. They're just steps that happen to run concurrently within the same workflow.

---

## Nested Graphs

When a `GraphNode` executes, it creates a step with `child_workflow_id` pointing to a separate workflow. Nested graphs are fully supported with flat storage and string-based parent-child relationships.

### Workflow ID Convention

```python
parent_id = "order-123"
child_id = child_workflow_id(parent_id, "rag")  # "order-123/rag"

# Helper functions (not classes)
def child_workflow_id(parent_id: str, node_name: str) -> str:
    return f"{parent_id}/{node_name}"

def parent_workflow_id(workflow_id: str) -> str | None:
    if "/" not in workflow_id:
        return None
    return workflow_id.rsplit("/", 1)[0]
```

### Flat Storage Model

Two separate `Workflow` records linked by string reference:

```
Workflow(id="order-123")
├── Step(index=0, node_name="fetch")
├── Step(index=1, node_name="embed")
├── Step(index=2, node_name="rag", child_workflow_id="order-123/rag")
└── Step(index=3, node_name="postprocess")

Workflow(id="order-123/rag")  # Separate record
├── Step(index=0, node_name="inner_embed")
└── Step(index=1, node_name="inner_retrieve")
```

No recursive in-memory structure. Just flat records with string references.

### Execution (Simplified)

```python
async def execute_graph_node(
    graph_node: GraphNode,
    workflow_id: str,
    step_index: int,
    inputs: dict,
):
    # 1. Create step with child reference
    child_id = child_workflow_id(workflow_id, graph_node.name)

    step = Step(
        index=step_index,
        node_name=graph_node.name,
        batch_index=batch.index,
        status=StepStatus.RUNNING,
        child_workflow_id=child_id,
    )
    await checkpointer.save_step(workflow_id, step)

    # 2. Execute child as separate workflow
    child_workflow = Workflow(id=child_id)
    result = await execute_graph(graph_node.graph, child_workflow, inputs)

    # 3. Mark parent step complete
    step.status = StepStatus.COMPLETED
    await checkpointer.save_step(workflow_id, step)

    return result
```

### Parallel vs Nested

| Aspect | Parallel Nodes | Nested Graphs |
|--------|---------------|---------------|
| **Storage** | Same workflow | Separate workflow record |
| **ID** | Same `workflow_id` | `parent_id/node_name` |
| **Linking** | None | `child_workflow_id` field |
| **Recovery** | Skip completed steps | Resume child workflow |

**The key distinction:**
- **Parallel** = multiple steps, one workflow
- **Nested** = one step links to child workflow

---

## DBOS vs Built-in Checkpointer

| Capability | Checkpointer | DBOS |
|------------|:------------:|:----:|
| Resume from latest | ✅ Manual (`resume=True`) | ✅ Automatic |
| Get current state | ✅ | ✅ |
| List workflows | ✅ | ✅ |
| Step history | ✅ | ✅ |
| **Automatic crash recovery** | ❌ | ✅ |
| Resume from specific step | ❌ | ✅ (`forkWorkflow`) |
| Workflow forking (time travel) | ❌ | ✅ |
| Durable sleep | ❌ | ✅ |
| Durable queues | ❌ | ✅ |
| Queue concurrency limits | ❌ | ✅ |
| Queue rate limiting | ❌ | ✅ |
| Workflow messaging | ❌ | ✅ (`send`/`recv`) |
| Scheduled workflows (cron) | ❌ | ✅ |
| `.iter()` streaming | ✅ | ❌ (use EventProcessor) |

---

## Advanced DBOS Features

For features beyond HyperNodes primitives, access DBOS directly **without breaking your graphs**.

### Workflow Forking (Time Travel)

Restart a workflow from a specific step:

```python
from dbos import DBOS

# Fork from step 2 to retry with fixed code
DBOS.fork_workflow(
    original_workflow_id="order-123",
    start_step=2,
    new_workflow_id="order-123-retry",
)
```

### Durable Queues

```python
from dbos import Queue

# Get the DBOS workflow that wraps this graph
workflow_fn = runner.get_dbos_workflow(graph)

queue = Queue("processing-queue", concurrency=10)

# Enqueue work
handles = []
for item in items:
    handle = queue.enqueue(workflow_fn, {"item": item})
    handles.append(handle)

# Wait for results
results = [h.get_result() for h in handles]
```

### Scheduled Workflows (Cron)

```python
from dbos import DBOS

workflow_fn = runner.get_dbos_workflow(graph)

@DBOS.scheduled('0 9 * * *')  # Every day at 9am
@DBOS.workflow()
def daily_report(scheduled_time, actual_time):
    workflow_fn({"report_date": scheduled_time.date()})
```

### Durable Sleep

For long delays that survive crashes:

```python
from dbos import DBOS

@DBOS.workflow()
def reminder_workflow(remind_at: datetime, message: str):
    seconds_until = (remind_at - datetime.now()).total_seconds()
    DBOS.sleep(seconds_until)  # Survives any interruption
    send_reminder(message)
```

---

## Runner Selection Guide

```
┌─────────────────────────────────────────────────────────────────────┐
│  What do you need?                                                  │
└───────────┬─────────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────┐     ┌───────────────────────┐
│ Just development?     │ YES │ AsyncRunner()         │
│ No persistence needed │────►│ (no checkpointer)     │
└───────────┬───────────┘     └───────────────────────┘
            │ NO
            ▼
┌───────────────────────┐     ┌───────────────────────┐
│ Need save points?     │ YES │ AsyncRunner +         │
│ Manual resume is OK?  │────►│ SqliteCheckpointer    │
└───────────┬───────────┘     └───────────────────────┘
            │ NO
            ▼
┌───────────────────────┐     ┌───────────────────────┐
│ Need auto recovery?   │ YES │ DBOSAsyncRunner       │
│ Production workload?  │────►│ (SQLite or Postgres)  │
└───────────┬───────────┘     └───────────────────────┘
            │ NO
            ▼
┌───────────────────────┐     ┌───────────────────────┐
│ Need queues,          │ YES │ DBOSAsyncRunner +     │
│ scheduling, etc?      │────►│ DBOS features         │
└───────────────────────┘     └───────────────────────┘
```

---

## Parameter Reference

### AsyncRunner with Checkpointer

```python
result = await runner.run(
    graph,
    inputs={"query": "hello"},
    workflow_id="session-123",  # Required with checkpointer
    resume=True,                # Resume from checkpointed state
)
```

| Parameter | Required | Description |
|-----------|:--------:|-------------|
| `workflow_id` | With checkpointer | Identifies the workflow for state storage |
| `resume` | No | If True, load state from checkpointer and resume |

### DBOSAsyncRunner

```python
result = await runner.run(
    graph,
    inputs={"query": "hello"},
    workflow_id="order-123",    # Required for DBOS
)
```

| Parameter | Required | Description |
|-----------|:--------:|-------------|
| `workflow_id` | Yes | Unique workflow identifier for DBOS durability |

Note: No `resume` parameter — DBOS handles recovery automatically.

---

## Module Structure

```
hypernodes/
├── runners/
│   ├── __init__.py
│   ├── base.py              # BaseRunner, RunnerCapabilities
│   ├── sync.py              # SyncRunner
│   ├── async_.py            # AsyncRunner
│   ├── daft.py              # DaftRunner
│   └── dbos.py              # DBOSAsyncRunner, DBOSSyncRunner
│
├── checkpointers/
│   ├── __init__.py
│   ├── base.py              # Checkpointer ABC
│   ├── sqlite.py            # SqliteCheckpointer
│   └── postgres.py          # PostgresCheckpointer
│
└── ...
```

---

## Installation

```bash
# Core (includes checkpointers)
pip install hypernodes

# With DBOS
pip install hypernodes[dbos]
```

---

## Migration Path

### From Checkpointer to DBOS

```python
# Before: Manual resume with checkpointer
runner = AsyncRunner(checkpointer=SqliteCheckpointer("./dev.db"))
result = await runner.run(graph, inputs={...}, workflow_id="123")
if result.pause:
    # Wait for user input...
    result = await runner.run(
        graph,
        inputs={result.pause.response_param: response},
        workflow_id="123",
        resume=True
    )

# After: Automatic recovery with DBOS
runner = DBOSAsyncRunner()
result = await runner.run(graph, inputs={...}, workflow_id="123")
if result.pause:
    # External system sends response via DBOS.send()
    # Workflow auto-resumes — no runner.run() call needed
```

**What changes:**
- Remove `checkpointer=` parameter
- Remove `resume=True` calls
- Resume via `DBOS.send()` instead of `runner.run()`
- Add DBOS initialization in app startup

**What doesn't change:**
- Graph definition
- Node functions
- InterruptNode usage
- `workflow_id` parameter

---

## Retry Configuration

Transient failures are common. HyperNodes has no built-in retry — just stack a retry decorator on your node.

### Design Principle

**Decorator stacking.** Use [stamina](https://stamina.hynek.me/) or any retry library you prefer.

```python
import stamina
import httpx
from hypernodes import node

@node(output_name="result")
@stamina.retry(on=httpx.HTTPError, attempts=5, timeout=60)
async def fetch(query: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/search?q={query}")
        response.raise_for_status()
        return response.json()
```

No retry params in `@node()`. No HyperNodes retry API. Just decorators.

### Why Stamina?

| Feature | Benefit |
|---------|---------|
| Exponential backoff with jitter | Production-ready defaults (100ms → 45s) |
| Explicit exception types | Forces you to think about what to retry |
| Built-in observability | Prometheus, structlog integration |
| Testing mode | `set_testing()` disables retries in tests |

```bash
pip install stamina
```

### Selective Retry

```python
import httpx
import stamina
from hypernodes import node

def is_retryable(exc: httpx.HTTPStatusError) -> bool:
    """Don't retry 4xx client errors (except 429 rate limit)."""
    return exc.response.status_code >= 500 or exc.response.status_code == 429

@node(output_name="result")
@stamina.retry(on=is_retryable, attempts=5)
async def fetch(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()
```

### Retries and Checkpointing

Retries happen *inside* the node, before checkpointing:

```
Runner calls node
  → Stamina wraps execution
    → attempt 1: fails
    → attempt 2: fails
    → attempt 3: succeeds ✓
  → Stamina returns result
→ Checkpointer saves result
```

The checkpointer only sees the final result. Same behavior with any runner.

**On crash recovery:** Retry count resets (retry state is in-memory, not persisted).

### Retry vs Recovery

| Concept | Scope | Trigger | Handled By |
|---------|-------|---------|------------|
| **Retry** | Single node | Transient exception | stamina (or any retry lib) |
| **Resume** | Entire workflow | User calls `resume()` | Checkpointer |
| **Recovery** | Entire workflow | Process restart | DBOS automatic |

### Best Practices

1. **Be explicit about exceptions** — Never retry on bare `Exception`
   ```python
   # ❌ Bad
   @stamina.retry(on=Exception, attempts=5)

   # ✅ Good
   @stamina.retry(on=(httpx.HTTPError, asyncio.TimeoutError), attempts=5)
   ```

2. **Don't retry forever** — Always set `attempts` and/or `timeout`

3. **Consider idempotency** — Retried operations should be safe to repeat

4. **Use testing mode** — Disable retries in tests
   ```python
   def test_my_node():
       with stamina.set_testing(attempts=1):
           result = my_node("input")
   ```

---

## Summary

| Layer | Responsibility | User Imports |
|-------|----------------|--------------|
| **Graph** | Structure, nodes, routing | `hypernodes` |
| **Runner** | Execution, event dispatch | `hypernodes.runners` |
| **Checkpointer** | Manual persistence | `hypernodes.checkpointers` |
| **Retries** | Transient failure handling | `stamina` (or any retry lib) |
| **DBOS (optional)** | Automatic durability, queues, scheduling | `dbos` |

**The principle:** Graph code stays pure. Durability is a runner concern. Retries are decorator stacking — no HyperNodes-specific retry API.
