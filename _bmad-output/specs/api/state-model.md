# State Model

**In HyperNodes, there is no separate "state" to define. Your node outputs are the state.**

---

## The Question

> "Where do I define state in HyperNodes?"

If you're coming from LangGraph or similar frameworks, you might expect to define an explicit state schema:

```python
# LangGraph - explicit state schema
class State(TypedDict):
    messages: Annotated[list, add]  # Reducer for combining
    query: str
    answer: str

graph = StateGraph(State)
```

**HyperNodes doesn't work this way.** There's no `State` class to define.

---

## Outputs ARE State

In HyperNodes, state emerges from your nodes' outputs:

```python
@node(output_name="messages")
def chat(messages: list, user_input: str) -> list:
    return messages + [{"role": "user", "content": user_input}]

@node(output_name="answer")
def generate(messages: list) -> str:
    return llm.chat(messages)

graph = Graph(nodes=[chat, generate])
```

- `messages` is an output that flows between nodes
- `answer` is an output returned to the user
- Together, they form the graph's "state"

---

## Why No Explicit State?

Frameworks require explicit state for specific reasons. HyperNodes addresses each differently:

### 1. Dataflow (Values Between Nodes)

**LangGraph:** State channels carry values between nodes.

**HyperNodes:** Outputs flow via edge inference. If node A produces `embedding` and node B takes `embedding` as input, they're connected automatically.

```python
@node(output_name="embedding")
def embed(text: str) -> list[float]: ...

@node(output_name="docs")
def retrieve(embedding: list[float]) -> list[str]: ...
# Edge inferred: embed → retrieve
```

### 2. Persistence (Surviving Crashes)

**LangGraph:** Everything in the state schema is checkpointed.

**HyperNodes:** Control what's checkpointed with the `persist` parameter.

```python
graph = Graph(
    nodes=[embed, retrieve, generate],
    persist=["messages", "answer"],  # Only these are checkpointed
)
```

### 3. Memory (Across Conversation Turns)

**LangGraph:** Memory is part of the state schema, persisted to threads.

**HyperNodes:** Pass conversation history as input, return updated history as output.

```python
@node(output_name="messages")
def chat(messages: list, user_input: str) -> list:
    response = llm.chat(messages + [{"role": "user", "content": user_input}])
    return messages + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response},
    ]

# Run with previous messages
result = runner.run(graph, inputs={
    "messages": previous_messages,
    "user_input": "Hello!",
})
```

### 4. Human-in-the-Loop (What Human Sees/Modifies)

**LangGraph:** Human edits state directly via `update_state()`.

**HyperNodes:** Use `InterruptNode` with explicit parameters.

```python
approval = InterruptNode(
    name="approval",
    input_param="draft",        # What human sees
    response_param="decision",  # What human provides
)
```

### 5. Reducers (Combining Updates)

**LangGraph:** Reducers like `Annotated[list, add]` combine multiple updates to the same key.

**HyperNodes:** Not needed. Each node produces distinctly-named outputs. No conflicts to resolve.

---

## The `persist` Parameter

The `persist` parameter is how you declare "these outputs matter for recovery":

### Graph-Level (Allowlist)

```python
graph = Graph(
    nodes=[embed, retrieve, generate],
    persist=["messages", "answer"],  # Only these are checkpointed
)
```

### Node-Level (Override)

```python
@node(output_name="embedding", persist=False)  # Never checkpoint
def embed(text: str) -> list[float]:
    return model.embed(text)

@node(output_name="answer", persist=True)  # Always checkpoint
def generate(docs: list[str]) -> str:
    return llm.generate(docs)
```

### Semantics

| `persist` | On Crash/Resume | Storage |
|-----------|-----------------|---------|
| `True` (default) | Load from checkpoint | Saved to DB |
| `False` | Re-execute node | Not saved |

---

## Three Layers of "State"

```
┌─────────────────────────────────────────────────────────────────┐
│  Runtime State (GraphState)                                      │
│                                                                  │
│  All outputs from all nodes, tracked with version numbers.      │
│  Used internally for staleness detection and cycle support.     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Persisted State (Checkpoints)                                  │
│                                                                  │
│  Only outputs where persist=True.                               │
│  Survives crashes, enables resume.                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Returned State (RunResult)                                     │
│                                                                  │
│  Outputs returned to the user.                                  │
│  Can be filtered with `select` parameter.                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Comparison with LangGraph

| Aspect | LangGraph | HyperNodes |
|--------|-----------|------------|
| **Define state** | Explicit `TypedDict` | Implicit from outputs |
| **What's persisted** | Everything in schema | Controlled by `persist` |
| **Reducers** | Required for shared keys | Not needed |
| **Memory** | Built into state | Pass as input/output |
| **Human edits** | `update_state()` | `InterruptNode` |
| **Philosophy** | "State is central" | "Outputs flow, persist what matters" |

---

## FAQ

### "How do I accumulate values across iterations (like a reducer)?"

Use a node that takes the previous value as input and returns the updated value:

```python
@node(output_name="messages")
def accumulate(messages: list, new_message: str) -> list:
    return messages + [new_message]
```

In cycles, HyperNodes tracks versions to know when to re-execute.

### "How do I share state across multiple workflows?"

HyperNodes' checkpointer handles state within a single workflow. For cross-workflow memory (like user preferences across conversations), use an external store (database, Redis, etc.) and pass values as inputs.

### "What if two nodes produce the same output name?"

This is a build-time error unless the nodes are mutually exclusive (via routing gates). HyperNodes validates this when constructing the graph.

### "Can I update state from outside the graph?"

For human-in-the-loop, use `InterruptNode`. The interrupt surfaces a value, waits for a response, and the response becomes an output that flows to downstream nodes.

---

## Summary

- **No explicit state schema** - Outputs are inferred from nodes
- **Persistence is configurable** - Use `persist` to control what's checkpointed
- **Memory is explicit** - Pass as input, return as output
- **Reducers not needed** - Each node has distinct outputs

**The mental model:** Nodes are pure functions. Outputs flow between them. Persistence is orthogonal - you choose what survives a crash.
