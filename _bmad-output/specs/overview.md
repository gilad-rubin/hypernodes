# Hypernodes v0.5 - Specification Overview

## Purpose

This document provides context for implementing Hypernodes v0.5 - a graph-native execution system that supports cycles, multi-turn interactions, and complex control flow while maintaining pure, portable functions.

## The Journey: From Hierarchical DAGs to Reactive Graphs

### Where It Started (v0.1-0.4)

HyperNodes began as an answer to existing DAG frameworks like Hamilton and Pipefunc. The key innovation: **hierarchical composition** - pipelines are nodes that can be nested infinitely.

```python
# The original vision: pipelines as composable building blocks
inner = Pipeline(nodes=[clean, tokenize])
outer = Pipeline(nodes=[fetch, inner.as_node(), analyze])
```

This enabled:
- ✅ Reusable pipeline components
- ✅ Modular testing (test small pipelines, compose into large ones)
- ✅ Visual hierarchy (expand/collapse nested pipelines)
- ✅ "Think singular, scale with map" - write for one item, map over collections

### Where It Hit the Wall

The DAG constraint (no cycles) works beautifully for:
- ETL workflows
- Single-pass ML inference  
- Batch data processing

But **fundamentally breaks** for modern AI workflows:

| Use Case | Why DAGs Fail |
|----------|---------------|
| **Multi-turn RAG** | User asks → retrieve → answer → *user follows up* → retrieve **more** → refine (needs to loop back) |
| **Agentic workflows** | LLM decides next action, may need to retry/refine until satisfied |
| **Iterative refinement** | Generate → evaluate → if not good enough → generate again |
| **Conversational AI** | Maintain conversation state, allow user to steer at any point |

### The Inciting Incident

Building a multi-turn RAG system where:
1. User asks a question
2. System retrieves documents and generates answer
3. User says "can you explain X in more detail?"
4. System needs to **retrieve more documents** using conversation context
5. System refines the answer

Step 4 is **impossible** in a DAG - can't loop back to retrieval. The entire architecture assumes single-pass execution.

### Why Not LangGraph or Pydantic-Graph?

Both solve cycles, but both require:
- Explicit state objects that functions must read from and write to
- Manual edge wiring
- Framework-coupled functions that are not portable
- Reducer annotations for append semantics
- Field names repeated in state class, reads, writes, and edges (not DRY)

**The frustration - we want to write this:**
```python
@node(output_name="messages")
def add_response(messages: list, response: str) -> list:
    return messages + [response]
```

**Not this:**
```python
def add_response(state: AgentState) -> dict:
    messages = state["messages"]  # Read from state
    response = state["response"]
    return {"messages": messages + [response]}  # Write to state
```

## Key Differentiators

| Aspect | LangGraph / Pydantic-Graph | HyperNodes |
|--------|---------------------------|------------|
| **State definition** | Static `TypedDict` or Pydantic model required | No state class - just function signatures |
| **Graph construction** | Edges defined at class definition time | Build graphs dynamically at runtime |
| **Validation timing** | Compile time (static types) | Build time (`Graph()` construction) |
| **Type hints** | Mandatory | Optional (opt-in for extra checks) |
| **Function portability** | Framework-coupled | Pure functions, testable without imports |

## The Solution: Dynamic Graphs with Build-Time Validation

HyperNodes 0.5 introduces **fully dynamic graph construction** with validation at build time (when `Graph()` is called), not compile time.

```python
# LangGraph - static, tied to schema
class AgentState(TypedDict):
    messages: list[str]  # Must know fields at definition time
graph = StateGraph(AgentState)

# HyperNodes - fully dynamic
nodes = [create_tool_node(t) for t in available_tools]  # Built at runtime!
graph = Graph(nodes=nodes)  # Validation happens here
```

### Why Implicit Edges by String Are Fine in the AI Era

LLMs already work in a write-then-validate loop - they write code, then get compiler/runtime feedback to fix issues. **Build-time validation = compiler feedback**.

```
Traditional: Write code → Compiler error → Fix → Repeat
HyperNodes:  Write code → Graph() error → Fix → Repeat
```

Both catch errors before runtime. The difference is *when* validation happens (compile time vs build time), not *whether* it happens.

## Core Architectural Changes from v0.4

1. **`Graph` replaces `Pipeline`** - Pure definition, constructed from list of nodes
2. **`Runner` / `AsyncRunner`** - Execution separated from definition; runners own cache and callbacks
3. **Reactive dataflow with versioning** - Values have versions, staleness drives execution
4. **Unified execution algorithm** - Same code handles DAGs, branches, AND cycles

## What This Enables

- ✅ Multi-turn conversational RAG
- ✅ Agentic workflows with loops
- ✅ Retry patterns
- ✅ Iterative refinement
- ✅ Message accumulators that don't infinite loop
- ✅ Human-in-the-loop with pause/resume (`InterruptNode`)
- ✅ Token-by-token streaming (`.iter()` API)
- ✅ Event streaming for observability
- ✅ Checkpointing and resume
- ✅ Distributed batch processing (DaftRunner for DAG-only graphs)

## Implementation Priority

1. **Core Graph + Runner** - Must work first
2. **@route decorator** - Enables cycles
3. **Staleness detection** - Prevents infinite loops
4. **AsyncRunner + streaming** - Modern LLM APIs need this
5. **InterruptNode** - Human-in-the-loop
6. **DaftRunner** - Distributed (DAG-only)
