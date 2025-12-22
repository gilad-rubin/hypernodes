---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
inputDocuments:
  - '_bmad-output/index.md'
  - '_bmad-output/architecture.md'
  - '_bmad-output/project-overview.md'
  - '_bmad-output/development-guide.md'
  - '_bmad-output/source-tree-analysis.md'
  - 'guides/runner_api_design.md'
  - 'guides/hypernodes_v2_design.md'
  - 'guides/graph_implementation_guide.md'
  - 'guides/graph_edge_cases.md'
  - 'guides/async_execution_design.md'
  - 'guides/nested_graph_results.md'
  - 'guides/node_chaining.md'
  - 'guides/optional_outputs_design.md'
documentCounts:
  briefs: 0
  research: 0
  brainstorming: 0
  projectDocs: 5
  designGuides: 8
workflowType: 'prd'
lastStep: 11
project_name: 'hypernodes'
user_name: 'Giladrubin'
date: '2025-12-22'
---

# Product Requirements Document - hypernodes

**Author:** Giladrubin
**Date:** 2025-12-22

## Executive Summary

HyperNodes is evolving from a DAG-only pipeline framework to a **graph-native execution system** that supports cycles, multi-turn interactions, and complex control flow - all while maintaining the framework's core philosophy of pure, portable functions.

**This is a personal infrastructure project** built to solve real problems in my own work, documented thoroughly, and shared in case the approach resonates with others building similar systems.

### The Journey: From Hierarchical DAGs to Reactive Graphs

**Where it started (v0.1-0.4):**

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

**Where it hit the wall:**

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

**The inciting incident:**

Building a multi-turn RAG system where:
1. User asks a question
2. System retrieves documents and generates answer
3. User says "can you explain X in more detail?"
4. System needs to **retrieve more documents** using conversation context
5. System refines the answer

Step 4 is **impossible** in a DAG - can't loop back to retrieval. The entire architecture assumes single-pass execution.

I looked at LangGraph and Pydantic-Graph as alternatives. Both solve cycles, but both require:
- Explicit state objects that functions must read from and write to
- Manual edge wiring
- Framework-coupled functions that can't be tested standalone
- Reducer annotations for append semantics
- Field names repeated in state class, reads, writes, and edges

**The frustration:**

I wanted to write this:
```python
@node(output_name="messages")
def add_response(messages: list, response: str) -> list:
    return messages + [response]
```

Not this:
```python
def add_response(state: AgentState) -> dict:
    messages = state["messages"]  # Read from state
    response = state["response"]
    return {"messages": messages + [response]}  # Write to state
```

**The realization:**

Reading graph theory papers and NetworkX documentation, I realized: **you don't need explicit state objects if you track versions and compute staleness**. The graph structure + versioned values + standard algorithms can handle cycles without framework coupling.

### The Solution: NetworkX-Native Reactive Dataflow Graphs

V2 restructures the internals to be **graph-theory native** using NetworkX as the foundation:

**Core architectural changes:**
1. **`Graph` replaces `Pipeline`** - Pure definition wrapping `nx.DiGraph` with node/edge attributes
2. **`Runner` / `AsyncRunner`** - Execution separated from definition; runners own cache and callbacks
3. **Reactive dataflow with versioning** - Values have versions, staleness drives execution
4. **Unified execution algorithm** - Same code handles DAGs, branches, AND cycles
5. **String-based routing** - `@route` decorator with Literal types, validated at build time
6. **GraphState** - Tracks versioned values, execution history, gate state
7. **Standard algorithms** - Leverage NetworkX for cycles, reachability, topological sort

**Example - Multi-turn RAG becomes possible:**

```python
from hypernodes import Graph, node, route, END
from typing import Literal

@node(output_name="docs")
def retrieve(query: str, messages: list) -> list:
    return vector_db.search(query, context=messages)

@node(output_name="response")
async def generate(docs: list, messages: list, llm) -> str:
    async for chunk in llm.stream(...):
        yield chunk

@node(output_name="messages")  # Accumulator pattern
def add_response(messages: list, response: str) -> list:
    return messages + [{"role": "assistant", "content": response}]

RouteDecision = Literal["retrieve", END]

@route  # Not @gate - validates "retrieve" exists at build time
def should_continue(messages: list) -> RouteDecision:
    if len(messages) > 10 or detect_done(messages[-1]):
        return END
    return "retrieve"  # Loops back! Creates cycle

graph = Graph(nodes=[retrieve, generate, add_response, should_continue])

# Runner pattern: Graph is pure definition, Runner handles execution
runner = AsyncRunner(cache=DiskCache("./cache"))
result = await runner.run(graph, inputs={"query": "What is RAG?", "messages": [], "llm": my_llm})
```

**What the framework handles automatically:**
- ✅ Cycle execution (retrieve can run multiple times)
- ✅ Staleness detection (knows when to re-run nodes)
- ✅ Sole producer rule (prevents infinite loops in accumulators)
- ✅ Gate validation (fails fast if "retrieve" doesn't exist)
- ✅ Version tracking (each message update increments version)

### Design Philosophy: Solving My Constraints

This rewrite is driven by the specific constraints of building multi-turn RAG, informed by what frustrated me in existing frameworks.

**What I need:**
- ✅ **Cycles** - Multi-turn conversations that loop back
- ✅ **Pure functions** - Easy to test in notebooks, reusable outside framework
- ✅ **Hierarchical composition** - Keep the pipeline-as-node pattern that works
- ✅ **Implicit edges** - Dependencies visible in signatures, not buried in code
- ✅ **Standard algorithms** - Use NetworkX instead of reinventing graph theory
- ✅ **Single execution model** - One algorithm handles DAGs, branches, AND cycles

**What I explicitly don't need (yet):**
- Maximum performance (correctness > speed for Phase 1)
- Enterprise features (RBAC, audit logs, compliance)
- Backward compatibility (no users to break, clean slate)
- Market positioning (solving my problem first, seeing if others care second)

**Comparison to state object frameworks (why I'm not using LangGraph):**

| Aspect | State Object Frameworks | HyperNodes Graph (What I Want) |
|--------|------------------------|--------------------------------|
| **Function coupling** | Functions read/write explicit state object | Pure input → output, framework handles state |
| **Dependencies** | Hidden in function bodies | Explicit in signatures |
| **Edge wiring** | Manual edge definitions | Inferred from signatures |
| **Portability** | Framework-coupled, can't test standalone | Use anywhere, test with plain function calls |
| **Accumulators** | Need reducer annotations | Just return a list |
| **DRY** | Field names repeated 4+ places | Names in signatures once |

**The philosophy in one sentence:**

> **Pure functions + implicit edges + reactive dataflow + NetworkX = powerful cyclic graphs without the complexity**

**The approach:**

Build it for myself, document it thoroughly, publish it as I go. If the design resonates with others building agentic systems, great. If not, I still have infrastructure that solves my problem cleanly.

This is **infrastructure-as-learning** - understanding reactive dataflow, graph theory, and execution semantics deeply enough to build something elegant.

### Technical Architecture: NetworkX-Native Reactive Dataflow

**Core architectural changes from Pipeline (v0.4.8) to Graph (v0.5.0):**

1. **`Graph` class wraps `nx.DiGraph`** with node/edge attributes
   - Nodes: hypernode object, is_gate flag, node metadata
   - Edges: edge_type (data vs control), value names, gate decisions

2. **`GraphState` with versioned values**
   - Every value has a version number (increments on update)
   - Staleness detection: "has input changed since I last ran?"
   - Sole producer rule: accumulators don't re-trigger from own output

3. **Unified execution algorithm**
   - Same code handles DAGs, branches, AND cycles
   - Reactive: nodes execute when inputs are stale
   - Gate-driven: conditional routing via control edges

4. **String-based routing with build-time validation**
   - `@route` decorator with `Literal` types
   - Targets validated at Graph initialization (fail fast)
   - `END` sentinel for termination

5. **Standard algorithms from NetworkX**
   - Cycle detection: `nx.is_directed_acyclic_graph()`
   - Reachability: `nx.has_path()`
   - Ancestors/descendants: `nx.ancestors()`, `nx.descendants()`
   - Topological sort: `nx.topological_sort()` (for DAG subgraphs)

6. **Runner pattern (separates definition from execution)**
   - `Graph` is pure structure - no `run()` method, no cache, no callbacks
   - `Runner` / `AsyncRunner` handles execution with runtime config
   - Same graph can be executed with different runners (sync, async, distributed)
   - Runners own: cache, callbacks, execution strategy

```python
from hypernodes import Graph, node, Runner, AsyncRunner, DiskCache

# Graph = pure definition
graph = Graph(nodes=[embed, retrieve, generate])

# Runner = execution runtime
runner = Runner(cache=DiskCache("./cache"))
result = runner.run(graph, inputs={"query": "hello"})

# AsyncRunner for async nodes and streaming
async_runner = AsyncRunner(cache=DiskCache("./cache"))
result = await async_runner.run(graph, inputs={"query": "hello"})

# Same graph, different runners
results = runner.map(graph, inputs={"query": queries}, map_over="query")
```

7. **InterruptNode for human-in-the-loop**
   - Declarative pause points in the graph
   - `input_param`: what to surface to user
   - `response_param`: where to write user's response
   - Framework provides plumbing, user defines prompt/response types

```python
from hypernodes import InterruptNode

approval = InterruptNode(
    name="approval",
    input_param="approval_prompt",     # Read prompt from here
    response_param="user_decision",    # Write response to here
    response_type=ApprovalResponse,    # Optional validation
)

graph = Graph(nodes=[create_prompt, approval, route_decision, finalize])
```

**Entrypoint Requirement:**

Entrypoints provide validation context and enable better error messages:

- **DAGs (no cycles):** Entrypoint is **optional** (execution order is unambiguous via topological sort)
- **Cyclic Graphs:** Entrypoint is **mandatory** (forces clarity about "where does this start?")

```python
# ❌ Fails at build time
graph = Graph(nodes=[retrieve, generate, route_back])
# ConfigError: Cyclic graph detected (route_back → retrieve).
#              Cycles require entrypoint: Graph(..., entrypoint="retrieve")

# ✅ Clear and validated
graph = Graph(
    nodes=[retrieve, generate, add_response, route_back],
    entrypoint="retrieve",  # Mandatory for cyclic graphs
)
```

**What entrypoint provides:**

**Build-time:**
- ✅ Path distance analysis (compute entrypoint → all nodes)
- ✅ Sequential producer validation (different distances = sequential, same distance = parallel)
- ✅ Reachability validation (all nodes reachable from entrypoint)
- ✅ Termination validation (can reach END from any cycle)
- ✅ Auto-suggests entrypoint in error (detects likely start nodes)

**Runtime:**
- ✅ Enhanced error messages (explains normal flow from entrypoint)
- ✅ Flow context in conflicts ("normal path is X→Y→Z, you skipped to Z")
- ✅ NO enforcement (still executes whatever is ready based on inputs)
- ✅ Silent operation (no warnings unless actual error)

**Checkpoints override entrypoint:**
```python
# Entrypoint ignored when resuming - checkpoint determines position
runner.run(graph, checkpoint=saved_state, inputs={...})
```

**Example - Build-time validation:**
```
Graph analysis (from entrypoint 'add_user'):
  add_user (distance 0) → generate (distance 1) → add_assistant (distance 2) → route (distance 3)
  Cycle: route → retrieve (loops back)
  
✓ Sequential producers for 'messages': add_user, add_assistant
  Different distances (0 vs 2) → deterministic execution order
  
✓ Termination possible: route can return END
```

**Example - Enhanced runtime error:**
```
ParallelProducersError: Value 'messages' has multiple producers ready:
  - add_user (needs: messages, user_input)
  - add_assistant (needs: messages, response)

FLOW ANALYSIS (from entrypoint 'add_user'):
  Normal path: add_user (0) → generate (1) → add_assistant (2)
  Your inputs made both ready simultaneously.

Options:
  1. Provide only 'user_input' (start from entrypoint)
  2. Resume from checkpoint with conversation state
  3. Add dependency between producers
```

**Validation Strategy: Fail Fast at Every Stage**

**Build-Time Validation (Graph Initialization):**

When you create a `Graph`, these errors are caught immediately:
- ✅ All `@route` targets reference existing nodes
- ✅ Mutually exclusive branches can share output names (validated transitively)
- ✅ Gates that can activate together don't produce conflicting outputs
- ✅ Cycles have valid termination paths (gates with `END`)
- ✅ No structural impossibilities (self-loops without gates, etc.)
- ✅ **With entrypoint:** Sequential producers validated by path distance analysis

**Runtime Validation (Before Execution Starts):**

When you call `runner.run(graph, inputs={...})`, these errors are caught BEFORE any node executes:
- ✅ Input-dependent conflicts (user provided inputs that make parallel producers ready)
- ✅ All required inputs are available
- ✅ No dynamic conflicts in initial ready set

**During Execution:**

After each node completes:
- ✅ Check next ready set for conflicts before executing
- ✅ Validate gate decisions reference valid targets

**Error Message Pattern:**

All errors include:
- What went wrong (clear error type)
- Why it went wrong (which inputs/nodes caused it)
- How to fix it (3 concrete options)

Example:
```
ParallelProducersError: Value 'messages' has multiple producers ready simultaneously:
  - add_user (needs: messages, user_input)
  - add_assistant (needs: messages, response)

This conflict was caused by providing both 'user_input' and 'response' as inputs.

Options to fix:
  1. Remove 'response' from inputs (run normal user→assistant flow)
  2. Remove 'user_input' from inputs (resume from assistant response)
  3. Add dependency: make add_assistant depend on add_user output
```

**Cache + Versioning Interaction:**

- Versions track execution history for staleness detection
- Cache signatures use actual input VALUES, not version numbers
- Multi-turn loops: Each iteration has different messages → different cache key → correctly recomputes
- Deterministic: Same inputs always produce same signature, regardless of iteration count

**Streaming Support:**

Phase 1 includes **generator handling** (internal accumulation):
- ✅ Detect if node returns generator (via `inspect.isgenerator()`)
- ✅ Accumulate chunks automatically
- ✅ Store final value in state
- ❌ NO streaming events to user (`.iter()` API is Phase 2)

Why: Modern LLM APIs return generators. Framework must handle them, but doesn't expose token-by-token streaming until Phase 2.

**Three-Layer Architecture (Phase 2+):**

The framework separates three distinct concerns that can be layered independently:

| Layer | Purpose | Example | Protocol |
|-------|---------|---------|----------|
| **UI Protocol** | Real-time streaming to frontends | AG-UI compatible streaming | Events → WebSocket |
| **Observability** | Logging, tracing, analytics | Langfuse, Logfire integration | Events → Callback |
| **Durability** | Checkpoint persistence | Redis, PostgreSQL, SQLite | Checkpointer interface |

```
┌─────────────────────────────────────────┐
│              User Frontend              │
│         (AG-UI / Custom / CLI)          │
└───────────────────┬─────────────────────┘
                    │ Streaming Events
┌───────────────────▼─────────────────────┐
│           UI Protocol Layer             │
│     (transforms events → AG-UI SSE)     │
└───────────────────┬─────────────────────┘
                    │
┌───────────────────▼─────────────────────┐
│         Observability Layer             │
│   (Langfuse spans, Logfire traces)      │
└───────────────────┬─────────────────────┘
                    │
┌───────────────────▼─────────────────────┐
│           Core Execution                │
│      (Runner + Graph + State)           │
└───────────────────┬─────────────────────┘
                    │
┌───────────────────▼─────────────────────┐
│          Durability Layer               │
│   (Checkpointer: Redis/SQL/Memory)      │
└─────────────────────────────────────────┘
```

**Key principle:** Layers consume a unified event stream. The core produces events; layers subscribe to what they need.

**Event Types:**
- `RunStartEvent`, `RunEndEvent` - Execution lifecycle
- `NodeStartEvent`, `NodeEndEvent` - Node execution
- `StreamingChunkEvent` - Token-by-token streaming
- `InterruptEvent` - Human-in-the-loop pauses
- `CacheHitEvent` - Cache usage

**Identity Model:**

| ID | Scope | Who Creates | Purpose |
|----|-------|-------------|---------|
| `session_id` | User conversation | User | Group related runs (multi-turn conversation) |
| `run_id` | Single execution | Framework | Identify specific graph execution |

```python
# session_id groups runs; run_id is auto-generated
result = await runner.run(graph, inputs={...}, session_id="conversation-123")
# result.run_id → "run-abc-456" (auto)
```

### Success Criteria

**Personal Success (Primary):**

This rewrite succeeds if:
- ✅ My multi-turn RAG code is cleaner than my LangGraph prototype
- ✅ I can test nodes in Jupyter without framework boilerplate
- ✅ The caching actually speeds up iteration (not fighting invalidation)
- ✅ I understand the execution model well enough to debug it
- ✅ Adding new features feels natural, not hacky
- ✅ I learned graph theory and reactive dataflow deeply

**Community Validation (Bonus):**

Signs that the approach resonates:
- 🎁 Someone stars the repo because the philosophy clicks
- 🎁 Someone opens an issue with a real use case
- 🎁 Someone contributes a PR improving something
- 🎁 Discussion on HN/Reddit validates "pure functions > state objects"

**Even if community validation never happens, success = solves my problem cleanly.**

### What This Enables

**Immediate (Phase 1 MVP):**
- ✅ Multi-turn conversational RAG (my use case)
- ✅ Agentic workflows with loops
- ✅ Retry patterns
- ✅ Iterative refinement
- ✅ Message accumulators that don't infinite loop

**Near-term (Phase 2 - Polish):**
- ✅ Human-in-the-loop with pause/resume (`InterruptNode`)
- ✅ Token-by-token streaming (`.iter()` API)
- ✅ Event streaming for observability
- ✅ Basic checkpointing

**Future (Phase 3 - Community-Driven):**
- 🤔 Distributed execution (DaftEngine for Graph) - if needed
- 🤔 Durable workflows - if use case emerges
- 🤔 Whatever the community asks for - if there is one

## Success Criteria

This section defines measurable, phase-specific criteria for determining when each milestone is complete and successful.

### Phase 1: Core Graph Architecture (MVP)

**Goal:** Validate the reactive dataflow model with a real multi-turn RAG implementation.

#### Must-Have Acceptance Criteria

| Criterion | Definition of Done | Validation Method |
|-----------|-------------------|-------------------|
| **Cyclic execution works** | Multi-turn RAG runs 3+ loops without infinite loops | Integration test with mock LLM |
| **Pure functions preserved** | All nodes testable with plain `assert node_func(x) == y` | Unit tests without framework imports |
| **Staleness detection correct** | Nodes re-execute only when inputs change | State inspection tests |
| **Sole producer rule enforced** | Accumulators don't trigger from own output | Cycle test with message accumulator |
| **Build-time validation works** | Invalid graphs fail at `Graph()` construction | Negative test cases |
| **Route targets validated** | `@route` with invalid target fails fast | Build-time error tests |
| **Generator handling works** | Async generators accumulate correctly | Streaming node tests |
| **Cache signatures stable** | Same inputs → same key across iterations | Signature determinism tests |

#### Quality Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| **Test coverage** | ≥80% line coverage on core modules | `pytest --cov` |
| **All tests pass** | Zero failures | CI/local test run |
| **No regressions in DAG mode** | Existing patterns still work | Compatibility tests |
| **Error messages actionable** | All errors include "what", "why", "how to fix" | Manual review |

#### Personal Validation

- [ ] Multi-turn RAG is cleaner than my LangGraph prototype
- [ ] I can test nodes in Jupyter without any framework boilerplate
- [ ] Debugging feels natural (I understand what's happening)
- [ ] Adding a new feature doesn't require touching 5+ files

### Phase 2: Polish & Developer Experience

**Goal:** Production-ready quality-of-life features for real-world usage.

#### Must-Have Acceptance Criteria

| Criterion | Definition of Done | Validation Method |
|-----------|-------------------|-------------------|
| **`.iter()` streaming works** | Token-by-token streaming with event types | Integration test with SSE/websocket |
| **`InterruptNode` pauses** | Execution pauses, state persists, resumes correctly | Human-in-loop test |
| **Checkpointing works** | Save state, kill process, resume from checkpoint | Persistence test |
| **Event streaming works** | All lifecycle events emitted with timing | Callback inspection |
| **Visualization updated** | Graph viz shows cycles, gates, active node | Manual + snapshot tests |

#### Quality Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| **Streaming latency** | First token < 100ms after LLM starts | Timing tests |
| **Checkpoint size** | < 10MB for typical conversation state | Size measurement |
| **Resume correctness** | 100% identical results after resume | Determinism tests |

### Phase 3: Community-Driven (Future)

**Goal:** Features driven by actual user needs (if community forms).

#### Potential Features (Not Committed)

These are tracked as "might do" based on community interest:

| Feature | Trigger Condition |
|---------|-------------------|
| **Distributed execution** | 3+ users request parallel execution for large batches |
| **Durable workflows** | Real use case with multi-day execution needs |
| **Web UI for debugging** | Demand for visual debugging beyond Jupyter |
| **Multi-backend cache** | Need for Redis/S3 cache adapters |

#### Success Signal

- GitHub stars > 100 (indicates resonance)
- 3+ real issues from different users
- 1+ external PR merged
- Discussion thread validates "pure functions > state objects"

### Non-Goals & Anti-Patterns

**Explicitly NOT optimizing for:**

| Non-Goal | Rationale |
|----------|-----------|
| Maximum performance | Correctness first, optimize when bottleneck proven |
| Enterprise features | No RBAC, audit logs, compliance (personal project) |
| Backward compatibility | No users to break, clean slate is faster |
| Framework lock-in | Functions must remain portable |
| Magic behavior | Explicit > implicit for debugging |

**Anti-patterns to avoid:**

| Anti-Pattern | Why It's Bad | Alternative |
|--------------|--------------|-------------|
| State object coupling | Tests require framework setup | Pure input/output functions |
| Hidden dependencies | Can't reason about execution order | Explicit in signatures |
| Implicit reducers | Append semantics unclear | Return complete list |
| Silent failures | Errors discovered too late | Fail fast at build time |

### Measuring Success Over Time

**Weekly Check-in Questions:**

1. Did I hit any frustrating edges this week?
2. Is the codebase getting simpler or more complex?
3. Can I still explain the execution model in one paragraph?
4. Did I need to look at the code to understand behavior, or were the abstractions clear?

**Monthly Review:**

- Lines of code trend (should be stable or decreasing per feature)
- Test-to-code ratio (should stay ≥0.7)
- Time to implement a new feature (should decrease)
- Time spent debugging vs building (should shift toward building)

**Definition of Done for V2:**

The v0.5.0 release is "done" when:
1. All Phase 1 acceptance criteria pass
2. Multi-turn RAG example runs end-to-end
3. Migration guide written (Pipeline → Graph patterns)
4. Architecture decision records exist for all major choices
5. I would choose this over LangGraph for my next project

## Project Classification

**Technical Type:** Python Library (Graph Execution Framework)  
**Domain:** AI/ML Workflows, Agentic Systems  
**Complexity:** High  
**Project Context:** Personal Infrastructure / Research Project  
**Status:** Brownfield → Greenfield (v0.5.0 architectural rewrite)

### Classification Details

**Project Type:** Personal Developer Tool → Open Source (Maybe)
- Personal infrastructure solving a real problem (multi-turn RAG)
- Zero-dependency core with optional integrations
- Published incrementally as development progresses
- Community engagement is a bonus, not a requirement

**Project Nature:**
- **Infrastructure-as-learning:** Deep dive into graph theory, reactive dataflow, execution semantics
- **Opinionated:** Strong stance on pure functions vs state objects
- **Practical:** Must solve the real multi-turn RAG problem, not just theory
- **Research-flavored:** Reading papers, implementing patterns, documenting findings

**Domain Complexity:** Scientific/Research (high complexity)
- **Key Concerns:** 
  - Graph theory correctness (cycle detection, staleness, mutual exclusivity)
  - Execution semantics (versioning, conflict resolution, termination)
  - Portability (pure functions, testability, no framework coupling)
  - Caching correctness (signature computation with cycles)

**Primary User (Phase 1):**
- Me (building multi-turn RAG)
- Future me (debugging, iterating, extending)

**Potential Secondary Users (Phase 2+):**
- Developers frustrated with state object frameworks
- Researchers building agentic systems
- ML engineers who value function purity
- Anyone building conversational AI / multi-turn workflows

**Migration Strategy:**
- **Clean break:** v0.5.0 introduces `Graph`, marks `Pipeline` as legacy
- **No backward compatibility needed:** No existing external users
- **Documentation focus:** 
  - Migration guide (Pipeline → Graph patterns)
  - Theory documentation (why reactive dataflow works)
  - Architecture decisions (ADRs for future reference)
- **Code preservation:** Archive `Pipeline` code in `src/hypernodes/old/` for reference

### Architectural Decisions

**Why NetworkX?**
- Battle-tested graph algorithms (don't reinvent cycle detection)
- Explicit graph modeling makes visualization natural
- Rich ecosystem (compatibility with other graph tools)
- Clear separation: NetworkX does graph theory, we do execution semantics

**Why Clean Break vs Gradual?**
- No user base to support (can move fast)
- Fundamental paradigm shift (DAG → reactive graphs with cycles)
- Simpler mental model (one execution system, not two)
- Learning opportunity (understand the model deeply, not just bolt-on features)

**Why Phased Implementation?**
- **Phase 1:** Validates core model with real use case (multi-turn RAG)
- **Phase 2:** Adds quality-of-life (streaming, human-in-loop)
- **Phase 3:** Community-driven (see what others actually need)
- Each phase ships working software, enables learning

**Why Document Everything?**
- Future me will forget design decisions
- Potential contributors need context
- Writing clarifies thinking
- Publishable artifact even if no community forms

## User Stories & Use Cases

This section captures the concrete scenarios that drive the Graph architecture design. Each use case represents a real problem that Pipeline (DAG) couldn't solve.

### Primary Use Case: Multi-Turn Conversational RAG

**As a** developer building a conversational AI system,  
**I want** to define a graph that loops back for follow-up retrieval,  
**So that** I can handle multi-turn conversations without restarting the pipeline.

#### Scenario: Research Assistant

```
User: "What are the key findings in the 2024 AI safety papers?"
System: [retrieves papers] → [generates summary]
User: "Can you focus on the alignment section?"
System: [retrieves MORE context using conversation history] → [refines answer]
User: "How does this compare to 2023?"
System: [retrieves 2023 papers too] → [generates comparison]
```

**Why Pipeline fails:** Step 2+ requires looping back to retrieval. DAGs can't cycle.

**Graph solution:**
```python
@route
def should_continue(messages: list) -> Literal["retrieve", END]:
    if user_says_done(messages[-1]) or len(messages) > 20:
        return END
    return "retrieve"  # Loops back

graph = Graph(
    nodes=[add_user_message, retrieve, generate, add_assistant, should_continue],
    entrypoint="add_user_message"
)
```

#### Acceptance Criteria
- [ ] Conversation runs 5+ turns without manual intervention
- [ ] Each turn correctly uses full conversation history for retrieval
- [ ] State persists between turns (messages accumulate)
- [ ] Clear termination (END when user is done)

---

### Use Case: Agentic Tool Loop

**As a** developer building an AI agent,  
**I want** the LLM to decide which tool to call and loop until satisfied,  
**So that** I can build autonomous agents that solve complex tasks.

#### Scenario: Code Generation Agent

```
Agent: Analyze task → decide: "need to read file"
Agent: [reads file] → analyze → decide: "need to search codebase"
Agent: [searches] → analyze → decide: "ready to generate code"
Agent: [generates] → self-review → decide: "needs refinement"
Agent: [refines] → self-review → decide: "done"
```

**Why Pipeline fails:** Number of iterations unknown. Can't pre-define DAG depth.

**Graph solution:**
```python
ToolChoice = Literal["read_file", "search", "generate", "refine", END]

@route
def decide_action(analysis: str, tools_used: list) -> ToolChoice:
    return llm.decide(analysis, tools_used)  # LLM picks next action

@node(output_name="tools_used")  # Accumulator
def track_tools(tools_used: list, last_tool: str) -> list:
    return tools_used + [last_tool]
```

#### Acceptance Criteria
- [ ] Agent runs variable number of iterations (not fixed)
- [ ] LLM can choose any available tool at each step
- [ ] Tool usage history tracked correctly (accumulator)
- [ ] Terminates when LLM decides task is complete

---

### Use Case: Iterative Refinement

**As a** developer building a content generation system,  
**I want** to generate → evaluate → refine in a loop,  
**So that** I can achieve quality thresholds without manual iteration.

#### Scenario: Document Generator with Quality Gate

```
Generate draft → Evaluate (score: 0.6) → Below threshold → Refine
Refine → Evaluate (score: 0.75) → Below threshold → Refine  
Refine → Evaluate (score: 0.92) → Above threshold → Done
```

**Graph solution:**
```python
@node(output_name="draft")
def generate(prompt: str, feedback: str | None) -> str:
    return llm.generate(prompt, previous_feedback=feedback)

@node(output_name=("score", "feedback"))
def evaluate(draft: str) -> tuple[float, str]:
    return critic.evaluate(draft)

@route
def quality_gate(score: float) -> Literal["generate", END]:
    return END if score > 0.9 else "generate"
```

#### Acceptance Criteria
- [ ] Loop continues until quality threshold met
- [ ] Feedback from evaluator passed to next generation
- [ ] Maximum iteration limit prevents infinite loops
- [ ] Final output is the high-quality version

---

### Use Case: Human-in-the-Loop Approval (Phase 2)

**As a** developer building a workflow requiring human approval,  
**I want** execution to pause, wait for human input, then resume,  
**So that** I can build supervised AI systems.

#### Scenario: Content Moderation Pipeline

```
Generate content → [PAUSE: await human review]
Human approves → Continue to publish
Human rejects → Loop back to regenerate with feedback
```

**Graph solution (Phase 2):**
```python
@interrupt  # New decorator for Phase 2
def human_review(content: str) -> Literal["publish", "regenerate"]:
    # Execution pauses here, state saved
    # Resumes when human provides decision
    pass

# Usage
result = graph.run(inputs={...})  # Returns checkpoint
# ... human reviews ...
result = graph.run(checkpoint=saved, inputs={"decision": "publish"})
```

#### Acceptance Criteria
- [ ] Execution pauses at interrupt node
- [ ] State persists across pause (can be hours/days)
- [ ] Resume from checkpoint with human input
- [ ] Both approve and reject paths work correctly

---

### Use Case: Parallel Branch Merge

**As a** developer with independent processing paths,  
**I want** branches to execute independently and merge results,  
**So that** I can parallelize where possible.

#### Scenario: Multi-Source Research

```
Query → [Branch A: search academic papers]
     → [Branch B: search news articles]  
     → [Branch C: search internal docs]
     → [Merge: combine all sources] → Generate answer
```

**Graph solution:**
```python
# Three independent retrievers (no data dependencies between them)
@node(output_name="academic_docs")
def search_academic(query: str) -> list: ...

@node(output_name="news_docs")
def search_news(query: str) -> list: ...

@node(output_name="internal_docs")
def search_internal(query: str) -> list: ...

# Merge waits for all three
@node(output_name="combined")
def merge(academic_docs: list, news_docs: list, internal_docs: list) -> list:
    return academic_docs + news_docs + internal_docs
```

#### Acceptance Criteria
- [ ] All three searches execute (order doesn't matter for correctness)
- [ ] Merge node waits for all inputs
- [ ] No artificial sequencing between independent branches
- [ ] Works with both SeqEngine (sequential) and future parallel engine

---

### Use Case: Conditional Skip (Existing Feature)

**As a** developer with optional processing steps,  
**I want** to skip nodes based on runtime conditions,  
**So that** I can build efficient conditional workflows.

#### Scenario: Cache-Aware Processing

```
Check cache → [HIT] → Return cached result (skip expensive processing)
           → [MISS] → Run expensive processing → Cache result → Return
```

**Graph solution (leveraging existing @branch):**
```python
@branch(when_true=return_cached, when_false=process_fresh)
def check_cache(query: str, cache: dict) -> bool:
    return query in cache

@node(output_name="result")
def return_cached(query: str, cache: dict) -> str:
    return cache[query]

@node(output_name="result")  # Same output name OK - exclusive branches
def process_fresh(query: str) -> str:
    return expensive_computation(query)
```

#### Acceptance Criteria
- [ ] Cache hit skips expensive node entirely
- [ ] Both branches produce same output name
- [ ] Skipped nodes don't execute (verified via callbacks)
- [ ] Gate signals track which path was taken

---

### Anti-Use-Cases (What NOT to Build)

These scenarios are explicitly out of scope:

| Scenario | Why Not | Alternative |
|----------|---------|-------------|
| **Distributed job queue** | Not a task queue, it's a graph executor | Use Celery/RQ + call graph.run() in worker |
| **Real-time event streaming** | Phase 1 doesn't expose token streaming | Wait for Phase 2 `.iter()` API |
| **Multi-tenant isolation** | No RBAC, single-user focus | Add at application layer if needed |
| **Sub-millisecond latency** | Correctness > performance for Phase 1 | Profile and optimize specific bottlenecks later |
| **Distributed graph execution** | Nodes run in single process | Future: explore Daft integration |

---

### User Story Priority Matrix

| Story | Phase | Priority | Complexity | Dependencies |
|-------|-------|----------|------------|--------------|
| Multi-turn RAG | 1 | **Critical** | High | Core architecture |
| Agentic tool loop | 1 | High | Medium | Route decorator |
| Iterative refinement | 1 | High | Medium | Route decorator |
| Parallel branch merge | 1 | Medium | Low | Already works with DAG |
| Conditional skip | 1 | Medium | Low | Existing @branch |
| Human-in-the-loop | 2 | High | High | Checkpoint system |

## Functional Requirements

This section specifies the capabilities the Graph system must provide, organized by component.

### FR1: Graph Construction

#### FR1.1: Node Registration

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR1.1.1 | `Graph` accepts a list of nodes (functions decorated with `@node`, `@route`, `@branch`) | Must | 1 |
| FR1.1.2 | Edges are inferred from function signatures (parameter names match output names) | Must | 1 |
| FR1.1.3 | Duplicate output names are rejected unless from mutually exclusive branches | Must | 1 |
| FR1.1.4 | Unknown parameter names (not produced by any node or provided as input) raise clear error | Must | 1 |
| FR1.1.5 | Self-referencing nodes (output_name in own parameters) detected and rejected | Must | 1 |

#### FR1.2: Graph Validation (Build-Time)

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR1.2.1 | Cycle detection: `Graph` identifies if graph contains cycles | Must | 1 |
| FR1.2.2 | Entrypoint required for cyclic graphs, optional for DAGs | Must | 1 |
| FR1.2.3 | All `@route` targets must reference existing node names or `END` | Must | 1 |
| FR1.2.4 | Route target validation uses `Literal` type hints for static checking | Should | 1 |
| FR1.2.5 | Termination path validation: cycles must have path to `END` | Must | 1 |
| FR1.2.6 | Sequential producer validation: nodes producing same output must have different distances from entrypoint | Must | 1 |
| FR1.2.7 | Invalid graphs fail with actionable error messages (what, why, how to fix) | Must | 1 |

#### FR1.3: NetworkX Integration

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR1.3.1 | `Graph` wraps `nx.DiGraph` internally | Must | 1 |
| FR1.3.2 | Node attributes store: hypernode object, is_gate flag, metadata | Must | 1 |
| FR1.3.3 | Edge attributes store: edge_type (data/control), value names, gate decisions | Must | 1 |
| FR1.3.4 | Standard NetworkX algorithms used for: cycle detection, reachability, ancestors, topological sort | Must | 1 |
| FR1.3.5 | Graph structure accessible for visualization (`graph.nx_graph` property) | Should | 1 |

---

### FR2: Execution Model

#### FR2.1: Reactive Dataflow

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR2.1.1 | Nodes execute when all required inputs are available | Must | 1 |
| FR2.1.2 | Staleness detection: node re-executes if any input version changed since last run | Must | 1 |
| FR2.1.3 | Sole producer rule: accumulator nodes don't re-trigger from own output | Must | 1 |
| FR2.1.4 | Version tracking: each value has monotonically increasing version number | Must | 1 |
| FR2.1.5 | Ready set computation: determine which nodes can execute given current state | Must | 1 |

#### FR2.2: Control Flow

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR2.2.1 | `@route` decorator returns target node name as string | Must | 1 |
| FR2.2.2 | Route can return `END` sentinel to terminate execution | Must | 1 |
| FR2.2.3 | Route decision creates control edge to target node | Must | 1 |
| FR2.2.4 | `@branch` decorator (existing) continues to work for mutually exclusive paths | Must | 1 |
| FR2.2.5 | Gates block downstream nodes until decision is made | Must | 1 |

#### FR2.3: Execution Loop

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR2.3.1 | Single unified algorithm handles DAGs, branches, AND cycles | Must | 1 |
| FR2.3.2 | Loop terminates when: no nodes ready AND (END reached OR all outputs produced) | Must | 1 |
| FR2.3.3 | Infinite loop detection: configurable max iterations with clear error | Must | 1 |
| FR2.3.4 | Execution order within ready set is deterministic (alphabetical or registration order) | Should | 1 |

---

### FR3: State Management

#### FR3.1: GraphState

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR3.1.1 | `GraphState` tracks all value names, their current values, and versions | Must | 1 |
| FR3.1.2 | Input values initialized with version 0 | Must | 1 |
| FR3.1.3 | Each node execution increments version of its outputs | Must | 1 |
| FR3.1.4 | State tracks which nodes have executed and their last input versions | Must | 1 |
| FR3.1.5 | State is serializable for checkpointing (Phase 2) | Should | 2 |

#### FR3.2: Conflict Detection

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR3.2.1 | Parallel producer conflict detected before execution starts | Must | 1 |
| FR3.2.2 | Conflict error includes: which nodes, which value, why conflict occurred | Must | 1 |
| FR3.2.3 | Error suggests resolution options (remove input, add dependency, use checkpoint) | Must | 1 |

---

### FR4: Caching

#### FR4.1: Signature Computation

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR4.1.1 | Cache signature = hash(code_hash + env_hash + input_values_hash) | Must | 1 |
| FR4.1.2 | Signature uses actual VALUES, not version numbers | Must | 1 |
| FR4.1.3 | Same inputs produce same signature regardless of iteration count | Must | 1 |
| FR4.1.4 | Different conversation turns (different messages) produce different signatures | Must | 1 |

#### FR4.2: Cache Integration

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR4.2.1 | Existing `DiskCache` works with `Graph` (same as Pipeline) | Must | 1 |
| FR4.2.2 | Cache check happens before node execution | Must | 1 |
| FR4.2.3 | Cache hit skips execution, uses cached value, updates state | Must | 1 |
| FR4.2.4 | Node-level `cache=False` disables caching for that node | Must | 1 |

---

### FR5: Decorators

#### FR5.1: @node Decorator

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR5.1.1 | `@node(output_name="x")` wraps function as pipeline node | Must | 1 |
| FR5.1.2 | Multiple outputs: `@node(output_name=("x", "y"))` with tuple return | Must | 1 |
| FR5.1.3 | `cache` parameter controls cacheability (default True) | Must | 1 |
| FR5.1.4 | Function remains callable without framework (`node.func(args)`) | Must | 1 |
| FR5.1.5 | Async functions supported (`async def`) | Must | 1 |
| FR5.1.6 | Generator functions accumulated automatically | Must | 1 |

#### FR5.2: @route Decorator (New)

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR5.2.1 | `@route` marks function as routing decision node | Must | 1 |
| FR5.2.2 | Return type must be `Literal[...]` with valid node names or `END` | Must | 1 |
| FR5.2.3 | Return value determines next node to activate | Must | 1 |
| FR5.2.4 | Route nodes are never cached (decisions must be re-evaluated) | Must | 1 |
| FR5.2.5 | Invalid return value (not in Literal) raises runtime error | Must | 1 |

#### FR5.3: @branch Decorator (Existing)

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR5.3.1 | `@branch(when_true="node_a", when_false="node_b")` routes based on bool (string targets) | Must | 1 |
| FR5.3.2 | Branch targets validated at Graph init (fail fast if target doesn't exist) | Must | 1 |
| FR5.3.3 | Branch targets can produce same output name (mutually exclusive) | Must | 1 |
| FR5.3.4 | Gate signals track which path was taken | Must | 1 |

#### FR5.4: InterruptNode (New)

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR5.4.1 | `InterruptNode(name="x", input_param="prompt", response_param="response")` declares pause point | Must | 2 |
| FR5.4.2 | `input_param` specifies which value to surface to user | Must | 2 |
| FR5.4.3 | `response_param` specifies where to write user's response | Must | 2 |
| FR5.4.4 | Optional `response_type` for validation | Should | 2 |
| FR5.4.5 | Framework provides plumbing, user defines prompt/response types | Must | 2 |

---

### FR6: API Surface

#### FR6.1: Graph Class (Pure Definition)

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR6.1.1 | `Graph(nodes=[...], entrypoint="name")` constructor | Must | 1 |
| FR6.1.2 | Graph has NO `run()` method - use Runner instead | Must | 1 |
| FR6.1.3 | `graph.visualize()` generates visual representation | Should | 1 |
| FR6.1.4 | `graph.bind(**kwargs)` sets default input values | Should | 1 |
| FR6.1.5 | `graph.as_node()` wraps graph for nesting | Must | 1 |
| FR6.1.6 | `.as_node().rename(inputs={old: new}, outputs={old: new})` renames interfaces | Must | 1 |
| FR6.1.7 | `.as_node().map_over(names, mode="zip")` enables internal batch processing | Should | 1 |
| FR6.1.8 | `graph.root_args` returns required inputs | Must | 1 |
| FR6.1.9 | `graph.unfulfilled_args` returns inputs not yet bound | Should | 1 |
| FR6.1.10 | `graph.bound_inputs` returns dict of bound values | Should | 1 |

**Input Value Priority (highest to lowest):**
1. **Edge connections** - Values from upstream nodes (exclusive - cancels optionality)
2. **Runtime inputs** - Values provided to `runner.run(graph, inputs={...})`
3. **Bound values** - Values set via `graph.bind(param=value)`
4. **Function defaults** - Parameter defaults in function signature

**Critical rule:** If an upstream node produces a value for a parameter, that parameter becomes REQUIRED and any defaults/bindings are ignored. This prevents ambiguity.

#### FR6.2: Runner Classes (Execution)

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR6.2.1 | `Runner(cache=..., callbacks=...)` for sync execution | Must | 1 |
| FR6.2.2 | `AsyncRunner(cache=..., callbacks=...)` for async execution | Must | 1 |
| FR6.2.3 | `runner.run(graph, inputs={...})` executes once | Must | 1 |
| FR6.2.4 | `runner.run(graph, inputs={...}, select=["pattern"])` filters outputs | Should | 1 |
| FR6.2.5 | `runner.map(graph, inputs={...}, map_over="x")` batch execution | Should | 1 |
| FR6.2.6 | `async_runner.iter(graph, inputs={...})` returns event stream | Must | 2 |
| FR6.2.7 | Runner owns cache and callbacks (execution-specific config) | Must | 1 |
| FR6.2.8 | Same graph can be used with different runners | Must | 1 |
| FR6.2.9 | `Runner` raises error if graph has async nodes | Must | 1 |

#### FR6.3: Specialized Runners

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR6.3.1 | `DaftRunner` for distributed DataFrame execution | Could | 3 |
| FR6.3.2 | `DaskRunner` for parallel batch processing | Could | 3 |
| FR6.3.3 | Runners can be nested via `.as_node(runner=...)` | Should | 2 |

---

### FR7: Callbacks & Observability

#### FR7.1: Lifecycle Events

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR7.1.1 | `on_graph_start(inputs)` fired when execution begins | Must | 1 |
| FR7.1.2 | `on_graph_end(outputs)` fired when execution completes | Must | 1 |
| FR7.1.3 | `on_node_start(node_name, inputs)` fired before each node | Must | 1 |
| FR7.1.4 | `on_node_end(node_name, outputs, duration)` fired after each node | Must | 1 |
| FR7.1.5 | `on_node_cached(node_name)` fired on cache hit | Must | 1 |
| FR7.1.6 | `on_route_decision(node_name, target)` fired when route decides | Must | 1 |
| FR7.1.7 | `on_iteration_start(iteration_number)` fired each cycle iteration | Should | 1 |

#### FR7.2: Existing Callbacks

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR7.2.1 | `ProgressCallback` works with Graph (shows progress) | Should | 1 |
| FR7.2.2 | `TelemetryCallback` works with Graph (Logfire tracing) | Should | 1 |

---

### FR8: Error Handling

#### FR8.1: Error Message Format

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR8.1.1 | All errors include: error type, what went wrong | Must | 1 |
| FR8.1.2 | All errors include: why it went wrong (context) | Must | 1 |
| FR8.1.3 | All errors include: how to fix (2-3 options) | Must | 1 |
| FR8.1.4 | Flow analysis included when entrypoint set (shows normal path) | Should | 1 |

#### FR8.2: Error Types

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR8.2.1 | `GraphConfigError` for build-time validation failures | Must | 1 |
| FR8.2.2 | `ParallelProducersError` for runtime conflicts | Must | 1 |
| FR8.2.3 | `RouteTargetError` for invalid route return values | Must | 1 |
| FR8.2.4 | `MaxIterationsError` for infinite loop detection | Must | 1 |
| FR8.2.5 | `MissingInputError` for unfulfilled required inputs | Must | 1 |

---

### FR9: Phase 2 Features (Deferred)

These are tracked but not implemented in Phase 1:

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR9.1 | `InterruptNode` for declarative human-in-the-loop pause points | Must | 2 |
| FR9.2 | `runner.run(graph, checkpoint=saved)` for resume from state | Must | 2 |
| FR9.3 | `async_runner.iter(graph, inputs={...})` for event streaming | Must | 2 |
| FR9.4 | Token-by-token streaming via `StreamingChunkEvent` | Must | 2 |
| FR9.5 | Checkpoint serialization/deserialization with `Checkpointer` protocol | Must | 2 |
| FR9.6 | Visualization shows cycles, active node, gate state | Should | 2 |
| FR9.7 | Three-layer architecture (UI Protocol, Observability, Durability) | Should | 2 |
| FR9.8 | `session_id` / `run_id` identity model for correlation | Must | 2 |

---

### FR10: Input Resolution & Initialization

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR10.1 | Input resolution priority: edge values (if available) > runtime inputs > bound values > function defaults | Must | 1 |
| FR10.2 | For cyclic edges, user-provided inputs serve as **initial values** on first iteration | Must | 1 |
| FR10.3 | Subsequent iterations use edge-produced values, ignoring the initial input | Must | 1 |
| FR10.4 | Parameters with cyclic edges must be provided at runtime as initial values (fail if missing) | Must | 1 |
| FR10.5 | `graph.root_args` includes parameters that need initialization (even if edge-connected in cycles) | Must | 1 |
| FR10.6 | `MissingInitialValueError` raised when cyclic parameter not provided, with clear explanation | Must | 1 |
| FR10.7 | `graph.bound_inputs` returns dict of values set via `.bind()` | Should | 1 |
| FR10.8 | `graph.unfulfilled_args` returns parameters needing initialization but not yet bound | Should | 1 |

---

### FR11: Nested Graph Composition

| ID | Requirement | Priority | Phase |
|----|-------------|----------|-------|
| FR11.1 | `graph.as_node()` wraps cyclic graphs as opaque execution units | Must | 1 |
| FR11.2 | Nested cyclic graphs execute their internal loops independently until they reach END | Must | 1 |
| FR11.3 | Outer graph waits for nested graph to complete before continuing | Must | 1 |
| FR11.4 | `.as_node().rename(inputs={...}, outputs={...})` works for cyclic graphs | Should | 1 |
| FR11.5 | `.as_node().map_over(names, mode="zip")` works for cyclic graphs | Could | 2 |

---

### Functional Requirements Traceability

| User Story | Required FRs |
|------------|--------------|
| Multi-turn RAG | FR1.2.1-2, FR2.1.1-5, FR2.2.1-3, FR2.3.1-3, FR4.1.3-4, FR10.1-6 |
| Agentic tool loop | FR2.2.1-3, FR5.2.1-5, FR2.3.3 |
| Iterative refinement | FR2.2.1-3, FR5.2.1-5, FR4.1.3-4 |
| Parallel branch merge | FR1.1.2, FR2.1.1, FR6.1.2 |
| Conditional skip | FR5.3.1-4, FR2.2.4-5 |
| Human-in-the-loop | FR9.1-5 (Phase 2) |
| Nested cyclic pipelines | FR11.1-4 |

## Technical Constraints & Non-Functional Requirements

This section defines the technical boundaries, dependencies, and quality attributes that constrain the implementation.

### TC1: Language & Runtime

| Constraint | Specification | Rationale |
|------------|---------------|-----------|
| **Python version** | ≥3.10 | Match patterns, `Literal` types, union syntax (`X \| Y`) |
| **Type hints** | Required on all public APIs | IDE support, documentation, static analysis |
| **Async support** | Native `async/await` | Modern LLM APIs are async-first |
| **No global state** | All state in explicit objects | Testability, thread safety |

### TC2: Dependencies

#### Core Dependencies (Required)

| Package | Version | Purpose |
|---------|---------|---------|
| `networkx` | ≥3.0 | Graph data structure, algorithms |

#### Optional Dependencies (Extras)

| Package | Extra Name | Purpose |
|---------|------------|---------|
| `graphviz` | `[viz]` | Static SVG visualization |
| `ipywidgets` | `[viz]` | Interactive Jupyter visualization |
| `logfire` | `[telemetry]` | Distributed tracing |
| `daft` | `[daft]` | Distributed execution (Phase 3) |

#### Dependency Philosophy

- **Zero required dependencies beyond NetworkX** for core functionality
- User installs extras only for features they need
- No transitive dependency on ML frameworks (torch, transformers, etc.)
- Pin minimum versions, not exact versions (flexibility for users)

### TC3: Compatibility

#### With Existing HyperNodes Code

| Item | Compatibility | Notes |
|------|---------------|-------|
| `@node` decorator | Full | Same signature, same behavior |
| `@branch` decorator | Full | Works in Graph context |
| `DiskCache` | Full | Same cache backend |
| `Pipeline` class | Deprecated | Marked legacy, not removed |
| Existing callbacks | Partial | May need Graph-specific events |
| `SeqEngine` | N/A | Graph uses `GraphEngine` |
| `DaftEngine` | Future | Phase 3 if needed |

#### With External Tools

| Tool | Compatibility | Notes |
|------|---------------|-------|
| Jupyter notebooks | Full | Primary development environment |
| pytest | Full | Standard test runner |
| mypy | Target | Type hints should pass strict mode |
| VS Code | Full | Should work with Python extension |
| Logfire | Full | Via `TelemetryCallback` |

### TC4: Code Quality Standards

| Standard | Target | Enforcement |
|----------|--------|-------------|
| **Test coverage** | ≥80% line coverage | `pytest-cov` in CI |
| **Type coverage** | ≥90% of public APIs | `mypy --strict` |
| **Docstrings** | All public classes/functions | Manual review |
| **No `# type: ignore`** | Minimize, document when needed | Code review |
| **Linting** | Zero errors | `ruff` in CI |
| **Formatting** | Consistent | `ruff format` |

### TC5: Architecture Constraints

#### Must Follow

| Constraint | Description |
|------------|-------------|
| **Pure functions** | Node functions must be testable without framework |
| **Explicit dependencies** | All dependencies visible in function signatures |
| **Engine owns runtime** | Cache, callbacks, execution strategy in Engine |
| **Graph owns structure** | DAG definition, validation, node registry in Graph |
| **State is immutable** | GraphState operations return new state, don't mutate |
| **Fail fast** | Validate as early as possible (build-time > run-time) |

#### Must Avoid

| Anti-Pattern | Why |
|--------------|-----|
| Global registries | Makes testing hard, hidden dependencies |
| Implicit state modification | Debugging nightmare |
| Framework-coupled functions | Can't reuse outside HyperNodes |
| Magic method resolution | Explicit > implicit |
| Inheritance hierarchies | Composition over inheritance |

---

### NFR1: Performance

#### Phase 1 Targets (Not Optimized)

| Metric | Target | Notes |
|--------|--------|-------|
| **Graph construction** | <100ms for 100 nodes | Build-time validation |
| **Per-node overhead** | <1ms | Framework overhead, not node execution |
| **Memory per value** | <1KB metadata | Version, timestamps, etc. |
| **Cache lookup** | <10ms | Disk cache signature check |

**Philosophy:** Phase 1 optimizes for correctness. Performance profiling happens after MVP validates the model.

#### Phase 2+ Optimization Opportunities

| Opportunity | When to Consider |
|-------------|------------------|
| Parallel node execution | Multiple independent nodes ready |
| Lazy value resolution | Large values not needed downstream |
| Incremental validation | Re-validate only changed subgraph |
| Cache warming | Pre-populate cache for known inputs |

---

### NFR2: Reliability

| Requirement | Specification |
|-------------|---------------|
| **Deterministic execution** | Same inputs + same code = same outputs |
| **No silent failures** | All errors raised, not swallowed |
| **Graceful degradation** | Missing optional deps don't crash |
| **Idempotent re-runs** | Running twice with cache = same result |
| **Infinite loop protection** | Configurable max iterations (default: 1000) |

---

### NFR3: Testability

| Requirement | Specification |
|-------------|---------------|
| **Unit testable nodes** | `assert node.func(x) == expected` works |
| **Mock-friendly** | No hidden dependencies to mock |
| **Deterministic tests** | No flaky tests from race conditions |
| **Fast tests** | Unit tests complete in <5s total |
| **Integration tests** | Full graph runs in <30s |

#### Testing Patterns

```python
# Unit test a node (no framework needed)
def test_retrieve():
    result = retrieve.func(query="test", messages=[])
    assert isinstance(result, list)

# Integration test a graph
def test_multi_turn():
    graph = Graph(nodes=[...], entrypoint="start")
    result = graph.run(inputs={"query": "test", "messages": []})
    assert "response" in result

# Test with mocked LLM
def test_with_mock():
    mock_llm = Mock(return_value="mocked response")
    result = graph.run(inputs={"llm": mock_llm, ...})
    mock_llm.assert_called_once()
```

---

### NFR4: Maintainability

| Requirement | Specification |
|-------------|---------------|
| **Single responsibility** | Each module has one clear purpose |
| **Low coupling** | Modules interact via defined interfaces |
| **High cohesion** | Related code lives together |
| **Self-documenting** | Code reads clearly with minimal comments |
| **Changelog** | All changes documented in CHANGELOG.md |

#### Module Responsibility Map

| Module | Responsibility |
|--------|----------------|
| `graph.py` | Graph construction, validation, structure |
| `graph_state.py` | Value storage, versioning, staleness |
| `graph_engine.py` | Execution loop, reactive scheduling |
| `route.py` | `@route` decorator, control flow |
| `node.py` | `@node` decorator (existing, unchanged) |
| `branch.py` | `@branch` decorator (existing, unchanged) |
| `cache.py` | Cache backends (existing, unchanged) |
| `callbacks.py` | Callback protocol (extended for Graph) |

---

### NFR5: Observability

| Requirement | Specification |
|-------------|---------------|
| **Execution trace** | Know which nodes ran in what order |
| **Timing data** | Duration per node, total duration |
| **State snapshots** | Value versions at each step |
| **Error context** | Full stack trace + graph state on error |
| **Structured logging** | JSON-friendly log output |

#### Observability via Callbacks

```python
class DebugCallback(PipelineCallback):
    def on_node_end(self, node_name, outputs, duration):
        print(f"{node_name}: {duration:.2f}ms")
        
    def on_route_decision(self, node_name, target):
        print(f"Route {node_name} → {target}")
```

---

### NFR6: Documentation

| Requirement | Specification |
|-------------|---------------|
| **API reference** | Docstrings on all public APIs |
| **Getting started** | 5-minute quickstart guide |
| **Migration guide** | Pipeline → Graph patterns |
| **Architecture docs** | ADRs for major decisions |
| **Examples** | Working code for each use case |

#### Documentation Structure

```
docs/
├── quickstart.md           # Get running in 5 minutes
├── concepts/
│   ├── reactive-dataflow.md    # How the execution model works
│   ├── cycles-and-routing.md   # @route, END, cycles
│   └── state-and-versioning.md # GraphState internals
├── migration/
│   └── pipeline-to-graph.md    # Pattern mapping
├── api/
│   └── reference.md            # Auto-generated from docstrings
├── examples/
│   ├── multi-turn-rag.py
│   ├── agentic-loop.py
│   └── iterative-refinement.py
└── adr/
    ├── 001-networkx-foundation.md
    ├── 002-reactive-vs-imperative.md
    └── 003-route-decorator-design.md
```

---

### NFR7: Security

| Requirement | Specification |
|-------------|---------------|
| **No code execution from strings** | No `eval()`, no dynamic imports |
| **No network access in core** | Network only in user nodes |
| **No filesystem access in core** | Only cache backends touch disk |
| **Safe pickling** | Cache uses restricted unpickler |
| **No credential storage** | User manages secrets externally |

**Note:** This is a library, not a service. Security is primarily about not introducing vulnerabilities, not about access control.

---

### NFR8: Async Execution

| Requirement | Target | Measurement |
|-------------|--------|-------------|
| **Async overhead** | <5ms per await | Benchmark async vs sync execution |
| **Concurrent node execution** | Framework supports (engine decides) | AsyncRunner can overlap I/O-bound nodes |
| **Generator accumulation** | Memory-efficient | Streaming chunks don't buffer entire response |

---

### NFR9: Error Recovery

| Requirement | Target | Measurement |
|-------------|--------|-------------|
| **Node failure isolation** | Single node failure doesn't corrupt state | Integration tests with failing nodes |
| **Partial results** | State available even after failure | Exception handler preserves GraphState |
| **Retry support** | Node-level retry possible | Callback hook for retry decisions |

---

### NFR10: Compatibility Policy

| Requirement | Specification |
|-------------|---------------|
| **Phase 1 (v0.5.x)** | Breaking changes allowed; no backward compatibility guarantees |
| **Phase 2+ (v0.6+)** | Deprecation warnings before breaking changes |
| **v1.0+** | Semantic versioning; breaking changes only in major versions |

---

### Technical Debt Tolerance

| Area | Tolerance | Phase 1 Approach |
|------|-----------|------------------|
| **Performance** | High | Correctness first, profile later |
| **Edge cases** | Low | Handle all known edge cases |
| **Test coverage** | Low | ≥80% from start |
| **Documentation** | Medium | Core docs now, polish later |
| **Error messages** | Low | Invest heavily upfront |

## Risks & Mitigations

This section identifies potential risks and mitigation strategies.

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Reactive dataflow complexity** | Medium | High | Start with simplest use case (multi-turn RAG); validate model before adding features |
| **Cycle detection edge cases** | Medium | Medium | Use NetworkX's battle-tested algorithms; extensive test coverage for cycle scenarios |
| **Staleness detection bugs** | Medium | High | Comprehensive unit tests for versioning; property-based testing for edge cases |
| **Sole producer rule failures** | Low | High | Static analysis at build time; clear error messages when rule is violated |
| **Runner pattern confusion** | Medium | Medium | Clear documentation; error messages explain "use Runner to execute" |
| **InterruptNode state management** | Medium | Medium | Defer to Phase 2; prototype with simple cases first |

### Design Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Over-engineering** | Medium | Medium | Validate each feature against real multi-turn RAG use case |
| **API churn** | Low | Medium | No users yet; can iterate freely in Phase 1 |
| **Wrong abstraction level** | Medium | High | Build the concrete use case first; abstract patterns after they emerge |
| **Runner vs Engine confusion** | Medium | Low | Consistent naming: Runner = user-facing, Engine = internal |
| **Literal type validation gaps** | Low | Medium | Test with mypy strict mode; validate at runtime too |

### Implementation Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **NetworkX performance** | Low | Medium | Profile after correctness; only optimize proven bottlenecks |
| **Async/sync mixing** | Medium | Medium | Clear separation via `Runner` vs `AsyncRunner`; error early if mismatch |
| **Generator accumulation bugs** | Medium | Medium | `inspect.isgenerator()` detection; explicit `streaming=True` opt-in for edge cases |
| **Checkpoint serialization** | Medium | Medium | Use standard pickling; test with complex state objects |
| **Callback timing** | Low | Low | Fire-and-forget callbacks; errors in callbacks don't break execution |

### Schedule Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Scope creep** | Medium | Medium | Strict phase boundaries; defer nice-to-haves to Phase 2+ |
| **Perfectionism** | High | Medium | Ship working MVP; iterate based on actual usage |
| **Research rabbit holes** | Medium | Low | Time-box research; document findings even if not implemented |
| **Testing time underestimated** | Medium | Medium | Write tests alongside code, not after |

### Dependency Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **NetworkX breaking changes** | Low | Medium | Pin minimum version; minimal surface area usage |
| **Python version requirements** | Low | Low | Target 3.10+ (widely adopted); document reasoning |
| **Optional dependency issues** | Low | Low | Graceful degradation; clear error if optional dep missing |

---

### Risk Matrix Summary

```
        High Impact
            ↑
  Staleness │ Reactive   
    bugs    │ dataflow
            │         
  ─────────────────────→ High Likelihood
            │
  Cycle     │ Scope
  detection │ creep
            │
        Low Impact
```

**Focus Areas:**
1. **Staleness detection** - Core to correctness; invest in testing
2. **Reactive dataflow complexity** - Validate with real use case early
3. **Scope creep** - Maintain strict phase boundaries

## Project Scoping & Phased Development

### MVP Strategy & Philosophy

**MVP Approach:** Problem-Solving MVP
- Solve the core problem (multi-turn RAG) with minimal features
- Validate through personal usage before community release
- "Infrastructure-as-learning" - deep understanding over feature breadth

**Resource Requirements (Phase 1):**
- Team: Solo developer
- Skills: Python, graph theory, async programming
- Timeline: Weeks to months (personal project pace)

### MVP Feature Set (Phase 1)

**Core User Journeys Supported:**
1. Multi-turn Conversational RAG (primary)
2. Agentic Tool Loop
3. Iterative Refinement
4. Conditional Skip (existing @branch)
5. Parallel Branch Merge (DAG subset)

**Must-Have Capabilities:**

| Capability | Rationale |
|------------|-----------|
| Cyclic execution | Core differentiator, enables multi-turn |
| Reactive dataflow with versioning | Prevents infinite loops, enables staleness detection |
| `@route` decorator | String-based routing with build-time validation |
| `Runner`/`AsyncRunner` separation | Clean architecture, pure Graph definition |
| Build-time validation | Fail-fast philosophy |
| Generator accumulation | Modern LLM APIs return generators |
| Existing `DiskCache` integration | Don't reinvent working infrastructure |

**Explicitly Deferred from MVP:**
- Token-by-token streaming to users (generators accumulated internally)
- Human-in-the-loop (`InterruptNode`)
- Checkpointing/resume
- Distributed execution
- Visualization updates for cycles

### Post-MVP Features

**Phase 2 (Polish) - Triggered by:**
- Phase 1 validated with real multi-turn RAG usage
- Pattern clarity (know what abstractions are needed)
- Personal need for streaming or human-in-loop

**Phase 2 Planned Features:**

| Feature | Priority | Complexity |
|---------|----------|------------|
| `InterruptNode` | High | High |
| `.iter()` streaming API | High | Medium |
| Checkpointing | High | High |
| Event streaming | Medium | Medium |
| Visualization updates | Medium | Medium |

**Phase 3 (Expansion) - Triggered by:**
- Community interest (stars, issues, PRs)
- Specific user requests
- Personal use case requiring scale

**Phase 3 Potential Features:**

| Feature | Trigger Condition |
|---------|-------------------|
| Distributed execution (DaftRunner) | 3+ users request parallel batch |
| Durable workflows | Multi-day execution needs emerge |
| Web debugging UI | Demand beyond Jupyter |
| Multi-backend cache | Redis/S3 adapter requests |

### Risk Mitigation Strategy

**Technical Risks:**

| Risk | Mitigation |
|------|------------|
| Reactive dataflow complexity | Start with simplest use case (multi-turn RAG); extensive testing |
| Staleness detection bugs | Property-based testing; comprehensive unit tests |
| Cycle detection edge cases | Use NetworkX battle-tested algorithms |

**Market Risks:**

| Risk | Mitigation |
|------|------------|
| No community adoption | Personal success is primary; community is bonus |
| LangGraph dominance | Different philosophy (pure functions); niche appeal acceptable |

**Resource Risks:**

| Risk | Mitigation |
|------|------------|
| Scope creep | Strict phase boundaries; defer nice-to-haves |
| Perfectionism | Ship working MVP; iterate based on usage |

### Scope Reduction Contingencies

**If time-constrained, cut in this order:**
1. Visualization updates (can use existing Pipeline viz)
2. `ProgressCallback` integration (callbacks optional)
3. `map()` support (focus on `run()` first)
4. Async support (sync `Runner` sufficient for validation)

**Minimum viable scope:**
- `Graph` class with NetworkX
- `@route` decorator  
- `Runner.run()` (sync only)
- Build-time validation
- Core staleness detection
