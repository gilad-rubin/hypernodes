# HyperNodes Examples

This directory contains example implementations that demonstrate HyperNodes concepts by porting tutorials from other graph frameworks.

## Examples

| # | Name | Source | Concepts |
|---|------|--------|----------|
| 01 | [Calculator Agent](01_calculator_agent.py) | [LangGraph Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart) | Async nodes, cycles, @branch, mutually exclusive producers |

## Running Examples

```bash
# Set up environment variable
export ANTHROPIC_API_KEY=your_key_here

# Run an example
uv run python examples/01_calculator_agent.py
```

## Key HyperNodes Concepts

### Implicit Edge Construction
Unlike LangGraph where you explicitly wire nodes with `add_edge()`, HyperNodes infers edges from parameter names:

```python
# LangGraph
builder.add_edge("call_llm", "call_tools")

# HyperNodes - edges inferred automatically
@node(output_name="llm_response")
def call_llm(messages): ...

@node(output_name="tool_results")
def call_tools(llm_response): ...  # ← consumes llm_response, edge is automatic
```

### Cycles via Same-Named Outputs
Cycles are created when a node outputs a value with the same name as an existing input:

```python
@node(output_name="messages")  # ← outputs "messages"
def accumulate(messages, llm_response, tool_results):  # ← consumes "messages"
    return messages + [llm_response] + tool_results
    # This creates a cycle: messages → ... → accumulate → messages
```

### Binary Routing with @branch
For true/false decisions:

```python
@branch(when_true="call_tools", when_false="finalize")
def has_tool_calls(llm_response) -> bool:
    return bool(llm_response.tool_calls)
```

### Multi-way Routing with @route
For multiple possible targets:

```python
@route(targets=["call_llm", END])
def should_continue(tool_results) -> str:
    if tool_results:
        return "call_llm"
    return END
```

### Mutually Exclusive Producers
Multiple nodes can produce the same output if they're on mutually exclusive paths:

```python
@node(output_name="tool_results")  # Same output name
def call_tools(llm_response): ...   # Only runs when branch = True

@node(output_name="tool_results")  # Same output name
def finalize(llm_response): ...     # Only runs when branch = False
```

## Comparison: LangGraph vs HyperNodes

| Aspect | LangGraph | HyperNodes |
|--------|-----------|------------|
| State | `TypedDict` with annotations | No state object - values flow by name |
| Edges | Explicit `add_edge()` | Implicit from parameter matching |
| Accumulation | `operator.add` annotation | Node outputs same-named value |
| Routing | `add_conditional_edges()` | `@branch` / `@route` decorators |
| Graph building | `StateGraph(State).compile()` | `Graph(nodes=[...])` |
| Execution | `graph.invoke()` | `runner.run(graph, inputs={})` |
