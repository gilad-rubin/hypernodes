# HyperNodes Quickstart: Calculator Agent

This quickstart demonstrates how to build a calculator agent using HyperNodes.

## Overview

We'll build an agent that:
1. Takes a user query
2. Decides whether to call tools (add, multiply, divide)
3. Executes tools when needed
4. Loops until the answer is ready
5. Returns the final result

## Prerequisites

You'll need:
- A [Claude (Anthropic)](https://www.anthropic.com/) account and API key
- Set `ANTHROPIC_API_KEY` environment variable

## Step 1: Define Tools and Model

First, we'll set up the LLM and define calculator tools.

```python
from langchain.tools import tool
from langchain.chat_models import init_chat_model

# Initialize the model
model = init_chat_model(
    "claude-sonnet-4-5-20250929",
    temperature=0
)

# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b

# Create tools lookup
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)
```

## Step 2: Define Graph Nodes

In HyperNodes, we use the `@node` decorator to create nodes. Each node is a function that takes inputs and produces outputs.

### LLM Node

The LLM node calls the model and returns both the response and updated messages.

```python
from hypernodes import node
from langchain.messages import SystemMessage, AnyMessage

@node(output_name=("llm_response", "messages"))
def call_llm(messages: list[AnyMessage]) -> tuple[AnyMessage, list[AnyMessage]]:
    """Call LLM to decide whether to use tools or respond directly."""

    response = model_with_tools.invoke([
        SystemMessage(
            content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
        )
    ] + messages)

    # Return both the response and the accumulated messages
    return response, messages + [response]
```

**Key HyperNodes Pattern**: Notice we return `(llm_response, messages)`. The `messages` output feeds back as input, creating an accumulator pattern for the conversation history.

### Tool Execution Node

The tool node executes any tool calls from the LLM response.

```python
from langchain.messages import ToolMessage

@node(output_name="messages")
def execute_tools(llm_response: AnyMessage, messages: list[AnyMessage]) -> list[AnyMessage]:
    """Execute tool calls from the LLM response."""

    tool_results = []
    for tool_call in llm_response.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        tool_results.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )

    # Append tool results to message history
    return messages + tool_results
```

## Step 3: Define Routing Logic

In HyperNodes, routing is done with `@route` or `@branch` decorators. Here we'll use `@route` to decide whether to execute tools or end.

```python
from hypernodes import route, END

@route(targets=["execute_tools", END])
def should_continue(llm_response: AnyMessage) -> str:
    """Route to tool execution or end based on LLM's decision."""

    # If the LLM made tool calls, route to execute_tools
    if llm_response.tool_calls:
        return "execute_tools"

    # Otherwise, we're done
    return END
```

**Key Difference from LangGraph**: In HyperNodes, `@route` replaces LangGraph's `add_conditional_edges`. The function returns the target node name directly.

## Step 4: Build the Graph

Now we construct the graph by passing all nodes. HyperNodes automatically infers edges from parameter/output name matching.

```python
from hypernodes import Graph

# Build the graph
agent = Graph(
    nodes=[
        call_llm,
        should_continue,
        execute_tools,
    ],
    name="calculator_agent"
)
```

**How Edges Work**: HyperNodes automatically connects nodes based on matching names:
- `call_llm` produces `llm_response` and `messages`
- `should_continue` reads `llm_response` → automatic edge from `call_llm`
- `execute_tools` reads `llm_response` and `messages` → edges from both `call_llm` and the gate
- The cycle: `execute_tools` produces `messages`, which feeds back to `call_llm`

## Step 5: Execute the Agent

Use a runner to execute the graph.

```python
from hypernodes import AsyncRunner
from langchain.messages import HumanMessage

# Create runner
runner = AsyncRunner()

# Execute
result = await runner.run(
    agent,
    inputs={"messages": [HumanMessage(content="Add 3 and 4.")]}
)

# Print the final messages
for msg in result["messages"]:
    msg.pretty_print()
```

## Understanding the Execution Flow

Let's trace what happens:

1. **Initial state**: `messages = [HumanMessage("Add 3 and 4.")]`

2. **First iteration**:
   - `call_llm` receives `messages` → calls LLM → produces `llm_response` (with tool call) and updated `messages`
   - `should_continue` receives `llm_response` → sees tool calls → routes to `"execute_tools"`
   - `execute_tools` receives `llm_response` and `messages` → executes `add(3, 4)` → produces updated `messages` with tool result

3. **Second iteration** (cycle):
   - `call_llm` receives updated `messages` (now includes tool result) → calls LLM → produces final response (no tool calls) and updated `messages`
   - `should_continue` receives `llm_response` → no tool calls → routes to `END`

4. **Done**: Returns final `messages` with complete conversation

## Key HyperNodes Concepts

### 1. Data Flow Creates Edges
```python
# LangGraph (explicit):
builder.add_edge("llm_call", "should_continue")

# HyperNodes (implicit):
# call_llm produces "llm_response"
# should_continue reads "llm_response"
# → Edge created automatically
```

### 2. Accumulator Pattern for State
```python
# LangGraph uses reducers:
class State(TypedDict):
    messages: Annotated[list, operator.add]

# HyperNodes uses explicit accumulation:
@node(output_name="messages")
def execute_tools(messages: list[AnyMessage], ...) -> list[AnyMessage]:
    return messages + [new_message]  # Explicit append
```

### 3. Routing via Return Values
```python
# LangGraph conditional edge:
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])

# HyperNodes @route:
@route(targets=["execute_tools", END])
def should_continue(...) -> str:
    return "execute_tools"  # or END
```

### 4. Cycles Through Self-Reference
The `messages` parameter forms a cycle:
- `call_llm` produces `messages`
- `execute_tools` consumes and produces `messages`
- `call_llm` consumes `messages` again → cycle!

## Complete Example

```python
from hypernodes import node, route, Graph, AsyncRunner, END
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, ToolMessage, AnyMessage

# 1. Setup model and tools
model = init_chat_model("claude-sonnet-4-5-20250929", temperature=0)

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide a by b."""
    return a / b

tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# 2. Define nodes
@node(output_name=("llm_response", "messages"))
def call_llm(messages: list[AnyMessage]) -> tuple[AnyMessage, list[AnyMessage]]:
    """Call LLM to decide whether to use tools."""
    response = model_with_tools.invoke([
        SystemMessage(content="You are a helpful assistant tasked with performing arithmetic.")
    ] + messages)
    return response, messages + [response]

@route(targets=["execute_tools", END])
def should_continue(llm_response: AnyMessage) -> str:
    """Route based on whether LLM made tool calls."""
    if llm_response.tool_calls:
        return "execute_tools"
    return END

@node(output_name="messages")
def execute_tools(llm_response: AnyMessage, messages: list[AnyMessage]) -> list[AnyMessage]:
    """Execute tool calls."""
    tool_results = []
    for tool_call in llm_response.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        tool_results.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return messages + tool_results

# 3. Build graph
agent = Graph(
    nodes=[call_llm, should_continue, execute_tools],
    name="calculator_agent"
)

# 4. Execute
async def main():
    runner = AsyncRunner()
    result = await runner.run(
        agent,
        inputs={"messages": [HumanMessage(content="What is 3 + 4?")]}
    )

    for msg in result["messages"]:
        msg.pretty_print()

# Run it
import asyncio
asyncio.run(main())
```

## Next Steps

- Learn about [streaming execution](./02-streaming.md) with `AsyncRunner.iter()`
- Explore [human-in-the-loop](./03-human-in-loop.md) with `InterruptNode`
- Build [nested graphs](./04-nested-graphs.md) with `Graph.as_node()`
- Use [batch processing](./05-batch-processing.md) with `.map()`

## Comparison: LangGraph vs HyperNodes

| Feature | LangGraph | HyperNodes |
|---------|-----------|------------|
| **State** | Explicit `TypedDict` with reducers | Implicit via data flow |
| **Edges** | `add_edge()` / `add_conditional_edges()` | Inferred from parameter names |
| **Routing** | Conditional edge functions | `@route` / `@branch` decorators |
| **Cycles** | Self-loops in graph structure | Parameter matches output name |
| **Accumulation** | `Annotated[list, operator.add]` | Explicit `messages + [new]` |
| **Graph Build** | Imperative builder pattern | Declarative node list |

Both frameworks are powerful, but HyperNodes emphasizes **data flow** and **type-driven connections** over explicit graph construction.
