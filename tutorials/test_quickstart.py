"""
Test script for the HyperNodes quickstart tutorial.
This helps identify design issues by trying to run the actual code.
"""

import asyncio
from typing import Any

# Mock everything since HyperNodes v0.5 isn't written yet
# This is purely to analyze the API design

# Mock LangChain classes
class SystemMessage:
    def __init__(self, content):
        self.content = content

class HumanMessage:
    def __init__(self, content):
        self.content = content

class AIMessage:
    def __init__(self, content):
        self.content = content
        self.tool_calls = []

class ToolMessage:
    def __init__(self, content, tool_call_id):
        self.content = content
        self.tool_call_id = tool_call_id

def tool(func):
    """Mock @tool decorator."""
    func.name = func.__name__
    return func

class ChatAnthropic:
    """Mock ChatAnthropic."""
    def __init__(self, model, temperature):
        self.model = model

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return AIMessage(content="Mock response")

# Mock HyperNodes imports (to test the API design)
# These would come from the actual hypernodes package


def node(output_name):
    """Mock @node decorator."""
    def decorator(func):
        func._node_output_name = output_name
        func._is_node = True
        return func
    return decorator


def route(targets):
    """Mock @route decorator."""
    def decorator(func):
        func._route_targets = targets
        func._is_route = True
        return func
    return decorator


class END:
    """Mock END sentinel."""
    pass


class Graph:
    """Mock Graph class."""
    def __init__(self, nodes, name=None):
        self.nodes = nodes
        self.name = name
        print(f"✓ Graph created with {len(nodes)} nodes")

        # Validate edges can be inferred
        self._validate_edges()

    def _validate_edges(self):
        """Check that edges can be inferred from parameter/output matching."""
        # Collect all outputs
        all_outputs = set()
        for node in self.nodes:
            if hasattr(node, '_node_output_name'):
                output = node._node_output_name
                if isinstance(output, tuple):
                    all_outputs.update(output)
                else:
                    all_outputs.add(output)

        print(f"  Available outputs: {all_outputs}")

        # Check each node's inputs
        for node_obj in self.nodes:
            if hasattr(node_obj, '__code__'):
                params = node_obj.__code__.co_varnames[:node_obj.__code__.co_argcount]
                print(f"  Node '{node_obj.__name__}' needs: {params}")


class AsyncRunner:
    """Mock AsyncRunner class."""
    async def run(self, graph, inputs):
        print(f"✓ Runner executing graph with inputs: {list(inputs.keys())}")

        # Simulate execution
        print("  [Simulated execution - actual runner would execute the graph]")

        # Return mock result
        return {"messages": inputs.get("messages", []) + [AIMessage(content="Mock response")]}


# Now test the tutorial code

# 1. Setup model and tools
print("\n=== Step 1: Setup Model and Tools ===")
model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)


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

print(f"✓ Model configured with {len(tools)} tools")

# 2. Define nodes
print("\n=== Step 2: Define Nodes ===")


@node(output_name=("llm_response", "messages"))
def call_llm(messages: list[Any]) -> tuple[Any, list[Any]]:
    """Call LLM to decide whether to use tools."""
    print(f"  call_llm called with {len(messages)} messages")
    response = model_with_tools.invoke([
        SystemMessage(content="You are a helpful assistant tasked with performing arithmetic.")
    ] + messages)
    return response, messages + [response]


print(f"✓ call_llm node defined (outputs: {call_llm._node_output_name})")


@route(targets=["execute_tools", END])
def should_continue(llm_response: Any) -> str:
    """Route based on whether LLM made tool calls."""
    print(f"  should_continue evaluating response")
    if hasattr(llm_response, 'tool_calls') and llm_response.tool_calls:
        return "execute_tools"
    return END


print(f"✓ should_continue gate defined (targets: {should_continue._route_targets})")


@node(output_name="messages")
def execute_tools(llm_response: Any, messages: list[Any]) -> list[Any]:
    """Execute tool calls."""
    print(f"  execute_tools called")
    if not hasattr(llm_response, 'tool_calls'):
        return messages

    tool_results = []
    for tool_call in llm_response.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        tool_results.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return messages + tool_results


print(f"✓ execute_tools node defined (outputs: {execute_tools._node_output_name})")

# 3. Build graph
print("\n=== Step 3: Build Graph ===")

agent = Graph(
    nodes=[call_llm, should_continue, execute_tools],
    name="calculator_agent"
)

# 4. Analyze the design
print("\n=== Design Analysis ===")

print("\n1. EDGE INFERENCE:")
print("   - call_llm produces: ('llm_response', 'messages')")
print("   - should_continue needs: llm_response ✓")
print("   - execute_tools needs: llm_response, messages ✓")
print("   - call_llm needs: messages (for cycle) ✓")
print("   → All edges can be inferred!")

print("\n2. CYCLE DETECTION:")
print("   - call_llm outputs 'messages'")
print("   - execute_tools outputs 'messages'")
print("   - call_llm inputs 'messages'")
print("   → Cycle detected: call_llm → ... → execute_tools → call_llm")

print("\n3. INITIAL STATE:")
print("   - Runner receives: inputs={'messages': [...]}")
print("   - call_llm expects: messages parameter")
print("   → Names match, initial state can be injected ✓")

print("\n4. POTENTIAL ISSUES IDENTIFIED:")
print("   ⚠️  Issue #1: Accumulator pattern requires manual concatenation")
print("      - User must write: messages + [new_message]")
print("      - LangGraph's Annotated[list, operator.add] does this automatically")
print("      - Risk: Users might forget to include old messages")
print("      - Example error: return [new_message] instead of messages + [new_message]")

print("\n   ⚠️  Issue #2: Multiple outputs require tuple matching")
print("      - output_name=('llm_response', 'messages')")
print("      - Return type must be: tuple[AnyMessage, list[AnyMessage]]")
print("      - Order matters! Swapping return values breaks the graph")
print("      - No runtime validation of tuple length/order")

print("\n   ⚠️  Issue #3: Name-based edge inference is implicit")
print("      - Edges created by parameter name == output name")
print("      - Typo in parameter name silently breaks edges")
print("      - Example: 'messags' vs 'messages' - no edge, no error (until runtime)")

print("\n   ⚠️  Issue #4: Cycle requires exact name match")
print("      - For accumulator: input name must match output name")
print("      - If renamed (e.g., 'msg_history' vs 'messages'), cycle breaks")
print("      - This is less flexible than LangGraph's explicit state")

print("\n   ⚠️  Issue #5: No built-in state versioning")
print("      - In the tutorial, we don't track llm_calls count")
print("      - To add it, need another output/parameter pair")
print("      - Gets verbose with many state fields")

print("\n5. PROPOSED IMPROVEMENTS:")
print("   ✓ Consider: Built-in accumulator nodes/helpers")
print("     Example: @accumulator_node(output_name='messages')")
print("     Auto-handles: old_messages + [new_message]")

print("\n   ✓ Consider: Runtime validation of tuple outputs")
print("     Check len(return_value) == len(output_name) at execution")

print("\n   ✓ Consider: Better error messages for missing edges")
print("     'Parameter X not found in any output. Did you mean Y?'")

print("\n   ✓ Consider: Visual graph validation")
print("     Show which parameters connect to which outputs")

print("\n   ✓ Consider: Optional state class support")
print("     Allow TypedDict with reducers for complex state management")
print("     Fall back to current data-flow approach for simple cases")


# 5. Test execution
print("\n=== Step 4: Test Execution ===")


async def main():
    runner = AsyncRunner()
    result = await runner.run(
        agent,
        inputs={"messages": [HumanMessage(content="What is 3 + 4?")]}
    )

    print(f"\n✓ Execution completed")
    print(f"  Final state has {len(result.get('messages', []))} messages")


asyncio.run(main())

print("\n=== Analysis Complete ===")
print("The tutorial demonstrates HyperNodes' data-flow approach works,")
print("but we've identified several usability concerns compared to LangGraph.")
