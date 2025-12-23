# Decorators API Specification

## @node

### Purpose

Wraps a function as a graph node with named output(s).

### Signature

```python
def node(
    output_name: str | tuple[str, ...],
    *,
    cache: bool = True,
) -> Callable[[F], HyperNode[F]]:
    """
    Decorate a function as a graph node.
    
    Args:
        output_name: Name(s) for the output value(s).
                     Single string for one output.
                     Tuple of strings for multiple outputs (function must return tuple).
        cache: Whether to cache results (default True).
    
    Returns:
        HyperNode wrapping the function.
    
    Example:
        @node(output_name="embedding")
        def embed(text: str) -> list[float]:
            return model.encode(text)
        
        @node(output_name=("score", "explanation"))
        def evaluate(text: str) -> tuple[float, str]:
            return (0.95, "Good quality")
    """
```

### HyperNode Object

The decorated function becomes a `HyperNode` object:

```python
class HyperNode(Generic[F]):
    name: str              # Function name (used as node identifier)
    func: F                # Original function (still callable)
    output_name: str | tuple[str, ...]
    parameters: list[str]  # Parameter names from signature
    cache: bool
    
    def __call__(self, *args, **kwargs) -> Any:
        """Call the underlying function directly."""
        return self.func(*args, **kwargs)
```

### Properties

- **Portability**: `node.func(x)` works without framework
- **Testability**: `assert embed.func("hello") == expected`
- **Introspection**: `node.parameters`, `node.output_name` available

### Async Support

```python
@node(output_name="response")
async def generate(messages: list) -> str:
    return await llm.chat(messages)

# Works with AsyncRunner
# Raises IncompatibleRunnerError with sync Runner
```

### Generator Support

```python
@node(output_name="response")
async def stream_generate(messages: list):
    async for chunk in llm.stream(messages):
        yield chunk

# Framework automatically accumulates chunks
# Final value stored in state
# Streaming available via AsyncRunner.iter()
```

### Multiple Outputs

```python
@node(output_name=("docs", "scores"))
def retrieve(query: str) -> tuple[list[str], list[float]]:
    results = search(query)
    return [r.text for r in results], [r.score for r in results]

# Creates two values in state: "docs" and "scores"
# Downstream nodes can depend on either
```

---

## @route

### Purpose

Multi-way routing decision node. Returns target node name as string.

### Signature

```python
def route(
    targets: list[str | type[END]],
    *,
    cache: bool = False,
) -> Callable[[F], RouteNode[F]]:
    """
    Decorate a function as a routing decision node.
    
    Args:
        targets: REQUIRED list of valid target node names and/or END.
                 Every possible return value must be declared here.
        cache: Whether to cache decisions (default False).
               Usually False because decisions depend on runtime state.
    
    Returns:
        RouteNode wrapping the function.
    
    Raises:
        GraphConfigError: At Graph() if any target doesn't exist.
        InvalidRouteError: At runtime if return value not in targets.
    
    Example:
        @route(targets=["retrieve", "generate", END])
        def decide_next(messages: list) -> str:
            if is_done(messages):
                return END
            if needs_more_context(messages):
                return "retrieve"
            return "generate"
    """
```

### RouteNode Object

```python
class RouteNode(Generic[F]):
    name: str
    func: F
    targets: list[str]     # Validated target names
    parameters: list[str]
    cache: bool
    
    # Routes don't have output_name - they control flow, not data
```

### Validation

**Build-time (at `Graph()` construction):**
```python
# All targets must exist
for target in route_node.targets:
    if target is not END and target not in graph.node_names:
        raise GraphConfigError(
            f"@route target '{target}' doesn't exist\n\n"
            f"  → {route_node.name}() declares targets={route_node.targets}\n"
            f"  → No node named '{target}' in this graph\n"
            f"  → Available nodes: {graph.node_names}\n"
        )
```

**Runtime (after route executes):**
```python
result = route_node.func(**inputs)
if result not in route_node.targets and result is not END:
    raise InvalidRouteError(
        f"Route returned '{result}' but that's not a valid target\n\n"
        f"  → {route_node.name}() returned \"{result}\"\n"
        f"  → Valid targets are: {route_node.targets}\n"
        + (f"\nHint: Did you mean '{closest_match}'?" if closest_match else "")
    )
```

### Type Hints (Optional)

```python
# All valid - type hints are optional
@route(targets=["a", "b", END])
def decide(x: int) -> str: ...

@route(targets=["a", "b", END])
def decide(x: int) -> Literal["a", "b"] | type[END]: ...

@route(targets=["a", "b", END])
def decide(x: int): ...  # No return type hint
```

---

## @branch

### Purpose

Binary routing for boolean decisions. Syntactic sugar over `@route`.

### Signature

```python
def branch(
    when_true: str | HyperNode,
    when_false: str | HyperNode,
    *,
    cache: bool = False,
) -> Callable[[F], BranchNode[F]]:
    """
    Decorate a function as a binary routing decision.
    
    Args:
        when_true: Target when function returns True.
                   Can be node name (str) or node object.
        when_false: Target when function returns False.
        cache: Whether to cache (default False).
    
    Returns:
        BranchNode wrapping the function.
    
    Example:
        @branch(when_true="use_cache", when_false="compute")
        def check_cache(key: str, cache: dict) -> bool:
            return key in cache
    """
```

### BranchNode Object

```python
class BranchNode(Generic[F]):
    name: str
    func: F
    when_true: str         # Target node name
    when_false: str        # Target node name
    parameters: list[str]
    cache: bool
```

### Target Resolution

Targets can be strings or node objects:

```python
@node(output_name="result")
def path_a(x: int) -> int:
    return x + 1

@node(output_name="result")
def path_b(x: int) -> int:
    return x - 1

# Both valid:
@branch(when_true="path_a", when_false="path_b")
def gate1(x: int) -> bool: ...

@branch(when_true=path_a, when_false=path_b)  # Node objects
def gate2(x: int) -> bool: ...
```

### Mutual Exclusivity

Branch targets can produce the same output name:

```python
@branch(when_true="positive", when_false="negative")
def check_sign(x: int) -> bool:
    return x > 0

@node(output_name="label")  # Same output name
def positive(x: int) -> str:
    return "positive"

@node(output_name="label")  # Same output name - OK!
def negative(x: int) -> str:
    return "negative"

# Valid: positive and negative are mutually exclusive
```

---

## InterruptNode

### Purpose

Declarative pause point for human-in-the-loop workflows.

### Constructor

```python
class InterruptNode:
    def __init__(
        self,
        name: str,
        input_param: str,
        response_param: str,
        *,
        response_type: type | None = None,
    ):
        """
        Create an interrupt node.
        
        Args:
            name: Unique identifier for this interrupt.
            input_param: Parameter name containing value to show user.
            response_param: Parameter name where user's response goes.
            response_type: Optional type for validating response.
        
        Example:
            approval = InterruptNode(
                name="human_review",
                input_param="draft_content",
                response_param="approval_decision",
                response_type=ApprovalDecision,
            )
        """
        self.name = name
        self.input_param = input_param
        self.response_param = response_param
        self.response_type = response_type
```

### Usage in Graph

```python
# Create interrupt
review = InterruptNode(
    name="content_review",
    input_param="generated_content",
    response_param="user_feedback",
)

# Use in graph
graph = Graph(nodes=[
    generate_content,
    review,           # ← Pause here
    route_feedback,
    finalize,
])

# Execute with AsyncRunner
runner = AsyncRunner()
result = await runner.run(graph, inputs={...})

# If interrupted, result contains checkpoint
if result.interrupted:
    # ... show generated_content to user, get feedback ...
    result = await runner.run(
        graph,
        inputs={"user_feedback": feedback},
        checkpoint=result.checkpoint,
    )
```

### Requirements

- **AsyncRunner only** - InterruptNode requires async execution
- **Checkpoint persistence** - State must be serializable
- **Clear prompt/response contract** - Framework provides plumbing, user defines types

---

## Common Patterns

### Accumulator

```python
@node(output_name="messages")
def add_message(messages: list, new_message: dict) -> list:
    return messages + [new_message]

# Note: returns NEW list, doesn't mutate
# Sole producer rule prevents infinite loop
```

### Conditional Processing

```python
@branch(when_true="expensive_path", when_false="cheap_path")
def should_use_expensive(data: dict) -> bool:
    return data.get("quality_required", False)
```

### Multi-Step Pipeline

```python
@node(output_name="cleaned")
def clean(raw: str) -> str: ...

@node(output_name="embedded")
def embed(cleaned: str) -> list[float]: ...

@node(output_name="result")
def classify(embedded: list[float]) -> str: ...

# Edges inferred: raw → clean → embed → classify
```

### Cycle with Termination

```python
@node(output_name="draft")
def generate(prompt: str, feedback: str | None = None) -> str: ...

@node(output_name=("score", "feedback"))
def evaluate(draft: str) -> tuple[float, str]: ...

@route(targets=["generate", END])
def quality_gate(score: float, threshold: float = 0.9) -> str:
    return END if score >= threshold else "generate"
```
