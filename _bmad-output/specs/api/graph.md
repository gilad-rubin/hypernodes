# Graph API Specification

## Overview

`Graph` is a pure structure definition. It has no execution logic, no state, no cache - just structure and validation.

## Constructor

```python
class Graph:
    def __init__(
        self,
        nodes: list[HyperNode | RouteNode | BranchNode | InterruptNode],
        *,
        validate_types: bool = False,
    ) -> None:
        """
        Construct a graph from nodes.
        
        Args:
            nodes: List of node objects (decorated functions or InterruptNode).
            validate_types: If True, check type hint congruence between
                           connected nodes (output type matches input type).
        
        Raises:
            GraphConfigError: Build-time validation failures.
        
        Build-time validations performed:
            1. All @route/@branch targets exist (or are END)
            2. No conflicting parallel producers (unless mutually exclusive)
            3. Cycles have termination paths (route to END or reach leaf)
            4. No deadlocks (cycles have valid starting inputs)
            5. No self-loops without gates
            6. If validate_types=True: type congruence checks
        
        Example:
            graph = Graph(nodes=[embed, retrieve, generate, route_decision])
        """
```

## Properties

### Structure Properties

```python
@property
def nodes(self) -> dict[str, HyperNode | RouteNode | BranchNode | InterruptNode]:
    """Map of node name → node object."""

@property
def node_names(self) -> set[str]:
    """Set of all node names."""

@property
def has_cycles(self) -> bool:
    """True if graph contains any cycles."""

@property
def cycles(self) -> list[list[str]]:
    """List of cycles (each cycle is list of node names)."""

@property
def leaf_nodes(self) -> list[str]:
    """Nodes with no outgoing edges."""

@property
def gates(self) -> list[RouteNode | BranchNode]:
    """All gate nodes (@route and @branch)."""

@property
def interrupt_nodes(self) -> list[InterruptNode]:
    """All interrupt nodes."""
```

### Input/Output Properties

```python
@property
def root_args(self) -> set[str]:
    """
    Parameter names that can be provided as inputs.
    
    Includes:
    - Parameters with no incoming edge and no default
    - Parameters with incoming edge (for cycle initialization)
    """

@property
def required_inputs(self) -> set[str]:
    """
    Parameters that MUST be provided.
    
    Parameters with no edge AND no default.
    """

@property
def optional_inputs(self) -> set[str]:
    """
    Parameters that CAN be provided but have defaults.
    
    Parameters with no edge BUT have default.
    """

@property
def outputs(self) -> set[str]:
    """All output names produced by nodes."""

@property
def leaf_outputs(self) -> set[str]:
    """Output names from leaf nodes (default return values)."""
```

### NetworkX Access

```python
@property
def nx_graph(self) -> nx.DiGraph:
    """
    Underlying NetworkX graph for visualization/analysis.
    
    Node attributes:
        - 'hypernode': The HyperNode/RouteNode/BranchNode object
        - 'is_gate': True for route/branch nodes
        - 'node_type': 'node' | 'route' | 'branch' | 'interrupt'
    
    Edge attributes:
        - 'edge_type': 'data' | 'control'
        - 'value_name': For data edges, the value being passed
        - 'condition': For control edges, the gate condition
    """
```

## Methods

### bind()

```python
def bind(self, **values: Any) -> Graph:
    """
    Return new Graph with values pre-bound.
    
    Bound values are used when:
    - Parameter has no incoming edge, AND
    - No runtime input provided
    
    Args:
        **values: Parameter name → value mappings
    
    Returns:
        New Graph instance with bound values.
        Original graph is not modified.
    
    Example:
        graph = Graph(nodes=[process])
        bound = graph.bind(temperature=0.7, max_tokens=1000)
        
        # These are equivalent:
        runner.run(bound, inputs={"query": "hello"})
        runner.run(graph, inputs={"query": "hello", "temperature": 0.7, "max_tokens": 1000})
    """
```

### as_node()

```python
def as_node(
    self,
    *,
    output_name: str | tuple[str, ...] | None = None,
    runner: Runner | AsyncRunner | None = None,
) -> HyperNode:
    """
    Wrap graph as a node for composition.
    
    Args:
        output_name: Override output name(s). Default: leaf outputs.
        runner: Runner for nested execution. Default: inherit from parent.
    
    Returns:
        HyperNode that executes this graph when called.
    
    Behavior:
        - Cyclic graphs execute until END, then return outputs
        - DAG graphs execute once
        - Nested graphs are opaque to parent (cache as single unit)
    
    Example:
        rag_graph = Graph(nodes=[retrieve, generate, refine, check_done])
        
        outer = Graph(nodes=[
            preprocess,
            rag_graph.as_node(output_name="rag_result"),
            postprocess,
        ])
    """
```

### Renaming for Composition

```python
def as_node(self).rename(
    inputs: dict[str, str] | None = None,
    outputs: dict[str, str] | None = None,
) -> HyperNode:
    """
    Rename inputs/outputs for composition.
    
    Args:
        inputs: Map old input name → new input name
        outputs: Map old output name → new output name
    
    Example:
        # Inner graph expects "query", outer has "user_question"
        nested = inner_graph.as_node().rename(
            inputs={"query": "user_question"},
            outputs={"response": "inner_response"},
        )
    """
```

## Build-Time Validation Details

### 1. Route Target Validation

```python
def _validate_route_targets(self):
    """All @route/@branch targets must exist or be END."""
    for gate in self.gates:
        targets = gate.targets if isinstance(gate, RouteNode) else [gate.when_true, gate.when_false]
        for target in targets:
            if target is END:
                continue
            if target not in self.node_names:
                closest = find_closest_match(target, self.node_names)
                raise GraphConfigError(
                    f"@route target '{target}' doesn't exist\n\n"
                    f"  → {gate.name}() declares target '{target}'\n"
                    f"  → No node named '{target}' in this graph\n"
                    f"  → Available nodes: {sorted(self.node_names)}\n"
                    + (f"\nDid you mean '{closest}'?" if closest else "")
                )
```

### 2. Parallel Producer Validation

```python
def _validate_no_conflicts(self):
    """Multiple producers of same output must be mutually exclusive."""
    for output in self.outputs:
        producers = self._producers_of(output)
        if len(producers) > 1:
            # Check if all pairs are mutually exclusive
            for i, p1 in enumerate(producers):
                for p2 in producers[i+1:]:
                    if not self._mutually_exclusive(p1, p2):
                        raise GraphConfigError(
                            f"Multiple nodes produce '{output}'\n\n"
                            f"  → {p1} creates '{output}'\n"
                            f"  → {p2} creates '{output}'\n\n"
                            f"The problem: If both run, which value should we use?\n\n"
                            f"How to fix:\n"
                            f"  Option A: Rename one output to avoid conflict\n"
                            f"  Option B: Add @branch to make them mutually exclusive\n"
                        )
```

### 3. Cycle Termination Validation

```python
def _validate_cycle_termination(self):
    """Every cycle must have a path to termination."""
    for cycle in self.cycles:
        can_terminate = False
        
        # Check for route with END
        for node_name in cycle:
            node = self.nodes[node_name]
            if isinstance(node, RouteNode) and END in node.targets:
                can_terminate = True
                break
        
        # Check for path to leaf
        if not can_terminate:
            for node_name in cycle:
                for leaf in self.leaf_nodes:
                    if nx.has_path(self.nx_graph, node_name, leaf):
                        can_terminate = True
                        break
        
        if not can_terminate:
            raise GraphConfigError(
                f"Cycle has no termination path\n\n"
                f"  → Cycle: {' → '.join(cycle)}\n"
                f"  → No @route returns END\n"
                f"  → No path to a leaf node\n\n"
                f"How to fix:\n"
                f"  Add a @route that can return END to break the cycle"
            )
```

### 4. Deadlock Detection

```python
def _validate_no_deadlock(self):
    """Cycles must have valid starting inputs."""
    for cycle in self.cycles:
        can_start = False
        
        for node_name in cycle:
            node = self.nodes[node_name]
            # Can this node start from external inputs?
            external_deps = [
                p for p in node.parameters
                if not self._is_produced_in_cycle(p, cycle)
            ]
            
            # If all external deps can be satisfied, cycle can start
            if all(self._can_satisfy(d) for d in external_deps):
                can_start = True
                break
        
        if not can_start:
            raise GraphConfigError(
                f"Cycle cannot start - deadlock detected\n\n"
                f"  → Cycle: {' → '.join(cycle)}\n"
                f"  → Every node depends on another node in the cycle\n"
                f"  → No external entry point\n\n"
                f"How to fix:\n"
                f"  Ensure at least one node can start from external inputs"
            )
```

## Usage Examples

### Basic Graph

```python
@node(output_name="embedded")
def embed(text: str) -> list[float]:
    return model.encode(text)

@node(output_name="result")
def classify(embedded: list[float]) -> str:
    return classifier.predict(embedded)

graph = Graph(nodes=[embed, classify])
# Edges inferred: text → embed → classify
```

### Cyclic Graph

```python
@node(output_name="response")
def generate(messages: list) -> str:
    return llm.chat(messages)

@node(output_name="messages")
def accumulate(messages: list, response: str) -> list:
    return messages + [{"role": "assistant", "content": response}]

@route(targets=["generate", END])
def check_done(messages: list) -> str:
    return END if is_complete(messages) else "generate"

graph = Graph(nodes=[generate, accumulate, check_done])
# Cycle: generate → accumulate → check_done → generate
```

### Nested Graphs

```python
inner = Graph(nodes=[step1, step2, step3])
outer = Graph(nodes=[
    preprocess,
    inner.as_node(output_name="inner_result"),
    postprocess,
])
```
