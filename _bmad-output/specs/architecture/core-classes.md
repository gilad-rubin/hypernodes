# Core Classes Specification

## Class Hierarchy Overview

```
Graph                 # Pure structure definition
├── wraps nx.DiGraph  # NetworkX graph internally
├── nodes: list       # Registered hypernodes
└── validation        # Build-time checks

GraphState            # Runtime value storage
├── values: dict      # name → value
├── versions: dict    # name → version number
└── node_history      # Which nodes ran, when

Runner                # Sync execution
├── cache             # Optional cache backend
├── callbacks         # Optional callbacks
└── run(graph, inputs) → outputs

AsyncRunner           # Async execution
├── cache             # Optional cache backend
├── callbacks         # Optional callbacks
├── run(graph, inputs) → outputs
└── iter(graph, inputs) → AsyncIterator[Event]

DaftRunner            # Distributed execution (DAG-only)
├── cache             # Optional cache backend
└── map(graph, inputs, map_over) → DataFrame
```

## Graph Class

### Purpose
Pure graph structure definition. No execution logic, no state, no cache. Just structure + validation.

### Constructor
```python
class Graph:
    def __init__(
        self,
        nodes: list[HyperNode | RouteNode | BranchNode | InterruptNode],
        *,
        validate_types: bool = False,  # Opt-in type congruence checking
    ) -> None:
        """
        Construct a graph from a list of nodes.
        
        Validation happens HERE (build-time):
        - All @route targets exist or are END
        - No conflicting parallel producers (unless mutually exclusive)
        - Cycles have termination paths
        - No deadlocks (cycles have valid starting inputs)
        
        Raises:
            GraphConfigError: If validation fails
        """
```

### Key Properties
```python
@property
def has_cycles(self) -> bool:
    """True if graph contains any cycles."""

@property
def root_args(self) -> set[str]:
    """Parameter names that can be provided as inputs."""

@property
def nx_graph(self) -> nx.DiGraph:
    """Access underlying NetworkX graph for visualization."""

@property
def nodes(self) -> dict[str, HyperNode]:
    """Map of node name → node object."""
```

### Key Methods
```python
def bind(self, **values) -> Graph:
    """
    Return new Graph with values pre-bound.
    Bound values apply when parameter has NO edge.
    
    Returns:
        New Graph instance with bound values
    """

def as_node(
    self,
    *,
    runner: Runner | AsyncRunner | None = None,
) -> HyperNode:
    """
    Wrap graph as a node for composition.
    Cyclic graphs execute their internal loops until END.
    
    Args:
        runner: Runner to use for nested execution (optional)
    
    Returns:
        HyperNode that executes this graph
    """
```

## GraphState Class

### Purpose
Tracks all values, their versions, and execution history during a run.

### Constructor
```python
class GraphState:
    def __init__(self, initial_values: dict[str, Any] | None = None) -> None:
        """
        Initialize state with optional starting values.
        Initial values get version 0.
        """
```

### Key Properties
```python
@property
def values(self) -> dict[str, Any]:
    """Current values (read-only view)."""

@property
def versions(self) -> dict[str, int]:
    """Version numbers for each value (read-only view)."""
```

### Key Methods
```python
def get(self, name: str) -> Any:
    """Get value by name. Raises KeyError if not present."""

def get_version(self, name: str) -> int:
    """Get version number for a value."""

def set(self, name: str, value: Any) -> GraphState:
    """
    Return new state with updated value.
    Increments version number.
    State is immutable - returns new instance.
    """

def is_stale(self, node_name: str, input_versions: dict[str, int]) -> bool:
    """
    Check if node needs re-execution.
    True if any input version > version when node last ran.
    """

def to_checkpoint(self) -> bytes:
    """Serialize state for persistence."""

@classmethod
def from_checkpoint(cls, data: bytes) -> GraphState:
    """Restore state from checkpoint."""
```

## Runner Class

### Purpose
Synchronous execution of graphs. Owns cache and callbacks.

### Constructor
```python
class Runner:
    def __init__(
        self,
        *,
        cache: Cache | None = None,
        callbacks: list[Callback] | None = None,
    ) -> None:
        """
        Create a runner with optional cache and callbacks.
        
        Args:
            cache: Cache backend (e.g., DiskCache)
            callbacks: List of callbacks for observability
        """
```

### Key Methods
```python
def run(
    self,
    graph: Graph,
    inputs: dict[str, Any],
    *,
    select: list[str] | None = None,
    session_id: str | None = None,
    max_iterations: int = 1000,
) -> dict[str, Any]:
    """
    Execute graph synchronously.
    
    Args:
        graph: Graph to execute
        inputs: Input values (determines where cycles start)
        select: Optional list of output names to return (default: all leaf outputs)
        session_id: Optional session ID for grouping related runs
        max_iterations: Maximum loop iterations before InfiniteLoopError
    
    Returns:
        Dict of output name → value
    
    Raises:
        GraphConfigError: If graph is invalid
        ConflictError: If parallel producers conflict
        MissingInputError: If required input not provided
        InfiniteLoopError: If max_iterations exceeded
        IncompatibleRunnerError: If graph has async nodes
    """

def map(
    self,
    graph: Graph,
    inputs: dict[str, Any],
    *,
    map_over: str | list[str],
) -> list[dict[str, Any]]:
    """
    Execute graph for each item in map_over parameter(s).
    
    Args:
        graph: Graph to execute
        inputs: Input values (map_over params should be lists)
        map_over: Parameter name(s) to iterate over
    
    Returns:
        List of output dicts, one per input item
    """
```

## AsyncRunner Class

### Purpose
Asynchronous execution with streaming support. Required for async nodes and InterruptNode.

### Constructor
```python
class AsyncRunner:
    def __init__(
        self,
        *,
        cache: Cache | None = None,
        callbacks: list[Callback] | None = None,
    ) -> None:
        """Same as Runner, but for async execution."""
```

### Key Methods
```python
async def run(
    self,
    graph: Graph,
    inputs: dict[str, Any],
    *,
    select: list[str] | None = None,
    session_id: str | None = None,
    max_iterations: int = 1000,
    checkpoint: bytes | None = None,  # Resume from checkpoint
) -> dict[str, Any]:
    """
    Execute graph asynchronously.
    
    Additional args vs Runner:
        checkpoint: Resume execution from saved state
    
    Returns:
        Dict of output name → value
        If interrupted, returns partial outputs + checkpoint
    """

async def iter(
    self,
    graph: Graph,
    inputs: dict[str, Any],
    *,
    session_id: str | None = None,
) -> AsyncIterator[Event]:
    """
    Execute graph and yield events as they occur.
    
    Yields:
        Event objects (see Events specification)
    
    Use for:
        - Token-by-token streaming
        - Real-time progress updates
        - Human-in-the-loop interrupts
    """
```

## DaftRunner Class

### Purpose
Distributed execution using Daft DataFrames. **DAG-only** - no cycles, gates, or interrupts.

### Constructor
```python
class DaftRunner:
    def __init__(
        self,
        *,
        cache: Cache | None = None,
    ) -> None:
        """
        Create distributed runner.
        Note: callbacks have limited support (no iteration events).
        """
```

### Key Methods
```python
def map(
    self,
    graph: Graph,
    inputs: dict[str, Any],
    *,
    map_over: str | list[str],
) -> "daft.DataFrame":
    """
    Execute graph in distributed fashion using Daft.
    
    Args:
        graph: Must be a DAG (no cycles)
        inputs: Input values
        map_over: Parameter(s) to distribute over
    
    Returns:
        Daft DataFrame with results
    
    Raises:
        IncompatibleRunnerError: If graph has cycles, gates, or interrupts
    """
```

### Validation
```python
# DaftRunner must validate at run time:
if graph.has_cycles:
    raise IncompatibleRunnerError(
        "This graph has cycles, but DaftRunner doesn't support cycles.\n\n"
        "The problem: DaftRunner uses Daft DataFrames for distributed execution, "
        "which requires a DAG structure.\n\n"
        "How to fix:\n"
        "  Option A: Use Runner or AsyncRunner instead\n"
        "  Option B: Restructure as a DAG"
    )
```
