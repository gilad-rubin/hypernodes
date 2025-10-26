# Pipeline Visualization

# Pipeline Visualization

The pipeline system provides powerful visualization capabilities using **Graphviz** to render DAGs. The visualization system automatically handles hierarchical nested pipelines with configurable expansion depth.

---

## Basic Usage

```python
# Simple visualization
pipeline.visualize()

# Save to file
pipeline.visualize(filename="pipeline.svg")

# Customize orientation
pipeline.visualize(orient="LR")  # Left to right instead of top to bottom
```

---

## Graph Building

The `build_graph()` function constructs a NetworkX directed graph from a pipeline:

```python
import networkx as nx
from pipeline_system.visualization import build_graph

graph = build_graph(pipeline)
```

**The graph contains:**

- **Function nodes**: Each `NodeDef` in the pipeline
- **Input parameter nodes**: String nodes for pipeline inputs
- **Edges**: Data flow connections with parameter names

**Graph construction algorithm:**

1. Add all function nodes from `pipeline.nodes`
2. For each function's parameters:
    - If parameter matches another node's output → edge from function to function
    - Otherwise → create input parameter node and edge to function

---

## Node Types

The visualization distinguishes three node types:

### Input Nodes

**External inputs to the pipeline**

- Displayed as dashed rectangles
- Light green background (`#90EE90`)
- Show parameter name, type hints, and default values

### Function Nodes

**Computation nodes**

- Displayed as rounded rectangles
- Sky blue background (`#87CEEB`)
- Show function name and output name with type annotation

### Grouped Input Nodes

**Multiple inputs used exclusively by one function**

- Displayed as solid rectangles
- Light green background (same as input nodes)
- Show multiple parameters in a table format
- Only created when `min_arg_group_size` threshold is met

---

## Hierarchical Visualization

**Nested pipelines can be visualized at different expansion depths:**

### Depth Control

```python
# Default: Show nested pipelines as single nodes
pipeline.visualize(depth=1)

# Expand one level: Show internal structure of immediate nested pipelines
pipeline.visualize(depth=2)

# Fully expand: Show all nested levels
pipeline.visualize(depth=None)  # or depth=float('inf')
```

### Collapsed View (depth=1)

**Nested pipelines appear as single function nodes:**

```
corpus ──▶ [encode_corpus] ──▶ encoded_corpus ──▶ [build_index] ──▶ index
```

- The pipeline node shows its name and output mapping
- Internal structure is hidden
- Cleanest view for high-level understanding

### Expanded View (depth=2)

**Nested pipelines are shown with internal structure:**

```
corpus ──▶ [encode_corpus (Pipeline)]
           ╭─────────────────────────────╮
           │ passage ──▶ [clean_text]    │
           │   ↓                          │
           │ cleaned_text ──▶ [encode]   │
           │   ↓                          │
           │ embedding ──▶ [pack]        │
           ╰─────────────────────────────╯
         ──▶ encoded_corpus ──▶ [build_index] ──▶ index
```

**Visual distinction:**

- Nested pipeline boundaries shown with container boxes
- Backend and cache configuration annotated on container
- Internal nodes indented or grouped visually

### Fully Expanded View (depth=None)

**All nesting levels are expanded recursively:**

- Useful for debugging deeply nested pipelines
- Can become large for complex hierarchies
- All intermediate data flows visible

---

## Flattening vs. Boxing

Two visualization modes for nested pipelines:

### Boxed Mode (default)

**Nested pipelines rendered as subgraphs with visible boundaries:**

```python
pipeline.visualize(depth=2, flatten=False)
```

**Graphviz cluster subgraphs** are used:

- Container box around nested pipeline nodes
- Label shows pipeline name and configuration
- Clear visual hierarchy
- Better for understanding structure

**Example output:**

```
┌─────────────────────────────────────────┐
│ gpu_pipeline [Modal, GPU: A100]         │
│  ┌────────────┐      ┌──────────────┐  │
│  │ preprocess │ ──▶  │   encode     │  │
│  └────────────┘      └──────────────┘  │
└─────────────────────────────────────────┘
```

### Flattened Mode

**Nested pipeline nodes rendered inline without containers:**

```python
pipeline.visualize(depth=2, flatten=True)
```

**Benefits:**

- Simpler visual layout
- Better for wide/complex DAGs
- No nested boxing overhead

**Trade-off:**

- Less obvious which nodes belong to which pipeline
- Configuration annotations move to individual nodes

**Example output:**

```
input ──▶ preprocess ──▶ encode ──▶ postprocess ──▶ output
         [Local]         [Modal GPU]  [Local]
```

---

## Configuration Annotations

Nested pipelines show their configuration in visualizations:

**Backend information:**

- `[Local]` - LocalBackend
- `[Modal, GPU: A100]` - ModalBackend with GPU
- `[Coiled]` - CoiledBackend

**Cache status:**

- `[RedisCache]` - Active Redis cache
- `[DiskCache: /tmp]` - Disk cache with path
- `[No cache]` - Caching disabled

**Inheritance indicators:**

- `[↓ Local]` - Inherited from parent
- `[Local*]` - Explicitly set (overrides parent)

**Example with full annotations:**

```
Outer Pipeline [Local*, RedisCache*] ━━━━━━━━━━━━━━━━ 100%
  ├─ preprocess [↓ Local, ↓ RedisCache]
  ├─ gpu_pipeline [Modal GPU*, ↓ RedisCache]
  │  ├─ encode [↓ Modal GPU, ↓ RedisCache]
  │  └─ transform [↓ Modal GPU, ↓ RedisCache]
  └─ postprocess [↓ Local, ↓ RedisCache]
```

---

## Parameter Grouping

When multiple input parameters are used exclusively by a single function, they can be grouped for cleaner visualization:

```python
# Default: Group parameters used by one function (min 2 params)
pipeline.visualize(min_arg_group_size=2)

# Disable grouping
pipeline.visualize(min_arg_group_size=None)

# Require at least 3 params to group
pipeline.visualize(min_arg_group_size=3)
```

**Without grouping:**

```
arg1 ──▶
arg2 ──▶ [function] ──▶ output
arg3 ──▶
```

**With grouping:**

```
┌─────────────┐
│ arg1: str   │
│ arg2: int   │
│ arg3: bool  │
└─────────────┘ ──▶ [function] ──▶ output
```

---

## Styling

Customize visualization appearance with `GraphvizStyle`:

```python
from pipeline_system.visualization import GraphvizStyle

style = GraphvizStyle(
    func_node_color="#FFB6C1",  # Light pink for functions
    arg_node_color="#98FB98",   # Pale green for inputs
    font_name="Arial",
    font_size=14,
    background_color="#F5F5F5"  # Light gray background
)

pipeline.visualize(style=style)
```

**Configurable style properties:**

- **Node colors**: `arg_node_color`, `func_node_color`, `grouped_args_node_color`
- **Edge colors**: `arg_edge_color`, `output_edge_color`, `grouped_args_edge_color`
- **Fonts**: `font_name`, `font_size`, `edge_font_size`, `legend_font_size`
- **Background**: `background_color`, `legend_background_color`

---

## Orientation

Control graph layout direction:

```python
pipeline.visualize(orient="TB")  # Top to Bottom (default)
pipeline.visualize(orient="LR")  # Left to Right
pipeline.visualize(orient="BT")  # Bottom to Top
pipeline.visualize(orient="RL")  # Right to Left
```

**Recommendations:**

- **TB (Top to Bottom)**: Best for deep pipelines with few parallel branches
- **LR (Left to Right)**: Best for wide pipelines with many parallel nodes
- **BT/RL**: Rarely used, for special layout requirements

---

## Legend

Display a legend explaining node types:

```python
pipeline.visualize(show_legend=True)
```

The legend shows:

- **Input**: External parameters (dashed rectangle)
- **Grouped Inputs**: Multiple params for one function (solid rectangle)
- **Function**: Computation nodes (rounded rectangle)

---

## Type Hints and Defaults

The visualization automatically extracts and displays:

**Type annotations:**

```python
@node(output_name="result")
def process(data: List[str], threshold: float = 0.5) -> Dict[str, Any]:
    ...
```

**Renders as:**

```
┌─────────────────────────────┐
│ data : List[str]            │
│ threshold : float = 0.5     │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│      process                │
├─────────────────────────────┤
│ result : Dict[str, Any]     │
└─────────────────────────────┘
```

**Type hint handling:**

- Extracts types using `get_type_hints()`
- Simplifies generic types (`typing.List[str]` → `List[str]`)
- Truncates long type strings to `MAX_LABEL_LENGTH` (30 chars)
- Shows default values from function signature

---

## Export Formats

Save visualizations to various formats:

```python
# SVG (vector, recommended)
pipeline.visualize(filename="pipeline.svg")

# PNG (raster)
pipeline.visualize(filename="pipeline.png")

# PDF (vector, for documents)
pipeline.visualize(filename="pipeline.pdf")

# DOT source (for manual editing)
pipeline.visualize(filename="[pipeline.dot](http://pipeline.dot)")
```

---

## Return Types

Control what the visualization returns:

```python
# Auto-detect (HTML in Jupyter, graphviz object otherwise)
result = pipeline.visualize()

# Force graphviz.Digraph object
digraph = pipeline.visualize(return_type="graphviz")

# Force HTML rendering (for Jupyter)
html = pipeline.visualize(return_type="html")
```

**In Jupyter notebooks:**

- Default returns `IPython.display.HTML` with embedded SVG
- SVG is responsive and scales with notebook width
- Interactive in some Jupyter environments

**In scripts:**

- Default returns `graphviz.Digraph` object
- Can be further manipulated or rendered
- Use `filename` parameter to save to disk

---

## Complete Example

```python
from pipeline_system import Pipeline, node
from pipeline_system.visualization import GraphvizStyle

# Define functions
@node(output_name="cleaned")
def clean(text: str) -> str:
    return text.strip().lower()

@node(output_name="tokens")
def tokenize(cleaned: str) -> List[str]:
    return cleaned.split()

@node(output_name="count")
def count_words(tokens: List[str]) -> int:
    return len(tokens)

# Build pipeline
pipeline = Pipeline(functions=[clean, tokenize, count_words])

# Visualize with custom style
style = GraphvizStyle(
    func_node_color="#B0E0E6",  # Powder blue
    arg_node_color="#F0E68C",   # Khaki
    font_size=14
)

pipeline.visualize(
    orient="LR",
    show_legend=True,
    style=style,
    filename="word_counter.svg"
)
```

---

## Nested Pipeline Visualization Example

```python
# Inner pipeline for encoding
@node(output_name="embedding")
def encode(text: str, model: Encoder) -> Vector:
    return model.encode(text)

encode_pipeline = Pipeline(
    functions=[clean, encode],
    backend=ModalBackend(gpu="A100")
)

# Outer pipeline using nested pipeline
@node(output_name="results")
def aggregate(embeddings: List[Vector]) -> Summary:
    return compute_summary(embeddings)

main_pipeline = Pipeline(
    functions=[encode_pipeline, aggregate],
    backend=LocalBackend()
)

# Collapsed view (encode_pipeline as single node)
main_pipeline.visualize(depth=1)

# Expanded view (show encode_pipeline internals)
main_pipeline.visualize(depth=2, flatten=False)

# Fully flattened view
main_pipeline.visualize(depth=2, flatten=True)
```

**Collapsed (depth=1):**

```
text ──▶ [encode_pipeline] ──▶ embeddings ──▶ [aggregate] ──▶ results
        [Modal, GPU: A100]                     [Local]
```

**Expanded, boxed (depth=2, flatten=False):**

```
                ┌─────────────────────────────────┐
text ──▶        │ encode_pipeline                 │
                │ [Modal, GPU: A100]              │
model ──▶       │                                 │
                │  text ──▶ [clean] ──▶ cleaned  │
                │  cleaned, model ──▶ [encode]   │
                └─────────────────────────────────┘
                             ↓
                        embeddings ──▶ [aggregate] ──▶ results
                                      [Local]
```

**Expanded, flattened (depth=2, flatten=True):**

```
text ──▶ [clean] ──▶ cleaned ──▶ [encode] ──▶ embeddings ──▶ [aggregate] ──▶ results
        [Modal GPU]   [Modal GPU]  [Modal GPU]               [Local]
model ───────────────────────────────┘
```

---

## Implementation Notes

### NetworkX Integration

The system uses NetworkX for graph representation:

- **Node attributes**: `node_type` ("input", "function", "grouped_args")
- **Edge attributes**: `param_name` (parameter name for the connection)
- Enables graph algorithms (topological sort, cycle detection, etc.)

### Graphviz Rendering

Graphviz handles the actual layout and rendering:

- Uses DOT language for graph specification
- HTML-like labels for rich formatting in nodes
- Cluster subgraphs for nested pipeline containers
- Multiple output formats (SVG, PNG, PDF)

### Unique Node Identification

Uses `id(node)` for unique node identifiers in Graphviz:

- Avoids naming collisions
- Supports duplicate node definitions
- Maps back to original node objects

---

## Related Documentation

- [Core Concepts](Core%20Concepts%204a4dd7402980462eb83fc2b3d5059ccc.md) - Understanding nodes and pipelines
- [Nested Pipelines](Nested%20Pipelines%20e1f81b1aceb749ba86d9079449edf976.md) - Hierarchical composition and `.as_node()`
- [Progress Visualization](Progress%20Visualization%20acf9de815df347f195c7eb98d79e72f8.md) - Runtime execution visualization
- [Tracing & Telemetry](Tracing%20&%20Telemetry%20da0bddf3d656448e99f2b968fd8c2b49.md) - Observability and debugging