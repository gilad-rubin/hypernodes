# Pipeline Visualization Guide

## Overview

The HyperNodes visualization system provides powerful, customizable pipeline visualizations using Graphviz. It supports multiple design styles, hierarchical pipeline expansion, and extensive configuration options.

## Quick Start

```python
from hypernodes import node, Pipeline

@node(output_name="result")
def process(x: int) -> int:
    return x * 2

pipeline = Pipeline(nodes=[process])

# Display in Jupyter
pipeline.visualize()

# Save to file
pipeline.visualize(filename="my_pipeline.svg")
```

## Design Styles

Choose from 7 pre-built design styles:

### 1. **default** (Classic)
- Sky blue function nodes
- Light green input nodes
- Good general-purpose readability
- **Use for:** Documentation, tutorials

### 2. **minimal** (Clean & Simple)
- White/gray color scheme
- Thin borders
- Professional and uncluttered
- **Use for:** Technical documentation, code reviews

### 3. **vibrant** (Bold & Colorful)
- Bright, saturated colors
- Thick borders and edges
- High contrast
- **Use for:** Presentations, demos, marketing materials

### 4. **monochrome** (Grayscale)
- Full grayscale palette
- Print-friendly
- Courier font for technical look
- **Use for:** Academic papers, black & white printing

### 5. **dark** (Dark Mode)
- Dark blue background
- Light colored nodes and edges
- Easy on eyes in low light
- **Use for:** Dark-themed presentations, developer docs

### 6. **professional** (Corporate)
- Subtle blue/yellow/green tones
- Clean and business-appropriate
- Arial font
- **Use for:** Business reports, stakeholder presentations

### 7. **pastel** (Soft Colors)
- Gentle, muted colors
- Easy on the eyes
- Friendly appearance
- **Use for:** Educational materials, user-facing docs

## Usage Examples

### Basic Visualization

```python
# Default style
pipeline.visualize()

# Choose a style
pipeline.visualize(style="professional")

# With legend
pipeline.visualize(style="vibrant", show_legend=True)
```

### Orientation Options

```python
# Top to bottom (default) - good for deep pipelines
pipeline.visualize(orient="TB")

# Left to right - good for wide pipelines
pipeline.visualize(orient="LR")

# Also available: "BT" (bottom to top), "RL" (right to left)
```

### Hierarchical Pipelines

```python
# Collapsed view - nested pipelines as single nodes
pipeline.visualize(depth=1)

# Expand one level
pipeline.visualize(depth=2)

# Fully expanded - show all nested levels
pipeline.visualize(depth=None)
```

### Input Parameter Grouping

```python
# Group inputs used by single function (default, min 2 params)
pipeline.visualize(min_arg_group_size=2)

# No grouping - show all inputs separately
pipeline.visualize(min_arg_group_size=None)

# Higher threshold - require at least 3 params to group
pipeline.visualize(min_arg_group_size=3)
```

### Type Hints Display

```python
# With type hints (default)
pipeline.visualize(show_types=True)

# Without type hints - minimal view
pipeline.visualize(show_types=False)
```

### File Export

```python
# SVG (vector format, recommended)
pipeline.visualize(filename="pipeline.svg")

# PNG (raster format)
pipeline.visualize(filename="pipeline.png")

# PDF (for documents)
pipeline.visualize(filename="pipeline.pdf")

# DOT source (for manual editing)
pipeline.visualize(filename="pipeline.dot")
```

## Custom Styles

Create your own custom style:

```python
from hypernodes import GraphvizStyle

custom_style = GraphvizStyle(
    func_node_color="#FFE5B4",  # Peach
    arg_node_color="#E0BBE4",   # Lavender
    grouped_args_node_color="#D4E4BC",  # Sage
    arg_edge_color="#957DAD",   # Purple
    output_edge_color="#FFDAB9",  # Peach puff
    font_size=13,
    node_border_width=3,
    edge_width=2,
)

pipeline.visualize(style=custom_style)
```

### GraphvizStyle Options

```python
@dataclass
class GraphvizStyle:
    # Node colors
    func_node_color: str = "#87CEEB"  # Function nodes
    arg_node_color: str = "#90EE90"  # Input parameters
    grouped_args_node_color: str = "#90EE90"  # Grouped inputs
    
    # Edge colors
    arg_edge_color: str = "#666666"  # From inputs
    output_edge_color: str = "#333333"  # Between functions
    grouped_args_edge_color: str = "#666666"  # From groups
    
    # Typography
    font_name: str = "Helvetica"
    font_size: int = 12
    edge_font_size: int = 10
    legend_font_size: int = 11
    
    # Background
    background_color: str = "#FFFFFF"
    legend_background_color: str = "#F5F5F5"
    
    # Dimensions
    node_border_width: int = 2
    edge_width: int = 2
    node_padding: str = "0.3,0.2"
    
    # Cluster styling (for nested pipelines)
    cluster_border_color: str = "#999999"
    cluster_border_width: int = 2
    cluster_fill_color: str = "#F9F9F9"
```

## Complete API Reference

```python
pipeline.visualize(
    filename: Optional[str] = None,          # Output filename
    orient: Literal["TB", "LR", "BT", "RL"] = "TB",  # Graph direction
    depth: Optional[int] = 1,                # Nesting expansion depth
    flatten: bool = False,                   # Flatten nested containers
    min_arg_group_size: Optional[int] = 2,   # Input grouping threshold
    show_legend: bool = False,               # Display legend
    show_types: bool = True,                 # Show type hints
    style: Union[str, GraphvizStyle] = "default",  # Style name or object
    return_type: Literal["auto", "graphviz", "html"] = "auto",  # Return format
)
```

## Comparison Matrix

| Feature | Best For | Recommendation |
|---------|----------|----------------|
| **Styles** | | |
| default | General use | Tutorials, docs |
| minimal | Clean look | Code reviews |
| vibrant | Eye-catching | Presentations |
| monochrome | B&W printing | Academic papers |
| dark | Dark themes | Developer docs |
| professional | Business | Reports |
| pastel | Friendly | Education |
| **Orientation** | | |
| TB | Deep pipelines | < 5 parallel branches |
| LR | Wide pipelines | > 5 parallel branches |
| **Depth** | | |
| depth=1 | High-level view | Executive summaries |
| depth=2 | One level detail | Team discussions |
| depth=None | Full detail | Debugging |
| **Grouping** | | |
| Grouped (2+) | Clean look | Functions with many inputs |
| Ungrouped | Full detail | Simple pipelines |
| **Type Hints** | | |
| With types | Documentation | API docs, tutorials |
| Without types | Simplicity | Quick sketches |

## Examples Gallery

### Simple Pipeline

```python
from hypernodes import node, Pipeline
from typing import List

@node(output_name="cleaned")
def clean_text(text: str) -> str:
    return text.strip().lower()

@node(output_name="tokens")
def tokenize(cleaned: str) -> List[str]:
    return cleaned.split()

@node(output_name="count")
def count_words(tokens: List[str]) -> int:
    return len(tokens)

pipeline = Pipeline(nodes=[clean_text, tokenize, count_words])
```

**Visualizations:**
- `simple_default.svg` - Default style
- `simple_professional.svg` - Professional style
- `simple_vibrant.svg` - Vibrant style

### Hierarchical Pipeline

```python
# Inner pipeline
@node(output_name="preprocessed")
def preprocess(text: str) -> str:
    return text.lower()

inner = Pipeline(nodes=[preprocess])

# Outer pipeline
@node(output_name="final")
def finalize(preprocessed: str) -> str:
    return f"Result: {preprocessed}"

outer = Pipeline(nodes=[inner, finalize])
```

**Visualizations:**
- `hierarchical_depth_1.svg` - Collapsed
- `hierarchical_depth_2.svg` - Expanded one level
- `hierarchical_depth_full.svg` - Fully expanded

## Tips & Best Practices

### 1. Choose the Right Style
- **Presentations**: Use `vibrant` or `professional`
- **Documentation**: Use `default` or `minimal`
- **Dark themes**: Use `dark`
- **Printing**: Use `monochrome`

### 2. Optimize Orientation
- **Deep pipelines** (many sequential steps): Use `orient="TB"`
- **Wide pipelines** (many parallel branches): Use `orient="LR"`

### 3. Manage Complexity
- **Simple pipelines**: Show everything (`depth=None`, `min_arg_group_size=None`)
- **Complex pipelines**: Start collapsed (`depth=1`), expand as needed
- **Many inputs**: Use grouping (`min_arg_group_size=2`)

### 4. Type Hints
- **Enable** for documentation and tutorials
- **Disable** for high-level overviews or when space is limited

### 5. File Formats
- **SVG**: Best for web, scales perfectly
- **PNG**: For embedding in documents, presentations
- **PDF**: For reports, papers
- **DOT**: For manual tweaking in Graphviz tools

## Troubleshooting

### Issue: Visualization too large
**Solution:** 
- Use `depth=1` to collapse nested pipelines
- Enable input grouping: `min_arg_group_size=2`
- Disable type hints: `show_types=False`

### Issue: Hard to read labels
**Solution:**
- Increase font size in custom style
- Use `orient="LR"` for better horizontal space
- Try `minimal` style for cleaner look

### Issue: Colors don't match my theme
**Solution:**
- Create a custom `GraphvizStyle`
- Or choose from 7 built-in styles

## Generated Samples

Check the `outputs/` directory for 18 sample visualizations demonstrating:
- 5 different styles on simple pipeline
- 2 orientation options
- 3 depth levels for hierarchical pipeline
- 4 styles on hierarchical pipeline
- Type hints comparison
- Input grouping comparison

Run `uv run python scripts/generate_samples.py` to regenerate.

## Jupyter Notebook

Open `notebooks/visualization_showcase.ipynb` for an interactive showcase of all features!

## See Also

- [Pipeline Documentation](docs/)
- [Node Decorator Guide](docs/)
- [Hierarchical Pipelines](docs/)
