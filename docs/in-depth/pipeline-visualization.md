# Visualization

Render a pipeline graph with Graphviz:

```python
p.visualize()                 # Return graph object (HTML in notebooks)
p.visualize(filename="pipeline.svg")
p.visualize(depth=2, style="professional", show_legend=True)
```

Options include `orient`, `depth`, `flatten`, `min_arg_group_size`, `show_types`, and style presets in `hypernodes.visualization.DESIGN_STYLES`.
