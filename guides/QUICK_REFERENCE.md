# Pipeline Visualization - Quick Reference

## 🚀 One-Line Usage

```python
pipeline.visualize()  # Display in Jupyter
pipeline.visualize(filename="pipeline.svg")  # Save to file
```

## 🎨 Choose Your Style

```python
pipeline.visualize(style="default")       # ⭐ Classic blue & green
pipeline.visualize(style="minimal")       # 🤍 Clean white/gray
pipeline.visualize(style="professional")  # 💼 Corporate look
pipeline.visualize(style="vibrant")       # 🌈 Bold & colorful
pipeline.visualize(style="pastel")        # 🎀 Soft colors
pipeline.visualize(style="dark")          # 🌙 Dark mode
pipeline.visualize(style="monochrome")    # ⚫ Grayscale
```

## 📐 Common Configurations

### For Documentation
```python
pipeline.visualize(
    style="professional",
    orient="LR",
    show_legend=True,
    filename="docs/pipeline.svg"
)
```

### For Presentations
```python
pipeline.visualize(
    style="vibrant",
    show_types=False,
    show_legend=True,
    filename="slides/pipeline.png"
)
```

### For Debugging
```python
pipeline.visualize(
    style="minimal",
    depth=None,  # Expand all
    min_arg_group_size=None,  # Don't group
    show_types=True
)
```

## 🔧 Key Parameters

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `style` | See above | `"default"` | Color scheme |
| `orient` | `"TB"`, `"LR"`, `"BT"`, `"RL"` | `"TB"` | Layout direction |
| `depth` | `1`, `2`, ... `None` | `1` | Nesting level |
| `show_types` | `True`, `False` | `True` | Type hints |
| `show_legend` | `True`, `False` | `False` | Show legend |
| `min_arg_group_size` | `2`, `3`, ... `None` | `2` | Group inputs |
| `filename` | `"path.svg"` | `None` | Save to file |

## 📊 View Comparisons

```bash
# Open comparison page in browser
open outputs/compare.html

# Or generate fresh samples
uv run python scripts/generate_samples.py
```

## 📚 Full Documentation

- **Complete Guide:** `VISUALIZATION_GUIDE.md`
- **Implementation Summary:** `VISUALIZATION_SUMMARY.md`
- **Interactive Demo:** `notebooks/visualization_showcase.ipynb`

## 💡 Quick Tips

1. **Wide pipelines?** Use `orient="LR"`
2. **Many inputs?** Use `min_arg_group_size=2`
3. **Nested pipelines?** Start with `depth=1`, expand as needed
4. **Presentations?** Use `style="vibrant"` or `style="professional"`
5. **Print?** Use `style="monochrome"` and save as PDF

---

**Try it now!** All 18 sample visualizations are in `outputs/` directory.
