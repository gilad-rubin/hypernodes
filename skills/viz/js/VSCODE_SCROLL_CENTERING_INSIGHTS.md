# VS Code Notebook Scroll & Centering Insights

## Problem Statement

In VS Code notebooks, the pipeline visualization has two issues:
1. **Scroll blocking**: When the mouse is over the iframe, scroll events are captured by the iframe instead of scrolling the notebook
2. **Centering**: The visualization needs to be centered in the notebook cell

In JupyterLab, both of these work perfectly without any special handling.

---

## Working Solution: ScrollablePipelineWidget

The `ScrollablePipelineWidget` class in `visualization_widget.py` implements a working solution using an **overlay approach**:

```python
# Key structure:
<div id="wrapper" style="position: relative; width: Xpx; height: Ypx; margin: 0 auto;">
    <iframe srcdoc="..." style="position: absolute; ..."></iframe>
    <div id="overlay" style="position: absolute; z-index: 10;"></div>
</div>
<script>
    // Click overlay → disable pointer-events (enable iframe interaction)
    // Mouse leave wrapper → re-enable overlay (enable scroll passthrough)
</script>
```

### How it works:
1. **Overlay blocks iframe by default** - scroll events pass through to notebook
2. **Click to interact** - clicking the overlay disables its pointer-events, allowing iframe interaction
3. **Mouse leave re-enables scroll** - leaving the wrapper area re-enables the overlay

---

## Centering Insights

### What Works for Centering

All three methods work in VS Code notebooks for simple divs:

```html
<!-- Method 1: margin auto (WORKS) -->
<div style="width: 200px; margin: 0 auto;">...</div>

<!-- Method 2: text-align + inline-block (WORKS) -->
<div style="text-align: center;">
    <div style="display: inline-block;">...</div>
</div>

<!-- Method 3: flexbox (WORKS) -->
<div style="display: flex; justify-content: center;">
    <div>...</div>
</div>
```

### What Breaks Centering

**Dynamic width modification breaks centering!**

When JavaScript modifies `wrapper.style.width` directly (e.g., in a resize handler), the `margin: 0 auto` centering stops working:

```javascript
// THIS BREAKS CENTERING:
window.addEventListener('message', function(event) {
    if (event.data.type === 'hypernodes-viz-resize') {
        wrapper.style.width = newWidth + 'px';  // ❌ Breaks margin: 0 auto
    }
});
```

**Solution**: Only modify height, not width:

```javascript
// THIS PRESERVES CENTERING:
window.addEventListener('message', function(event) {
    if (event.data.type === 'hypernodes-viz-resize') {
        wrapper.style.height = newHeight + 'px';  // ✅ OK
        iframe.style.height = newHeight + 'px';   // ✅ OK
        // Don't touch width!
    }
});
```

### Whitespace in HTML Matters

Leading whitespace/newlines in f-strings can create text nodes that affect layout:

```python
# ❌ BAD - has leading newline and whitespace
css_fix = """
        <style>...</style>
        """
iframe_html = f'''
{css_fix}
<div>...</div>
'''

# ✅ GOOD - no leading whitespace
css_fix = """<style>...</style>"""
iframe_html = f'''{css_fix}
<div>...</div>
'''
```

---

## Animation/Glitch Insights

### fitView Causes Animation Glitch

React Flow's `fitView` with `duration > 0` causes a sliding animation:

```javascript
// ❌ Causes sliding animation
fitView({ padding: 0.1, duration: 200 });

// ✅ Instant, no animation
fitView({ padding: 0.1, duration: 0 });
```

### Multiple fitView Calls Cause Jumps

Having multiple `fitView` calls (even with `duration: 0`) in useEffects can cause layout jumps:

```javascript
// ❌ BAD - causes jump
useEffect(() => {
    fitView({ duration: 0 });
    setTimeout(() => fitView({ duration: 200 }), 100);  // Second call causes jump
}, [layoutedNodes]);

// ✅ GOOD - single fitView on ReactFlow component
<ReactFlow fitView fitViewOptions={{ duration: 0 }} />
```

### "No nodes" Flash

The error message "Layout produced no nodes" can flash briefly during initial render. Fix by checking `isLayouting`:

```javascript
// ❌ Shows error during initial layout
${(layoutError || !layoutedNodes.length) ? html`<Error />` : null}

// ✅ Only show error after layout completes
${(!isLayouting && (layoutError || !layoutedNodes.length)) ? html`<Error />` : null}
```

---

## Implementation Checklist

To implement VS Code scroll compatibility for `pipeline.visualize()`:

1. **Add overlay div** over the iframe
2. **Add click handler** to disable overlay pointer-events
3. **Add mouseleave handler** on wrapper to re-enable overlay
4. **Use `margin: 0 auto`** for centering (with `display: block`)
5. **Only modify height** in resize handlers, not width
6. **Remove leading whitespace** from HTML template strings
7. **Set `fitView` duration to 0** everywhere
8. **Check `isLayouting`** before showing error messages

---

## Files to Modify

- `src/hypernodes/viz/__init__.py` - `_render_interactive()` function
- `src/hypernodes/viz/js/html_generator.py` - fitView options
- `src/hypernodes/viz/visualization_widget.py` - PipelineWidget class

---

## Testing

1. Run `pipeline.visualize()` in VS Code notebook
2. Verify:
   - [ ] Visualization is centered
   - [ ] Scrolling over visualization scrolls notebook
   - [ ] Click enables interaction (pan/zoom)
   - [ ] Mouse leave re-enables scroll
   - [ ] No animation/sliding on load
   - [ ] No error flash on load
   - [ ] Expand/collapse adjusts height without scrollbar
