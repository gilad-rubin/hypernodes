# VSCode Notebook JavaScript Compatibility Guide

## Overview

This guide documents critical compatibility requirements for running JavaScript visualizations inside VSCode notebook cells. These learnings apply to any ipywidget that embeds HTML/JS via iframes.

## The Problem

Interactive visualizations using React Flow (or any complex JS) may render correctly in:
- Standalone HTML files opened in a browser ✅
- JupyterLab ✅
- Jupyter Notebook ✅

But show **blank space** in VSCode notebooks ❌

## Root Causes & Solutions

### 1. ES Module Scripts Don't Work in iframe srcdoc

**Problem**: `<script type="module">` doesn't execute in iframe `srcdoc` contexts within VSCode's webview.

```html
<!-- ❌ DOESN'T WORK in VSCode notebooks -->
<script type="module">
  const data = JSON.parse(document.getElementById('data').textContent);
  // ... React code
</script>
```

**Why**: VSCode's notebook webview applies strict Content Security Policy (CSP) restrictions that block ES module execution in nested iframe contexts.

**Solution**: Use regular scripts with IIFE (Immediately Invoked Function Expression):

```html
<!-- ✅ WORKS in VSCode notebooks -->
<script>
  (function() {
    'use strict';
    const data = JSON.parse(document.getElementById('data').textContent);
    // ... React code
  })();
</script>
```

### 2. Regular Scripts Need DOM Ready

**Problem**: When switching from module to regular scripts, the script may execute before the DOM is fully parsed.

Module scripts are **automatically deferred** - they wait for HTML parsing to complete. Regular scripts execute **immediately** when encountered.

```html
<script>
  // ❌ This runs BEFORE the #data element exists!
  const data = JSON.parse(document.getElementById('data').textContent);
</script>
<!-- ... more HTML ... -->
<script id="data" type="application/json">{"nodes":[]}</script>
```

**Error**: `Cannot read properties of null (reading 'textContent')`

**Solution**: Wrap in `DOMContentLoaded` event listener:

```html
<script>
  document.addEventListener('DOMContentLoaded', function() {
    (function() {
      'use strict';
      // ✅ DOM is now fully loaded
      const data = JSON.parse(document.getElementById('data').textContent);
      // ... React code
    })();
  });
</script>
```

### 3. Base64 Data URIs Don't Work

**Problem**: iframe `src="data:text/html;base64,..."` may be blocked.

```python
# ❌ May not work in VSCode
html_b64 = base64.b64encode(html_content.encode()).decode()
iframe_html = f'<iframe src="data:text/html;base64,{html_b64}" ...>'
```

**Solution**: Use `srcdoc` attribute with HTML-escaped content:

```python
# ✅ Works in VSCode
import html
escaped_html = html.escape(html_content, quote=True)
iframe_html = f'<iframe srcdoc="{escaped_html}" ...>'
```

### 4. External CDN Scripts May Be Blocked

**Problem**: Scripts loaded from external CDNs require network access which may be restricted.

```html
<!-- ⚠️ May fail if network restricted -->
<script src="https://cdn.jsdelivr.net/npm/react@18/umd/react.production.min.js"></script>
```

**Solution**: Bundle scripts inline or use local vendored assets:

```python
def _read_asset(name: str) -> str:
    path = Path(__file__).parent / "assets" / name
    return path.read_text() if path.exists() else None

react_js = _read_asset("react.production.min.js")
html = f"<script>{react_js}</script>" if react_js else "<script src='cdn...'></script>"
```

## Complete Pattern

Here's the complete pattern that works across all environments:

```html
<!DOCTYPE html>
<html>
<head>
    <!-- Inline CSS or vendored CSS -->
    <style>/* styles */</style>
    
    <!-- Vendored JS libraries (preferred) or CDN fallback -->
    <script>/* React UMD */</script>
    <script>/* ReactDOM UMD */</script>
    <script>/* Other libs */</script>
</head>
<body>
    <div id="root">Loading...</div>
    
    <!-- Main app script - regular script with DOMContentLoaded -->
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            (function() {
                'use strict';
                
                // Safe to access DOM now
                const data = JSON.parse(
                    document.getElementById('graph-data').textContent
                );
                
                // React app initialization
                const root = ReactDOM.createRoot(document.getElementById('root'));
                root.render(/* ... */);
                
            })();
        });
    </script>
    
    <!-- Data element MUST be in the HTML (can be before or after script with DOMContentLoaded) -->
    <script id="graph-data" type="application/json">{"nodes":[],"edges":[]}</script>
</body>
</html>
```

## Testing Checklist

When making changes to the visualization HTML generator, verify:

1. **No module scripts**: `grep -o '<script[^>]*type="module"' output.html` should return nothing
2. **Has DOMContentLoaded**: `grep 'DOMContentLoaded' output.html` should find the wrapper
3. **Has IIFE**: `grep "(function()" output.html` should find the pattern
4. **Playwright test passes**: React Flow nodes should render in headless browser

## Debugging

### Quick Diagnosis Script

```python
from hypernodes.viz.visualization_widget import PipelineWidget
import html as html_module
import re

widget = PipelineWidget(pipeline)
decoded = html_module.unescape(widget.value)

# Check for issues
checks = {
    "No module scripts": not bool(re.search(r'<script\s+type=["\']module["\']', decoded)),
    "Has IIFE": '(function()' in decoded and "'use strict'" in decoded,
    "Has DOMContentLoaded": 'DOMContentLoaded' in decoded,
    "Uses srcdoc": 'srcdoc=' in widget.value,
}

for check, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"{status} {check}")
```

### Browser Console Debugging

If visualization shows blank, check browser console (F12) for:
- `Cannot read properties of null` → DOM not ready, need DOMContentLoaded
- `Failed to load module script` → Module scripts blocked, use regular script
- `net::ERR_BLOCKED_BY_CLIENT` → External scripts blocked, use vendored

### Diagnostic Notebook

Use `notebooks/debug_widget_vscode.ipynb` to test individual layers:
- Cell 4: iframe with srcdoc (no JS)
- Cell 5: iframe with inline regular script
- Cell 6: iframe with module script (will fail in VSCode)
- Cell 7: iframe with external CDN scripts

## Key Files

- `src/hypernodes/viz/js/html_generator.py` - Main HTML generation
- `src/hypernodes/viz/visualization_widget.py` - Widget wrapper with iframe
- `tests/viz/test_vscode_compatibility.py` - Regression tests
- `notebooks/debug_widget_vscode.ipynb` - Interactive diagnostic notebook

## Historical Context

- **Nov 2025**: Fixed blank visualization in VSCode by:
  1. Changing from `<script type="module">` to regular `<script>`
  2. Wrapping code in `DOMContentLoaded` event listener
  3. Using IIFE pattern for strict mode and scope isolation
