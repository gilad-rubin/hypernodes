# Theme Detection Guide

This guide documents the theme detection system for HyperNodes visualizations across different notebook environments.

## Overview

The visualization widget needs to detect:
1. **Host environment** (VS Code, JupyterLab, Marimo)
2. **Background color** (to seamlessly blend with notebook)
3. **Theme** (light/dark for appropriate node colors)

## Key Principle: Detect Host First

**CRITICAL**: Different notebook environments require different detection strategies. Always identify the host environment BEFORE attempting to read background colors.

```javascript
let hostEnv = 'unknown';
const parentDoc = window.parent?.document;
if (parentDoc) {
    // VS Code
    if (parentDoc.body.getAttribute('data-vscode-theme-kind') || 
        parentDoc.body.className?.includes('vscode')) {
        hostEnv = 'vscode';
    }
    // JupyterLab
    else if (parentDoc.body.dataset.jpThemeLight !== undefined || 
             parentDoc.querySelector('.jp-Notebook')) {
        hostEnv = 'jupyterlab';
    }
    // Marimo
    else if (parentDoc.body.dataset.theme || parentDoc.body.dataset.mode) {
        hostEnv = 'marimo';
    }
}
```

---

## Environment-Specific Detection

### VS Code

**Background Color**:
```javascript
const rootStyle = getComputedStyle(parentDoc.documentElement);
const bg = rootStyle.getPropertyValue('--vscode-editor-background');
// Returns: "#1e1e1e", "rgb(30, 30, 30)", etc.
```

**Theme**:
```javascript
const kind = parentDoc.body.getAttribute('data-vscode-theme-kind');
// Returns: "vscode-dark", "vscode-light", "vscode-high-contrast"
```

### JupyterLab

**Background Color** (priority order):
1. `.jp-Notebook` element (BEST - actual visible background)
2. `--jp-layout-color0` CSS variable (may return named color like "white")

```javascript
// Best source
const jpNotebook = parentDoc.querySelector('.jp-Notebook');
const bg = getComputedStyle(jpNotebook).backgroundColor;
// Light: rgb(255, 255, 255)
// Dark: rgb(17, 17, 17)

// Fallback
const rootStyle = getComputedStyle(parentDoc.documentElement);
const bg = rootStyle.getPropertyValue('--jp-layout-color0');
// Light: "white"
// Dark: "#111"
```

**Theme**:
```javascript
const jpThemeLight = parentDoc.body.dataset.jpThemeLight;
// "true" = light, "false" = dark

const themeName = parentDoc.body.dataset.jpThemeName;
// "JupyterLab Light" or "JupyterLab Dark"
```

### Marimo

**Theme**:
```javascript
const dataTheme = parentDoc.body.dataset.theme;
const dataMode = parentDoc.body.dataset.mode;
// Returns: "dark" or "light"

const colorScheme = getComputedStyle(parentDoc.documentElement)
    .getPropertyValue('color-scheme');
// Returns: "dark", "light", "dark light"
```

---

## Color Parsing

Named colors (like "white") need to be resolved to RGB for luminance calculation:

```javascript
function parseColorString(value) {
    if (!value) return null;
    const scratch = document.createElement('div');
    scratch.style.backgroundColor = value;
    document.body.appendChild(scratch);
    const resolved = getComputedStyle(scratch).backgroundColor;
    scratch.remove();
    
    const nums = resolved.match(/[\d.]+/g);
    if (nums && nums.length >= 3) {
        const [r, g, b] = nums.slice(0, 3).map(Number);
        const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
        return { r, g, b, luminance, resolved };
    }
    return null;
}
```

**Theme from luminance**: `luminance > 150 ? 'light' : 'dark'`

---

## Filtering Invalid Values

Always filter transparent/empty values:

```javascript
const pushCandidate = (value, source) => {
    if (value && value !== 'transparent' && value !== 'rgba(0, 0, 0, 0)') {
        attempts.push({ value: value.trim(), source });
    }
};
```

---

## Theme Toggle (2-State)

The widget uses a simple 2-state toggle:

| Current State | Icon | Click Action |
|--------------|------|--------------|
| Auto (detected dark) | ☀️ Sun | Switch to light + predefined bg |
| Auto (detected light) | 🌙 Moon | Switch to dark + predefined bg |
| Manual (any) | Opposite icon | Return to auto + notebook bg |

**Predefined backgrounds**:
- Light: `#f8fafc` (slate-50)
- Dark: `#020617` (slate-950)

---

## Testing

Use `notebooks/jupyterlab_theme_test.ipynb` to verify detection in JupyterLab:
- Test 1: CSS Variables
- Test 2: Body Data Attributes
- Test 3: Element Background Colors
- Test 6: Comprehensive Detection
- Test 7: Find Best Background Source

Each test shows color preview boxes for visual verification.

---

## Implementation Files

| File | Purpose |
|------|---------|
| `src/hypernodes/viz/js/html_generator.py` | Inline `detectHostTheme()` (~line 1232) |
| `src/hypernodes/viz/assets/theme_utils.js` | Standalone theme utilities |
| `notebooks/jupyterlab_theme_test.ipynb` | Detection test notebook |
| `.ruler/3-theme_detection.md` | Quick reference |

---

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Wrong bg in JupyterLab | Using CSS var instead of element | Query `.jp-Notebook` element |
| Wrong bg in VS Code | JupyterLab sources checked first | Detect host env first |
| Named color not parsed | "white" not converted to RGB | Use scratch element to resolve |
| Transparent picked | Not filtering invalid values | Check for `transparent`/`rgba(0,0,0,0)` |
