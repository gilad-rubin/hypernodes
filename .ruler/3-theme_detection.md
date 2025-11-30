# Theme Detection Patterns

This document describes the working theme detection methods for each notebook environment.

## Host Environment Detection

**IMPORTANT**: Detect the host environment FIRST, then use environment-specific methods for background color.

```javascript
let hostEnv = 'unknown';
const parentDoc = window.parent?.document;
if (parentDoc) {
    // VS Code detection
    if (parentDoc.body.getAttribute('data-vscode-theme-kind') || 
        (parentDoc.body.className && parentDoc.body.className.includes('vscode'))) {
        hostEnv = 'vscode';
    }
    // JupyterLab detection
    else if (parentDoc.body.dataset.jpThemeLight !== undefined || 
             parentDoc.querySelector('.jp-Notebook')) {
        hostEnv = 'jupyterlab';
    }
    // Marimo detection
    else if (parentDoc.body.dataset.theme || parentDoc.body.dataset.mode ||
             (parentDoc.body.className && parentDoc.body.className.includes('marimo'))) {
        hostEnv = 'marimo';
    }
}
```

---

## VS Code Notebook

### Background Color (Reliable)
```javascript
const style = getComputedStyle(window.parent.document.documentElement);
const bg = style.getPropertyValue('--vscode-editor-background');
// Returns actual background color like "rgb(30, 30, 30)" or "#1e1e1e"
```

### Theme Detection
```javascript
const kind = window.parent.document.body.getAttribute('data-vscode-theme-kind');
// Returns "vscode-dark", "vscode-light", or "vscode-high-contrast"
```

Also works: `vscode-light`/`vscode-dark` classes on body

---

## JupyterLab

### Background Color (Reliable)

**Best source**: `.jp-Notebook` element background
```javascript
const jpNotebook = parentDoc.querySelector('.jp-Notebook');
const bg = getComputedStyle(jpNotebook).backgroundColor;
// Light: rgb(255, 255, 255)
// Dark: rgb(17, 17, 17)
```

**Fallback**: CSS variables (these may return named colors like "white")
```javascript
const rootStyle = getComputedStyle(parentDoc.documentElement);
const bg = rootStyle.getPropertyValue('--jp-layout-color0');
// Light: "white" 
// Dark: "#111"
```

### Theme Detection
```javascript
const jpThemeLight = parentDoc.body.dataset.jpThemeLight;
// Returns "true" for light theme, "false" for dark theme

const themeName = parentDoc.body.dataset.jpThemeName;
// Returns "JupyterLab Light" or "JupyterLab Dark"
```

### JupyterLab CSS Variables Reference
| Variable | Light | Dark |
|----------|-------|------|
| `--jp-layout-color0` | white | #111 |
| `--jp-layout-color1` | white | #212121 |
| `--jp-layout-color2` | #eee | #424242 |
| `--jp-cell-editor-background` | #f5f5f5 | #212121 |
| `--jp-content-font-color1` | rgba(0,0,0,0.87) | rgba(255,255,255,1) |

**Note**: `--jp-notebook-background` does NOT exist (returns empty)

---

## Marimo Notebook

### Theme Detection
```javascript
const dataTheme = parentDoc.body.dataset.theme || parentDoc.documentElement.dataset.theme;
const dataMode = parentDoc.body.dataset.mode || parentDoc.documentElement.dataset.mode;
// Returns "dark" or "light"

// Body classes
const bodyClass = parentDoc.body.className;
// Check for "dark-mode" or "dark" class

// color-scheme CSS property
const colorScheme = getComputedStyle(parentDoc.documentElement).getPropertyValue('color-scheme');
// Returns "dark", "light", "dark light", etc.
```

---

## Theme Toggle Behavior (2-State)

The widget implements a 2-state toggle:

1. **Auto/Detected** (default) → Uses detected notebook background color + theme
2. **Manual/Opposite** → Uses predefined background for opposite theme

Icons:
- In dark mode → Show ☀️ (sun) icon → Click switches to light with predefined bg
- In light mode → Show 🌙 (moon) icon → Click switches to dark with predefined bg
- Click again → Returns to auto mode with notebook's actual background

Predefined backgrounds:
- Light: `#f8fafc` (slate-50)
- Dark: `#020617` (slate-950)

---

## Filtering Invalid Colors

Always filter out transparent values when building candidate list:
```javascript
if (value && value !== 'transparent' && value !== 'rgba(0, 0, 0, 0)') {
    attempts.push({ value, source });
}
```

---

## Implementation Files

- `src/hypernodes/viz/js/html_generator.py` - Inline `detectHostTheme()` function (~line 1232)
- `src/hypernodes/viz/assets/theme_utils.js` - Standalone `detectHostTheme()` function

