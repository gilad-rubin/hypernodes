# Theme Detection Patterns

This document describes the working theme detection methods for each notebook environment.

## Detection Priority Order

1. **JupyterLab** - `data-jp-theme-light` attribute, `jp-mod-dark`/`jp-mod-light` classes
2. **VS Code** - `data-vscode-theme-kind` attribute, `vscode-light`/`vscode-dark` classes
3. **Marimo** - `data-theme`/`data-mode` attributes, `dark-mode`/`dark` classes, `color-scheme` CSS property
4. **Background Luminance** - Parse background color and calculate luminance
5. **prefers-color-scheme** - Media query fallback

---

## VS Code Notebook

### Working Methods

**Trial 5: Parent CSS Variables** - Works for background color!
```javascript
const style = getComputedStyle(window.parent.document.documentElement);
const bg = style.getPropertyValue('--vscode-editor-background');
// Returns actual background color like "rgb(30, 30, 30)" or "#1e1e1e"
```

**Trial 6: Body Attribute** - Works for light/dark theme!
```javascript
const kind = window.parent.document.body.getAttribute('data-vscode-theme-kind');
// Returns "vscode-dark", "vscode-light", or "vscode-high-contrast"
// Just search for "dark" or "light" in the text
```

Also works: `vscode-light`/`vscode-dark` classes on body

---

## JupyterLab

### Working Methods

**Method 2: Parent Body Classes/Attributes** - Works!
```javascript
const parentDoc = window.parent?.document;
// JupyterLab uses data-jp-theme-light attribute
const jpThemeLight = parentDoc.body.dataset.jpThemeLight;
// Returns "true" for light theme, "false" for dark theme

// Also check body classes
const bodyClass = parentDoc.body.className;
// May contain "jp-mod-dark" or "jp-mod-light"
```

**Background Color** - Also works:
```javascript
const bodyStyle = getComputedStyle(parentDoc.body);
const bg = bodyStyle.backgroundColor;
// Returns actual background color, can calculate luminance
```

---

## Marimo Notebook

### Working Methods

**Data Attributes**:
```javascript
const parentDoc = window.parent?.document;
const dataTheme = parentDoc.body.dataset.theme || parentDoc.documentElement.dataset.theme;
const dataMode = parentDoc.body.dataset.mode || parentDoc.documentElement.dataset.mode;
// Returns "dark" or "light"
```

**Body Classes**:
```javascript
const bodyClass = parentDoc.body.className;
// Check for "dark-mode" or "dark" class
```

**color-scheme CSS Property**:
```javascript
const colorScheme = getComputedStyle(parentDoc.documentElement).getPropertyValue('color-scheme');
// Returns "dark", "light", "dark light", etc.
```

---

## Theme Toggle Behavior

The widget implements a 3-state toggle cycle:

1. **Auto** (default) → Uses detected notebook background color + theme
2. **Light** → Uses predefined light background (`#f8fafc`)
3. **Dark** → Uses predefined dark background (`#020617`)
4. Back to **Auto** → Returns to dynamic notebook background

This allows users to:
- Start with the native notebook theme (seamless integration)
- Toggle to opposite theme for visibility
- Toggle again to predefined alternative
- Return to auto mode to sync with notebook again

