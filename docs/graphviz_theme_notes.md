# Graphviz Theme Findings (Jan 2025)

- Scope: All changes limited to the Graphviz renderer (`src/hypernodes/viz/graphviz/renderer.py`); React/JS UI assets remain untouched.
- Palettes: Light mode now uses pastel/faded fills with deep-slate text; dark mode stays neon/contrasty. Cluster backgrounds follow the same split so nested pipelines stay legible.
- Theme source: Theme detection now relies solely on `theme_utils.detectHostTheme()` (VS Code data attributes / CSS vars). If it fails, it uses `parseColorString` from the same utility to infer luminance or falls back to `prefers-color-scheme`. The detected VS Code background is injected as `--hn-surface-bg` and applied to both container and SVG for true editor parity.
- Resilience: Theme application is wrapped in try/catch and silently no-ops on missing data. CSS variables are applied to both container and SVG to keep colors in sync even after re-render or mutation observer ticks.
- Verification: `uv run python tests/verify_graphviz_colors.py` checks for the updated light/dark palettes.

Notes:
- If you want an explicit assertion that container background equals the detected VS Code background, I can add a small Graphviz-specific test harness (mocking `theme_utils.detectHostTheme`).
