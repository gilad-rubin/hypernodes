# Visualization Best Practices & Lessons Learned

## Key Principles

1.  **Frontend Agnostic Serialization**: Always rely on `GraphSerializer` for computing graph structure (nodes, edges, hierarchy). The rendering engine (Graphviz, Widget) should only handle display concerns.
2.  **Nested Pipeline Visibility**:
    *   Inputs to nested pipelines must be visualized at the correct scope (root vs nested).
    *   Use `GraphSerializer.input_levels` to determine where an input node belongs.
    *   Ensure `parentNode` in React Flow or clusters in Graphviz respect this hierarchy.
3.  **Interactive Widget Resilience**:
    *   **Error Handling**: Javascript errors inside the iframe must be caught and surfaced to the user (not silently fail).
    *   **Fallback**: Always provide a fallback UI if layout calculation fails (e.g., ELK failure).
    *   **State Management**: Expansion state of nested nodes must be robust. If a parent collapses, all children (even grandchildren) must be hidden.
4.  **Layout & Styling**:
    *   **Auto-Centering**: Trigger `fitView` on resize and layout changes.
    *   **Theme Matching**: Detect host environment (VSCode) theme and background to blend in seamlessly.
    *   **Edge Routing**: Use orthogonal or spline routing to avoid edges crossing *behind* nodes.
    *   **Readability**: Avoid dashed lines for standard data flow. Group inputs logically.
5.  **Verification**:
    *   Always verify visualization changes by generating the HTML and inspecting it with a browser (or Playwright tool).
    *   Check for console errors in the generated HTML.
    *   Verify correct nesting of nodes (parents vs children).

## Common Pitfalls

*   **Graphviz vs Widget Parity**: Logic for grouping, filtering, and scoping often diverges. Use `GraphSerializer` as the single source of truth.
*   **Unused Outputs**: Pipelines may declare outputs that aren't used in a specific context. Visualization should filter these to reduce noise, unless explicitly requested.
*   **Iframe Communication**: Communication between the notebook kernel and the iframe is one-way (HTML injection). State does not persist across re-renders unless handled carefully.

## How to Verify

1.  Create a reproduction script (like `tests/repro_viz_issue.py`) that constructs a representative pipeline.
2.  Generate the HTML artifact using `PipelineWidget._generate_html()`.
3.  Open the HTML in a browser and visually inspect:
    *   Node placement (nesting).
    *   Edge connections (especially across boundaries).
    *   Console for errors.






