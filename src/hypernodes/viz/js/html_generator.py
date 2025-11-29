import json
from pathlib import Path
from typing import Any, Dict, Optional


def generate_widget_html(graph_data: Dict[str, Any]) -> str:
    """Generate an HTML document for React Flow rendering.

    Uses local vendored JS/CSS assets (assets/viz/*). Falls back to remote CDNs
    only if a file is missing at runtime.
    """

    graph_json = json.dumps(graph_data)

    def _read_asset(name: str, kind: str) -> Optional[str]:
        """Read an asset file from assets/viz; return None if missing."""
        try:
            # Adjust path relative to this file: src/hypernodes/viz/js/html_generator.py
            # Assets are in: assets/viz
            # So we go up: js -> viz -> hypernodes -> src -> root -> assets
            path = (
                Path(__file__).resolve().parent.parent.parent.parent.parent
                / "assets"
                / "viz"
                / name
            )
            if not path.exists():
                return None
            text = path.read_text(encoding="utf-8")
            if kind == "js":
                return f"<script>{text}</script>"
            if kind == "css":
                return f"<style>{text}</style>"
        except Exception:
            return None
        return None

    react_js = _read_asset("react.production.min.js", "js")
    react_dom_js = _read_asset("react-dom.production.min.js", "js")
    htm_js = _read_asset("htm.min.js", "js")
    elk_js = _read_asset("elk.bundled.js", "js")
    rf_js = _read_asset("reactflow.umd.js", "js")
    rf_css = _read_asset("reactflow.css", "css")
    theme_js = _read_asset("theme_utils.js", "js")
    state_js = _read_asset("state_utils.js", "js")

    # If local assets are missing, keep a minimal external fallback.
    fallback_css = '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reactflow@11.10.1/dist/style.css" />'
    fallback_js = """
    <script src="https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.production.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.production.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/htm@3.1.1/dist/htm.umd.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/elkjs@0.8.2/lib/elk.bundled.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/reactflow@11.10.1/dist/umd/index.js"></script>
    """

    # Build HTML header with Python string interpolation
    html_head = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <script src="https://cdn.tailwindcss.com"></script>
    {rf_css or fallback_css}
    {_read_asset("custom.css", "css") or ""}
    <style>
        /* Reset and Base Styles */
        body {{ margin: 0; overflow: hidden; background: transparent; color: #e5e7eb; font-family: 'Inter', system-ui, -apple-system, sans-serif; }}
        .react-flow__attribution {{ display: none; }}
        #root {{ height: 100vh; width: 100vw; background: transparent; display: flex; align-items: flex-start; justify-content: center; }}
        #fallback {{ font-size: 13px; letter-spacing: 0.4px; color: #94a3b8; }}
        
        /* Canvas Outline */
        .canvas-outline {{
            outline: 1px dashed rgba(148, 163, 184, 0.2);
            margin: 2px;
            height: calc(100vh - 4px);
            width: calc(100vw - 4px);
            border-radius: 8px;
            pointer-events: none;
            position: absolute;
            top: 0;
            left: 0;
            z-index: 50;
        }}
        
        /* Function Node Light Mode Fix */
        .node-function-light {{
            border-bottom-width: 1px !important; /* Prevent artifact */
        }}
    </style>
    <!-- Prefer local vendored assets; fall back to CDN if missing -->
    {react_js or ""}
    {react_dom_js or ""}
    {htm_js or ""}
    {elk_js or ""}
    {rf_js or ""}
    {theme_js or ""}
    {state_js or ""}
    {fallback_js if not all([react_js, react_dom_js, htm_js, elk_js, rf_js]) else ""}
</head>"""

    # JavaScript body
    html_body = r"""<body>
  <div id="root">
    <div id="fallback">Rendering interactive view…</div>
  </div>
  <div class="canvas-outline"></div>
  <script>
    window.onerror = function(message, source, lineno, colno, error) {
      const el = document.getElementById("fallback");
      if (el) {
        el.textContent = "Viz error: " + message + (source ? " (" + source + ":" + lineno + ")" : "");
        el.style.color = "#f87171";
        el.style.fontFamily = "monospace";
      }
    };
  </script>
  <script type="module">
    const fallback = document.getElementById("fallback");
    const fail = (msg) => {
      if (fallback) {
        fallback.innerHTML = `
            <div style="display: flex; flex-direction: column; gap: 8px; max-width: 80%;">
                <div style="color: #f87171; font-family: monospace; user-select: text; background: #2a1b1b; padding: 12px; rounded: 4px;">${msg}</div>
                <button onclick="navigator.clipboard.writeText(this.previousElementSibling.innerText)" style="padding: 4px 8px; background: #374151; border: none; color: white; border-radius: 4px; cursor: pointer; align-self: flex-start;">Copy Error</button>
                <button onclick="window.location.reload()" style="margin-top: 8px; padding: 6px 12px; background: #2563eb; border: none; color: white; border-radius: 4px; cursor: pointer; align-self: flex-start;">Retry Visualization</button>
            </div>
        `;
      }
    };

    // Keep-alive mechanism to prevent iframe cleanup in some environments
    setInterval(() => {
      try {
        // Minimal DOM interaction to keep the context alive
        document.documentElement.dataset.lastPing = Date.now();
      } catch(e) {}
    }, 5000);

    try {
      const React = window.React;
      const ReactDOM = window.ReactDOM;
      const RF = window.ReactFlow;
      const htm = window.htm;
      const ELK = window.ELK;
      const stateUtils = window.HyperNodesVizState || {};
      const themeUtils = window.HyperNodesTheme || {};

      if (!React || !ReactDOM || !RF || !htm || !ELK) {
        throw new Error("Missing globals: " + JSON.stringify({
          React: !!React, ReactDOM: !!ReactDOM, ReactFlow: !!RF, htm: !!htm, ELK: !!ELK
        }));
      }

      const { ReactFlow, Background, Controls, MiniMap, Handle, Position, ReactFlowProvider, useEdgesState, useNodesState, MarkerType, BaseEdge, getBezierPath, getSmoothStepPath, EdgeLabelRenderer, useReactFlow, Panel, useUpdateNodeInternals } = RF;
      const { useState, useEffect, useMemo, useCallback, useRef } = React;

      const html = htm.bind(React.createElement);
      const elk = new ELK();
      const fallbackApplyState = (baseNodes, baseEdges, options) => {
        const { expansionState, separateOutputs, showTypes, theme } = options;
        const expMap = expansionState instanceof Map ? expansionState : new Map(Object.entries(expansionState || {}));

        const applyMeta = (nodeList) => nodeList.map((n) => {
          const isPipeline = n.data?.nodeType === 'PIPELINE';
          const expanded = isPipeline ? Boolean(expMap.get(n.id)) : undefined;
          return {
            ...n,
            type: isPipeline && expanded ? 'pipelineGroup' : n.type,
            style: isPipeline && !expanded ? undefined : n.style,
            data: {
              ...n.data,
              theme,
              showTypes,
              isExpanded: expanded,
            },
          };
        });

        if (separateOutputs) {
          return {
            nodes: applyMeta(baseNodes).map((n) => ({
              ...n,
              data: { ...n.data, separateOutputs: true },
            })),
            edges: baseEdges,
          };
        }

        const outputNodes = new Set(baseNodes.filter((n) => n.data?.sourceId).map((n) => n.id));
        const functionOutputs = {};
        baseNodes.forEach((n) => {
          if (n.data?.sourceId) {
            if (!functionOutputs[n.data.sourceId]) functionOutputs[n.data.sourceId] = [];
            functionOutputs[n.data.sourceId].push({ name: n.data.label, type: n.data.typeHint });
          }
        });

        const nodes = applyMeta(baseNodes)
          .filter((n) => !outputNodes.has(n.id))
          .map((n) => ({
            ...n,
            data: {
              ...n.data,
              separateOutputs: false,
              outputs: functionOutputs[n.id] || [],
            },
          }));

        const edges = baseEdges
          .filter((e) => !outputNodes.has(e.target))
          .map((e) => {
            if (outputNodes.has(e.source)) {
              const outputNode = baseNodes.find((n) => n.id === e.source);
              if (outputNode?.data?.sourceId) {
                return { ...e, id: `e_${outputNode.data.sourceId}_${e.target}`, source: outputNode.data.sourceId };
              }
            }
            return e;
          });

        return { nodes, edges };
      };

      const fallbackApplyVisibility = (nodes, expansionState) => {
        const expMap = expansionState instanceof Map ? expansionState : new Map(Object.entries(expansionState || {}));
        const parentMap = new Map();
        nodes.forEach((n) => {
          if (n.parentNode) parentMap.set(n.id, n.parentNode);
        });

        const isHidden = (nodeId) => {
          let curr = nodeId;
          while (curr) {
            const parent = parentMap.get(curr);
            if (!parent) return false;
            if (expMap.get(parent) === false) return true;
            curr = parent;
          }
          return false;
        };

        return nodes.map((n) => ({ ...n, hidden: isHidden(n.id) }));
      };

      // --- Icons ---
      const Icons = {
        Moon: () => html`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>`,
        Sun: () => html`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>`,
        ZoomIn: () => html`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>`,
        ZoomOut: () => html`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="8" y1="11" x2="14" y2="11"/></svg>`,
        Center: () => html`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>`,
        Function: () => html`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect><line x1="7" y1="2" x2="7" y2="22"></line><line x1="17" y1="2" x2="17" y2="22"></line><line x1="2" y1="12" x2="22" y2="12"></line></svg>`,
        Pipeline: () => html`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>`,
        Dual: () => html`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><path d="M12 2a10 10 0 1 0 10 10H12V2z"></path><path d="M12 12L2 12"></path><path d="M12 12L12 22"></path></svg>`,
        Input: () => html`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="16"></line><line x1="8" y1="12" x2="16" y2="12"></line></svg>`,
        Data: () => html`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-3 h-3"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><line x1="10" y1="9" x2="8" y2="9"></line></svg>`,
        Map: () => html`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"></polygon><line x1="8" y1="2" x2="8" y2="18"></line><line x1="16" y1="6" x2="16" y2="22"></line></svg>`
      };

      // --- Custom Controls ---
      const CustomControls = ({ theme, onToggleTheme, showMiniMap, onToggleMiniMap }) => {
        const { zoomIn, zoomOut, fitView, setCenter } = useReactFlow();
        
        const btnClass = `p-2 rounded-lg shadow-lg border transition-all duration-200 ${
            theme === 'light' 
            ? 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50 hover:text-slate-900' 
            : 'bg-slate-900 border-slate-700 text-slate-400 hover:bg-slate-800 hover:text-slate-100'
        }`;

        return html`
            <${Panel} position="bottom-right" className="flex flex-col gap-2 pb-4 pr-4">
                <button className=${btnClass} onClick=${() => zoomIn()} title="Zoom In">
                    <${Icons.ZoomIn} />
          </button>
                <button className=${btnClass} onClick=${() => zoomOut()} title="Zoom Out">
                    <${Icons.ZoomOut} />
                </button>
                <button className=${btnClass} onClick=${() => fitView({ padding: 0.2, duration: 200 })} title="Fit View">
                    <${Icons.Center} />
                </button>
                <button className=${`${btnClass} ${showMiniMap ? (theme === 'light' ? 'bg-slate-100 text-indigo-600' : 'bg-slate-800 text-indigo-400') : ''}`} onClick=${onToggleMiniMap} title="Toggle Minimap">
                    <${Icons.Map} />
                </button>
                <div className="h-px bg-slate-200 dark:bg-slate-700 my-1"></div>
                <button className=${btnClass} onClick=${onToggleTheme} title="Toggle Theme">
                    ${theme === 'light' ? html`<${Icons.Moon} />` : html`<${Icons.Sun} />`}
                </button>
            <//>
        `;
      };

      // --- View Controls (top-left toggles for separate_outputs and show_types) ---
      const ViewControls = ({ separateOutputs, showTypes, onToggleSeparate, onToggleTypes, theme }) => {
        const isLight = theme === 'light';
        const containerClass = isLight 
            ? "bg-white/95 border-slate-200 shadow-lg" 
            : "bg-slate-900/95 border-slate-700 shadow-xl";
        const labelClass = isLight ? "text-slate-600" : "text-slate-400";
        const activeClass = isLight ? "bg-indigo-500" : "bg-indigo-600";
        const inactiveClass = isLight ? "bg-slate-300" : "bg-slate-600";
        
        return html`
            <${Panel} position="top-left" className="mt-4 ml-4">
                <div className=${`flex flex-col gap-2.5 px-3 py-2.5 rounded-lg border backdrop-blur-sm text-[11px] ${containerClass}`}>
                    <div className="flex items-center justify-between gap-3">
                        <span className=${`font-medium ${labelClass}`}>Separate outputs</span>
                        <button
                            onClick=${onToggleSeparate}
                            className=${`relative w-8 h-4 rounded-full transition-all duration-300 ${separateOutputs ? activeClass : inactiveClass}`}
                        >
                            <div className=${`absolute top-0.5 w-3 h-3 rounded-full bg-white shadow-sm transition-all duration-300 ${separateOutputs ? "left-4" : "left-0.5"}`}></div>
                        </button>
                    </div>
                    <div className="flex items-center justify-between gap-3">
                        <span className=${`font-medium ${labelClass}`}>Show types</span>
                        <button
                            onClick=${onToggleTypes}
                            className=${`relative w-8 h-4 rounded-full transition-all duration-300 ${showTypes ? activeClass : inactiveClass}`}
                        >
                            <div className=${`absolute top-0.5 w-3 h-3 rounded-full bg-white shadow-sm transition-all duration-300 ${showTypes ? "left-4" : "left-0.5"}`}></div>
                        </button>
                    </div>
                </div>
            <//>
        `;
      };

      // --- Outputs Section (combined outputs display in function nodes) ---
      const OutputsSection = ({ outputs, showTypes, isLight }) => {
        if (!outputs || outputs.length === 0) return null;
        const bgClass = isLight ? "bg-slate-50/80" : "bg-slate-900/50";
        const textClass = isLight ? "text-slate-600" : "text-slate-400";
        const arrowClass = isLight ? "text-emerald-500" : "text-emerald-400";
        const typeClass = isLight ? "text-slate-400" : "text-slate-500";
        const borderClass = isLight ? "border-slate-100" : "border-slate-800/50";
        
        return html`
            <div className=${`px-3 py-2.5 border-t transition-all duration-300 ${bgClass} ${borderClass}`}>
                <div className="flex flex-col items-start gap-1">
                    ${outputs.map(out => html`
                        <div key=${out.name} className=${`flex items-center gap-1.5 text-xs max-w-full overflow-hidden ${textClass}`}>
                            <span className=${`shrink-0 ${arrowClass}`}>→</span>
                            <span className="font-mono font-medium shrink-0">${out.name}</span>
                            ${showTypes && out.type ? html`<span className=${`font-mono truncate min-w-0 ${typeClass}`} title=${out.type}>: ${out.type}</span>` : null}
                        </div>
                    `)}
                </div>
            </div>
        `;
      };

      // --- Edge Component ---
      const CustomEdge = ({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, style = {}, markerEnd, label }) => {
        const [edgePath, labelX, labelY] = getBezierPath({
          sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition
        });

        return html`
          <${React.Fragment}>
            <${BaseEdge} path=${edgePath} markerEnd=${markerEnd} style=${style} />
            ${label ? html`
              <${EdgeLabelRenderer}>
                <div
                  style=${{
                    position: 'absolute',
                    transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
                    pointerEvents: 'all',
                  }}
                  className="px-2 py-1 rounded bg-slate-900/90 border border-slate-700 text-[10px] text-slate-300 font-mono shadow-md backdrop-blur"
                >
                  ${label}
                </div>
              <//>
            ` : null}
          <//>
        `;
      };

      // --- Node Component ---
      const CustomNode = ({ data, id }) => {
        const isExpanded = data.isExpanded;
        // Get theme from node data (updated via setNodes when theme changes)
        const theme = data.theme || 'dark';
        const updateNodeInternals = useUpdateNodeInternals();
        
        // Style Configuration
        let colors = { bg: "slate", border: "slate", text: "slate", icon: "slate" };
        let Icon = Icons.Function;
        let labelType = "NODE";
        
        if (data.nodeType === 'PIPELINE') {
          colors = { bg: "amber", border: "amber", text: "amber", icon: "amber" };
          Icon = Icons.Pipeline;
          labelType = "PIPELINE";
        } else if (data.nodeType === 'DUAL') {
          colors = { bg: "fuchsia", border: "fuchsia", text: "fuchsia", icon: "fuchsia" };
          Icon = Icons.Dual;
          labelType = "DUAL NODE";
        } else if (data.nodeType === 'INPUT') {
          colors = { bg: "cyan", border: "cyan", text: "cyan", icon: "cyan" };
          Icon = Icons.Input;
          labelType = "INPUT";
        } else if (data.nodeType === 'DATA') {
          colors = { bg: "slate", border: "slate", text: "slate", icon: "slate" };
          Icon = Icons.Data;
          labelType = "DATA";
        } else if (data.nodeType === 'INPUT_GROUP') {
          colors = { bg: "cyan", border: "cyan", text: "cyan", icon: "cyan" };
          Icon = Icons.Input;
          labelType = "INPUT GROUP";
        } else {
          colors = { bg: "indigo", border: "indigo", text: "indigo", icon: "indigo" };
          Icon = Icons.Function;
          labelType = "FUNCTION";
        }

        useEffect(() => {
          updateNodeInternals(id);
        }, [
          id,
          data.separateOutputs,
          data.showTypes,
          data.outputs ? data.outputs.length : 0,
          data.inputs ? data.inputs.length : 0,
          isExpanded,
          theme,
        ]);
        
        // --- Render Data Node (Compact) ---
        if (data.nodeType === 'DATA') {
            const isLight = theme === 'light';
            const isOutput = data.sourceId != null;
            const showAsOutput = data.separateOutputs && isOutput;
            const showTypes = data.showTypes;
            const typeClass = isLight ? 'text-slate-400' : 'text-slate-500';
            const hasTypeHint = showTypes && data.typeHint;
            return html`
                <div className=${`px-3 py-1.5 w-full relative rounded-full border shadow-sm flex items-center justify-center gap-1.5 transition-all duration-200 hover:-translate-y-0.5 overflow-hidden
                    ${showAsOutput ? 'ring-2 ring-emerald-500/30' : ''}
                    ${isLight 
                        ? 'bg-white border-slate-200 text-slate-700 shadow-slate-200' 
                        : 'bg-slate-900 border-slate-700 text-slate-300 shadow-black/50'}
                `}>
                     <span className=${`shrink-0 ${isLight ? 'text-slate-400' : 'text-slate-500'}`}><${Icon} /></span>
                     <span className="text-xs font-mono font-medium shrink-0">${data.label}</span>
                     ${hasTypeHint ? html`<span className=${`text-[10px] font-mono truncate min-w-0 ${typeClass}`} title=${data.typeHint}>: ${data.typeHint}</span>` : null}
                     <${Handle} type="target" position=${Position.Top} className="!w-2 !h-2 !opacity-0" style=${{ top: '-2px' }} />
                     <${Handle} type="source" position=${Position.Bottom} className="!w-2 !h-2 !opacity-0" style=${{ bottom: '-2px' }} />
                </div>
            `;
        }

        // --- Render Input Node (Compact - styled as DATA) ---
        if (data.nodeType === 'INPUT') {
             const isLight = theme === 'light';
             const isBound = Boolean(data.isBound);
             const showTypes = data.showTypes;
             const typeHint = data.typeHint;
             const hasType = showTypes && typeHint;
             const typeClass = isLight ? 'text-slate-400' : 'text-slate-500';
             // Reuse DATA node styling but preserve dashed border for bound inputs
             return html`
                <div className=${`px-3 py-1.5 w-full relative rounded-full border shadow-sm flex items-center justify-center gap-2 transition-all duration-200 hover:-translate-y-0.5
                    ${isBound ? 'border-dashed' : ''}
                    ${isLight 
                        ? 'bg-white border-slate-200 text-slate-700 shadow-slate-200' 
                        : 'bg-slate-900 border-slate-700 text-slate-300 shadow-black/50'}
                `}>
                    <span className=${isLight ? 'text-slate-400' : 'text-slate-500'}><${Icons.Data} /></span>
                    <span className="text-xs font-mono font-medium truncate">${data.label}</span>
                    ${hasType ? html`<span className=${`text-[10px] font-mono truncate ${typeClass}`}>: ${typeHint}</span>` : null}
                    <${Handle} type="source" position=${Position.Bottom} className="!w-2 !h-2 !opacity-0" style=${{ bottom: '-2px' }} />
                </div>
             `;
        }

        // --- Render Input Group Node ---
        if (data.nodeType === 'INPUT_GROUP') {
             const isLight = theme === 'light';
             const params = data.params || [];
             const paramTypes = data.paramTypes || [];
             const isBound = data.isBound;
             const showTypes = data.showTypes;
             const typeClass = isLight ? 'text-slate-400' : 'text-slate-500';
             
             return html`
                <div className=${`px-3 py-2 w-full relative rounded-xl border shadow-sm flex flex-col gap-1 min-w-[120px] transition-all duration-200 hover:-translate-y-0.5
                    ${isBound ? 'border-dashed' : ''}
                    ${isLight
                        ? 'bg-white border-slate-200 text-slate-700 shadow-slate-200'
                        : 'bg-slate-900 border-slate-700 text-slate-300 shadow-black/50'}
                `}>
                    ${params.map((p, i) => html`
                        <div className="flex items-center gap-2 whitespace-nowrap">
                            <span className=${isLight ? 'text-slate-400' : 'text-slate-500'}><${Icons.Data} className="w-3 h-3" /></span>
                            <div className="text-xs font-mono leading-tight">${p}</div>
                            ${showTypes && paramTypes[i] ? html`<span className=${`text-[10px] font-mono ${typeClass}`}>: ${paramTypes[i]}</span>` : null}
                        </div>
                    `)}
                    <${Handle} type="source" position=${Position.Bottom} className="!w-2 !h-2 !opacity-0" style=${{ bottom: '-2px' }} />
                </div>
             `;
        }

        // --- Render Expanded Pipeline Group ---
        if (data.nodeType === 'PIPELINE' && isExpanded) {
          const isLight = theme === 'light';
          const handleCollapseClick = (e) => {
            e.stopPropagation();
            e.preventDefault();
            if (data.onToggleExpand) data.onToggleExpand();
          };
          
          return html`
            <div className=${`relative w-full h-full rounded-2xl border-2 border-dashed p-6 transition-all duration-300
                ${isLight 
                    ? 'border-amber-300 bg-amber-50/30' 
                    : 'border-amber-500/30 bg-amber-500/5'}
            `}>
              <button 
                   type="button"
                   className=${`absolute -top-3 left-4 px-3 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider flex items-center gap-2 cursor-pointer transition-colors z-10
                        ${isLight
                            ? 'bg-amber-100 text-amber-700 hover:bg-amber-200 border border-amber-200'
                            : 'bg-slate-950 text-amber-400 hover:text-amber-300 border border-amber-500/50'}
                   `}
                   onClick=${handleCollapseClick}>
                <${Icon} />
                ${data.label}
                <span className="text-[9px] opacity-60 normal-case font-normal ml-1">Click to collapse</span>
              </button>
              <${Handle} type="target" position=${Position.Top} className="!w-2 !h-2 !opacity-0" />
              <${Handle} type="source" position=${Position.Bottom} className="!w-2 !h-2 !opacity-0" />
            </div>
          `;
        }

        // --- Render Standard Node ---
        const isLight = theme === 'light';
        const boundInputs = data.inputs ? data.inputs.filter(i => i.is_bound).length : 0;
        const outputs = data.outputs || [];
        const showCombined = !data.separateOutputs && outputs.length > 0;
        const showTypes = data.showTypes;

        return html`
          <div className=${`group relative w-full rounded-lg border shadow-lg backdrop-blur-sm transition-all duration-300 cursor-pointer hover:-translate-y-1 node-function-${theme} overflow-hidden
               ${isLight 
                 ? `bg-white/90 border-slate-200 shadow-slate-200 hover:border-${colors.border}-400 hover:shadow-${colors.border}-200`
                 : `bg-slate-950/90 border-slate-800 shadow-black/50 hover:border-${colors.border}-500/50 hover:shadow-${colors.border}-500/10`}
               `}
               onClick=${data.nodeType === 'PIPELINE' ? (e) => { e.stopPropagation(); if(data.onToggleExpand) data.onToggleExpand(); } : undefined}>
            
            <!-- Header -->
            <div className=${`px-4 py-2.5 flex items-center gap-3
                 ${showCombined ? (isLight ? 'border-b border-slate-100' : 'border-b border-slate-800/50') : ''}`}>
              <div className=${`p-1.5 rounded-md shrink-0
                   ${isLight 
                     ? `bg-${colors.bg}-50 text-${colors.text}-600` 
                     : `bg-${colors.bg}-500/10 text-${colors.text}-400 border border-${colors.border}-500/20`}`}>
                <${Icon} />
                </div>
              <div className="min-w-0 flex-1">
                <div className=${`text-[9px] font-bold tracking-wider uppercase mb-0.5
                     ${isLight ? `text-${colors.text}-600` : `text-${colors.text}-400`}`}>${labelType}</div>
                <div className=${`text-sm font-semibold truncate
                     ${isLight ? 'text-slate-800' : 'text-slate-100'}`} title=${data.label}>${data.label}</div>
            </div>

              <!-- Bound Input Badge -->
              ${boundInputs > 0 ? html`
                  <div className=${`w-2 h-2 rounded-full ring-2 ring-offset-1
                      ${isLight 
                          ? 'bg-indigo-400 ring-indigo-100 ring-offset-white' 
                          : 'bg-indigo-500 ring-indigo-500/30 ring-offset-slate-950'}`}
                       title="${boundInputs} bound inputs">
                </div>
              ` : null}
            </div>
            
            <!-- Combined Outputs Section -->
            ${showCombined ? html`<${OutputsSection} outputs=${outputs} showTypes=${showTypes} isLight=${isLight} />` : null}

            <!-- Handles (invisible) -->
            <${Handle} type="target" position=${Position.Top} className="!w-2 !h-2 !opacity-0" />
            <${Handle} type="source" position=${Position.Bottom} className="!w-2 !h-2 !opacity-0" />
            
            ${data.nodeType === 'PIPELINE' ? html`
               <div className="absolute -bottom-5 left-1/2 -translate-x-1/2 text-[9px] text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                 Click to expand
               </div>
            ` : null}
          </div>
        `;
      };

      const nodeTypes = { custom: CustomNode, pipelineGroup: CustomNode };
      const edgeTypes = { custom: CustomEdge };

      const useLayout = (nodes, edges) => {
        const [layoutedNodes, setLayoutedNodes] = useState([]);
        const [layoutedEdges, setLayoutedEdges] = useState([]);
        const [layoutError, setLayoutError] = useState(null);
        const [graphHeight, setGraphHeight] = useState(600);
        const [graphWidth, setGraphWidth] = useState(600);

        useEffect(() => {
          if (!nodes.length) return;

          const buildElkHierarchy = (nodes, edges) => {
            const visibleNodes = nodes.filter(n => !n.hidden);
            const visibleNodeIds = new Set(visibleNodes.map(n => n.id));
            const visibleEdges = edges.filter(e => visibleNodeIds.has(e.source) && visibleNodeIds.has(e.target));
            
            const visibleNodeMap = new Map(visibleNodes.map(n => [n.id, { ...n, children: [], edges: [] }]));
            const rootChildren = [];
            
            visibleNodes.forEach(n => {
              if (n.parentNode && visibleNodeIds.has(n.parentNode)) {
                const parent = visibleNodeMap.get(n.parentNode);
                if (parent) parent.children.push(visibleNodeMap.get(n.id));
              } else {
                rootChildren.push(visibleNodeMap.get(n.id));
              }
            });
            
            const mapToElk = (n) => {
                let width = 200;
                let height = 90;
                
                if (n.data?.nodeType === 'DATA') {
                    width = 140;
                    height = 36;
                    // Calculate width based on label + optional type hint
                    const labelLen = n.data.label ? n.data.label.length : 0;
                    const typeLen = (n.data.showTypes && n.data.typeHint) ? Math.min(n.data.typeHint.length, 15) : 0;
                    width = Math.max(100, (labelLen + typeLen + 4) * 7 + 50);
                } else if (n.data?.nodeType === 'PIPELINE' && !n.data?.isExpanded) {
                    // Collapsed pipeline - compact size based on label
                    const labelLen = n.data.label ? n.data.label.length : 10;
                    width = Math.max(140, labelLen * 8 + 80);
                    height = 68;
                } else                 if (n.data?.nodeType === 'INPUT') {
                    width = 160;
                    height = 46;
                } else if (n.data?.nodeType === 'INPUT_GROUP') {
                    width = 200;
                    // Dynamic height based on number of inputs
                    const paramCount = n.data.params ? n.data.params.length : 1;
                    height = 46 + (paramCount * 18);
                }
                
                if (n.style && n.style.width) width = n.style.width;
                if (n.style && n.style.height) height = n.style.height;

                // For compound nodes (with children), let ELK calculate dimensions unless explicit
                const isCompound = n.children && n.children.length > 0;
                
                return {
                  id: n.id,
                  width: width,
                  height: height,
                  children: n.children.length ? n.children.map(mapToElk) : undefined,
                  layoutOptions: {
                     'elk.padding': '[top=60,left=24,bottom=40,right=24]',
                     'elk.spacing.nodeNode': '24',
                     'elk.layered.spacing.nodeNodeBetweenLayers': '48',
                     'elk.direction': 'DOWN',
                     // Ensure compound nodes are sized by content
                     'elk.resize.fixed': 'false', 
                  }
                };
            };

            const elkGraph = {
              id: 'root',
              layoutOptions: {
                // Sugiyama-style layered algorithm (like Kedro-viz)
                'elk.algorithm': 'layered',
                'elk.direction': 'DOWN',
                
                // POLYLINE routing for straight edges with gentle curves at corners
                'elk.edgeRouting': 'POLYLINE',
                'elk.layered.edgeRouting.polyline.slopedEdgeZoneWidth': '3.0',
                
                // Increased spacing to ensure edges don't touch nodes
                'elk.layered.spacing.nodeNodeBetweenLayers': '60', // More breathing room vertically
                'elk.spacing.nodeNode': '24', // Space between nodes in same layer
                'elk.spacing.edgeNode': '40', // Increased from 30 - prevents edge-node contact
                'elk.spacing.edgeEdge': '15', // Increased from 15 - space between parallel edges
                
                // Advanced crossing minimization (Sugiyama algorithm core)
                'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
                'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES',
                'elk.layered.thoroughness': '10', // Higher = better quality (default 7)
                
                // Node placement for straighter edges
                'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX', // Better than BRANDES_KOEPF for DAGs
                'elk.layered.nodePlacement.favorStraightEdges': 'true',
                
                // Compaction for tighter, cleaner layout
                'elk.layered.compaction.postCompaction.strategy': 'EDGE_LENGTH',
                'elk.layered.compaction.connectedComponents': 'true',
                
                // Hierarchical handling for nested pipelines
                'elk.hierarchyHandling': 'INCLUDE_CHILDREN',
                'elk.resize.fixed': 'false',
                
                // Port and edge aesthetics
                'elk.portConstraints': 'FIXED_ORDER',
                'elk.layered.unnecessaryBendpoints': 'false', // Remove redundant bends
                'elk.layered.mergeEdges': 'false', // Keep edges separate for clarity
              },
              children: rootChildren.map(mapToElk),
              edges: visibleEdges.map(e => ({
                id: e.id,
                sources: [e.source],
                targets: [e.target],
              })),
            };
            
            return { elkGraph, visibleEdges };
          };

          const { elkGraph, visibleEdges } = buildElkHierarchy(nodes, edges);

          elk.layout(elkGraph).then((graph) => {
              setLayoutError(null);
              const positionedNodes = [];
              
              const traverse = (node, parentX = 0, parentY = 0) => {
                const x = node.x || 0;
                const y = node.y || 0;
                
                const original = nodes.find(n => n.id === node.id);
                if (original) {
                  positionedNodes.push({
                    ...original,
                    position: { x, y },
                    style: { ...original.style, width: node.width, height: node.height },
                  });
                }
                if (node.children) node.children.forEach(child => traverse(child, x, y));
              };
              
              (graph.children || []).forEach(n => traverse(n));
              setLayoutedNodes(positionedNodes);
              setLayoutedEdges(visibleEdges);
              if (graph.height) setGraphHeight(graph.height);
              if (graph.width) setGraphWidth(graph.width);
            })
            .catch((err) => {
                console.error('ELK layout error', err);
                setLayoutError(err?.message || 'Layout error');
                // Fallback
                const fallbackNodes = nodes.map((n, idx) => ({
                    ...n,
                    position: { x: 80 * (idx % 4), y: 120 * Math.floor(idx / 4) },
                }));
                setLayoutedNodes(fallbackNodes);
                setLayoutedEdges(edges);
            });
        }, [nodes, edges]);

        return { layoutedNodes, layoutedEdges, layoutError, graphHeight, graphWidth };
      };

      const initialData = JSON.parse(document.getElementById('graph-data').textContent || '{"nodes":[],"edges":[]}');
      const normalizeThemePref = (pref) => {
        const lower = (pref || '').toLowerCase();
        return ['light', 'dark', 'auto'].includes(lower) ? lower : 'auto';
      };
      const themePreference = normalizeThemePref(initialData.meta?.theme_preference || 'auto');
      const showThemeDebug = Boolean(initialData.meta?.theme_debug);
      const panOnScroll = Boolean(initialData.meta?.pan_on_scroll);
      const initialSeparateOutputs = Boolean(initialData.meta?.separate_outputs ?? false);
      const initialShowTypes = Boolean(initialData.meta?.show_types ?? true);

      const parseColorString = (value) => {
        if (themeUtils?.parseColorString) return themeUtils.parseColorString(value);
        if (!value) return null;
        const scratch = document.createElement('div');
        scratch.style.color = value;
        scratch.style.backgroundColor = value;
        scratch.style.display = 'none';
        document.body.appendChild(scratch);
        const resolved = getComputedStyle(scratch).color || '';
        scratch.remove();
        const nums = resolved.match(/[\d\.]+/g);
        if (nums && nums.length >= 3) {
            const [r, g, b] = nums.slice(0, 3).map(Number);
            const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
            return { r, g, b, luminance, resolved, raw: value };
        }
        return null;
      };

      const App = () => {
        const [showMiniMap, setShowMiniMap] = useState(false);
        const [separateOutputs, setSeparateOutputs] = useState(initialSeparateOutputs);
        const [showTypes, setShowTypes] = useState(initialShowTypes);
        const [themeDebug, setThemeDebug] = useState({ source: 'init', luminance: null, background: 'transparent', appliedTheme: themePreference });
        const [detectedTheme, setDetectedTheme] = useState(() => {
            if (themeUtils?.detectHostTheme) return themeUtils.detectHostTheme();
            const fallback = parseColorString('transparent');
            return { theme: themePreference === 'auto' ? 'dark' : themePreference, background: fallback?.resolved || 'transparent', luminance: fallback?.luminance ?? null, source: 'init' };
        });
        const [manualTheme, setManualTheme] = useState(null);
        const [bgColor, setBgColor] = useState(detectedTheme.background || 'transparent');
        
        // Track expansion state separately to preserve it across theme/toggle changes
        const [expansionState, setExpansionState] = useState(() => {
            const map = new Map();
            initialData.nodes.forEach(n => {
                if (n.data.nodeType === 'PIPELINE') {
                    map.set(n.id, n.data.isExpanded || false);
                }
            });
            return map;
        });
        
        // Use React Flow's state management
        const [rfNodes, setNodes, onNodesChange] = useNodesState([]);
        const [rfEdges, setEdges, onEdgesChange] = useEdgesState([]);
        
        const nodesRef = useRef(initialData.nodes);

        const resolvedDetected = detectedTheme || { theme: themePreference === 'auto' ? 'dark' : themePreference, background: 'transparent', luminance: null, source: 'init' };
        const activeTheme = useMemo(() => {
            const base = themePreference === 'auto' ? (resolvedDetected.theme || 'dark') : themePreference;
            return manualTheme || base;
        }, [manualTheme, resolvedDetected.theme, themePreference]);
        const activeBackground = useMemo(() => {
            if (manualTheme) return manualTheme === 'light' ? '#f8fafc' : '#020617';
            return resolvedDetected.background || 'transparent';
        }, [manualTheme, resolvedDetected.background]);
        const theme = activeTheme;

        // Expansion logic
        const onToggleExpand = useCallback((nodeId) => {
          setExpansionState(prev => {
            const newMap = new Map(prev);
            const isCurrentlyExpanded = newMap.get(nodeId) || false;
            const willExpand = !isCurrentlyExpanded;
            newMap.set(nodeId, willExpand);

            if (!willExpand) {
                const currentNodes = nodesRef.current || [];
                const childrenMap = new Map();
                currentNodes.forEach(n => {
                    if (n.parentNode) {
                        if (!childrenMap.has(n.parentNode)) childrenMap.set(n.parentNode, []);
                        childrenMap.get(n.parentNode).push(n.id);
                    }
                });

                const getDescendants = (id) => {
                    const children = childrenMap.get(id) || [];
                    let res = [...children];
                    children.forEach(childId => {
                        res = res.concat(getDescendants(childId));
                    });
                    return res;
                };

                getDescendants(nodeId).forEach(descId => {
                    if (newMap.has(descId)) newMap.set(descId, false);
                });
            }

            return newMap;
          });
        }, []);

        const applyStateFn = stateUtils.applyState || fallbackApplyState;
        const applyVisibilityFn = stateUtils.applyVisibility || fallbackApplyVisibility;

        const stateResult = useMemo(() => {
            return applyStateFn(initialData.nodes, initialData.edges, {
                expansionState,
                separateOutputs,
                showTypes,
                theme: activeTheme,
            });
        }, [applyStateFn, initialData, expansionState, separateOutputs, showTypes, activeTheme]);

        // Add callbacks and visibility in a single path so hidden flags persist through toggles
        const nodesWithCallbacks = useMemo(() => stateResult.nodes.map(n => ({
            ...n,
            data: { ...n.data, onToggleExpand: n.data.nodeType === 'PIPELINE' ? () => onToggleExpand(n.id) : n.data.onToggleExpand },
        })), [stateResult.nodes, onToggleExpand]);

        const nodesWithVisibility = useMemo(() => {
            const nextNodes = applyVisibilityFn(nodesWithCallbacks, expansionState);
            nodesRef.current = nextNodes;
            return nextNodes;
        }, [nodesWithCallbacks, expansionState, applyVisibilityFn]);

        useEffect(() => {
          setNodes(nodesWithVisibility);
          setEdges(stateResult.edges);
        }, [nodesWithVisibility, stateResult.edges, setNodes, setEdges]);

        const detectHostTheme = useCallback(() => {
            if (themeUtils?.detectHostTheme) return themeUtils.detectHostTheme();

            const attempts = [];
            const pushCandidate = (value, source) => {
                if (value) attempts.push({ value: value.trim(), source });
            };

            try {
                const parentDoc = window.parent?.document;
                if (parentDoc) {
                    const rootStyle = getComputedStyle(parentDoc.documentElement);
                    const bodyStyle = getComputedStyle(parentDoc.body);
                    pushCandidate(rootStyle.getPropertyValue('--vscode-editor-background'), '--vscode-editor-background');
                    pushCandidate(bodyStyle.backgroundColor, 'parent body background');
                    pushCandidate(rootStyle.backgroundColor, 'parent root background');
                }
            } catch (e) {}

            pushCandidate(getComputedStyle(document.body).backgroundColor, 'iframe body');

            let chosen = attempts.find(c => parseColorString(c.value));
            if (!chosen) chosen = { value: 'transparent', source: 'default' };
            const parsed = parseColorString(chosen.value);
            const luminance = parsed ? parsed.luminance : null;

            let autoTheme = luminance !== null ? (luminance > 150 ? 'light' : 'dark') : 'dark';
            let source = luminance !== null ? `${chosen.source} luminance` : chosen.source;

            try {
                const parentDoc = window.parent?.document;
                if (parentDoc) {
                    const themeKind = parentDoc.body.getAttribute('data-vscode-theme-kind');
                    if (themeKind) {
                        autoTheme = themeKind.includes('light') ? 'light' : 'dark';
                        source = 'vscode-theme-kind';
                    } else if (parentDoc.body.className && parentDoc.body.className.includes('vscode-light')) {
                        autoTheme = 'light';
                        source = 'vscode body class';
                    } else if (parentDoc.body.className && parentDoc.body.className.includes('vscode-dark')) {
                        autoTheme = 'dark';
                        source = 'vscode body class';
                    }
                }
            } catch (e) {}

            if (source === 'default' && window.matchMedia) {
                if (window.matchMedia('(prefers-color-scheme: light)').matches) {
                    autoTheme = 'light';
                    source = 'prefers-color-scheme';
                } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
                    autoTheme = 'dark';
                    source = 'prefers-color-scheme';
                }
            }

            return {
                theme: autoTheme,
                background: parsed ? (parsed.resolved || parsed.raw || chosen.value) : chosen.value,
                luminance,
                source,
            };
        }, []);

        // Theme detection listener (updates detected theme only)
        useEffect(() => {
          const applyThemeDetection = () => {
             const detected = detectHostTheme();
             setDetectedTheme(detected);
          };

          applyThemeDetection();

          const observers = [];
          try { 
            const parentDoc = window.parent?.document;
            if (parentDoc) {
                const config = { attributes: true, attributeFilter: ['class', 'data-vscode-theme-kind', 'style'] };
                const observer = new MutationObserver(applyThemeDetection);
                observer.observe(parentDoc.body, config);
                observer.observe(parentDoc.documentElement, config);
                observers.push(observer);
            }
          } catch(e) {}

          const mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
          const mqHandler = () => applyThemeDetection();
          if (mq && mq.addEventListener) mq.addEventListener('change', mqHandler);
          else if (mq && mq.addListener) mq.addListener(mqHandler);

          return () => {
            observers.forEach(o => o.disconnect());
            if (mq && mq.removeEventListener) mq.removeEventListener('change', mqHandler);
            else if (mq && mq.removeListener) mq.removeListener(mqHandler);
          };
        }, [detectHostTheme]);

        // Apply effective theme + background
        useEffect(() => {
            setBgColor(activeBackground);
            document.body.classList.toggle('light-mode', activeTheme === 'light');
            if (showThemeDebug) {
                setThemeDebug({
                    source: manualTheme ? 'manual toggle' : resolvedDetected.source,
                    luminance: resolvedDetected.luminance,
                    background: activeBackground,
                    appliedTheme: activeTheme,
                });
            }
            window.__hypernodesVizThemeState = {
                ...resolvedDetected,
                appliedTheme: activeTheme,
                background: activeBackground,
                preference: themePreference,
                manualTheme,
            };
        }, [activeTheme, activeBackground, resolvedDetected, showThemeDebug, themePreference, manualTheme]);

        const toggleTheme = useCallback(() => {
            const current = manualTheme || (themePreference === 'auto' ? resolvedDetected.theme : themePreference);
            const next = current === 'light' ? 'dark' : 'light';
            setManualTheme(next);
        }, [manualTheme, themePreference, resolvedDetected.theme]);

        // Compress edges first (remaps to visible ancestors when pipelines collapse)
        // IMPORTANT: Use nodesWithVisibility and stateResult.edges (synchronous) instead of 
        // rfNodes/rfEdges (async via setNodes/setEdges) to avoid stale data on first render
        const compressedEdges = useMemo(() => {
            const compressor = stateUtils.compressEdges || ((nodes, edges) => edges);
            return compressor(nodesWithVisibility, stateResult.edges);
        }, [nodesWithVisibility, stateResult.edges]);

        // Group inputs that share the same targets after compression
        const { nodes: groupedNodes, edges: groupedEdges } = useMemo(() => {
            const grouper = stateUtils.groupInputs || ((nodes, edges) => ({ nodes, edges }));
            return grouper(nodesWithVisibility, compressedEdges);
        }, [nodesWithVisibility, compressedEdges]);

        const { layoutedNodes, layoutedEdges, layoutError, graphHeight, graphWidth } = useLayout(groupedNodes, groupedEdges);
        const { fitView } = useReactFlow();

        // --- Iframe Resize Logic (Task 2) ---
        useEffect(() => {
            if (graphHeight && graphWidth) {
                const desiredHeight = Math.max(400, graphHeight + 50);
                const desiredWidth = Math.max(400, graphWidth + 50);
                try {
                    // Try to resize the hosting iframe to avoid internal scrollbars and excess padding
                    if (window.frameElement) {
                        window.frameElement.style.height = desiredHeight + 'px';
                        window.frameElement.style.width = desiredWidth + 'px';
                    }
                } catch (e) {
                    // Ignore cross-origin errors or missing frameElement
                }
            }
        }, [graphHeight, graphWidth]);

        // --- Resize Handling (Task 2) ---
        useEffect(() => {
            const handleResize = () => {
                fitView({ padding: 0.1, duration: 200, minZoom: 0.5, maxZoom: 1 });
            };
            window.addEventListener('resize', handleResize);
            return () => window.removeEventListener('resize', handleResize);
        }, [fitView]);
        
        // Re-fit when layout changes - Forced recentering after slight delay to allow iframe resize
        useEffect(() => {
            if (layoutedNodes.length > 0) {
                // Immediate fit
                window.requestAnimationFrame(() => fitView({ padding: 0.1, duration: 0, minZoom: 0.5, maxZoom: 1 }));
                // Delayed fit to catch iframe resize
                setTimeout(() => {
                    fitView({ padding: 0.1, duration: 200, minZoom: 0.5, maxZoom: 1 });
                }, 100);
            }
        }, [layoutedNodes, fitView]);

        const edgeOptions = {
            type: 'custom',
            sourcePosition: Position.Bottom,
            targetPosition: Position.Top,
            style: { stroke: theme === 'light' ? '#94a3b8' : '#64748b', strokeWidth: 2 },
            markerEnd: { type: MarkerType.ArrowClosed, color: theme === 'light' ? '#94a3b8' : '#64748b' },
        };

        const styledEdges = useMemo(() => layoutedEdges.map(e => {
            const isDataLink = e.data && e.data.isDataLink;
            return { 
                ...e, 
                ...edgeOptions,
                style: { 
                    ...edgeOptions.style, 
                    strokeWidth: isDataLink ? 1.5 : 2,
                },
                markerEnd: edgeOptions.markerEnd
            };
        }), [layoutedEdges, theme]);

        return html`
          <div 
            className=${`w-full relative overflow-hidden transition-colors duration-300`}
            style=${{ backgroundColor: bgColor, height: '100vh', width: '100vw' }}
          >
            <!-- Background Grid -->
            <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 pointer-events-none mix-blend-overlay"></div>
            
            <${ReactFlow}
              nodes=${layoutedNodes}
              edges=${styledEdges}
              nodeTypes=${nodeTypes}
              edgeTypes=${edgeTypes}
              onNodesChange=${onNodesChange}
              onEdgesChange=${onEdgesChange}
              onNodeClick=${(e, node) => {
                // Handle click on collapsed pipeline nodes to expand
                if (node.data.nodeType === 'PIPELINE' && !node.data.isExpanded && node.data.onToggleExpand) {
                  e.stopPropagation();
                  node.data.onToggleExpand();
                }
              }}
              fitView
              fitViewOptions=${{ padding: 0.02, minZoom: 0.5, maxZoom: 1 }}
              minZoom=${0.1}
              maxZoom=${2}
              className=${'bg-transparent'}
              panOnScroll=${panOnScroll}
              zoomOnScroll=${false}
              panOnDrag=${true}
              zoomOnPinch=${true} 
              preventScrolling=${false}
              style=${{ width: '100%', height: '100%' }}
            >
              <${Background} color=${theme === 'light' ? '#94a3b8' : '#334155'} gap=${24} size=${1} variant="dots" />
              <${CustomControls} theme=${theme} onToggleTheme=${toggleTheme} showMiniMap=${showMiniMap} onToggleMiniMap=${() => setShowMiniMap(m => !m)} />
              <${ViewControls} 
                separateOutputs=${separateOutputs} 
                showTypes=${showTypes}
                onToggleSeparate=${() => setSeparateOutputs(s => !s)}
                onToggleTypes=${() => setShowTypes(t => !t)}
                theme=${theme} 
              />
              ${showThemeDebug ? html`
              <${Panel} position="top-left" className=${`backdrop-blur-sm rounded-lg shadow-lg border text-xs px-3 py-2 mt-3 ml-3 max-w-xs
                    ${theme === 'light' ? 'bg-white/95 border-slate-200 text-slate-700' : 'bg-slate-900/90 border-slate-700 text-slate-200'}`}>
                  <div className="text-[10px] font-semibold tracking-wide uppercase opacity-70">Theme debug</div>
                  <div className="mt-0.5">Applied: <span className="font-semibold">${theme}</span> (pref: ${themePreference})</div>
                  <div>Source: ${themeDebug.source || 'n/a'}</div>
                  <div className="truncate" title=${bgColor}>BG: ${bgColor || 'transparent'}</div>
                  ${themeDebug.luminance !== null ? html`<div>Luma: ${Math.round(themeDebug.luminance)}</div>` : null}
              <//>
              ` : null}
              ${showMiniMap ? html`
              <${MiniMap} 
                className=${theme === 'light' ? '!bg-white !border-slate-200 !shadow-xl rounded-lg overflow-hidden' : '!bg-slate-900 !border-slate-700 !shadow-xl rounded-lg overflow-hidden'}
                maskColor=${theme === 'light' ? 'rgba(241, 245, 249, 0.6)' : 'rgba(15, 23, 42, 0.6)'}
                nodeColor=${(n) => theme === 'light' ? '#cbd5e1' : '#475569'}
              />
              ` : null}
            <//>
            ${(layoutError || (!layoutedNodes.length && rfNodes.length) || (!rfNodes.length)) ? html`
                <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                  <div className="px-4 py-2 rounded-lg border text-xs font-mono bg-slate-900/80 text-amber-200 border-amber-500/40 shadow-lg pointer-events-auto">
                    ${layoutError ? `Layout error: ${layoutError}` : (!rfNodes.length ? 'No graph data' : 'Layout produced no nodes. Showing fallback.')}
                    <button className="ml-4 underline text-amber-400 hover:text-amber-100" onClick=${() => window.location.reload()}>Reload</button>
                  </div>
                </div>
            ` : null}
          </div>
        `;
      };

      const root = ReactDOM.createRoot(document.getElementById('root'));
    root.render(html`
      <${ReactFlowProvider}>
        <${App} />
      <//>
    `);
      if (fallback) {
        fallback.remove();
      }
    } catch (err) {
      console.error(err);
      fail("Viz error: " + (err && err.message ? err.message : err));
    }
  </script>
  <script id="graph-data" type="application/json">__GRAPH_JSON__</script>
</body>
</html>"""

    html_template = html_head + html_body
    return html_template.replace("__GRAPH_JSON__", graph_json)
