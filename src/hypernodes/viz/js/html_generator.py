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
        body {{ margin: 0; overflow: auto; background: transparent; color: #e5e7eb; font-family: 'Inter', system-ui, -apple-system, sans-serif; }}
        .react-flow__attribution {{ display: none; }}
        #root {{ min-height: 100vh; min-width: 100vw; background: transparent; display: flex; align-items: flex-start; justify-content: center; }}
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
            </div>
        `;
      }
    };

    try {
      const React = window.React;
      const ReactDOM = window.ReactDOM;
      const RF = window.ReactFlow;
      const htm = window.htm;
      const ELK = window.ELK;

      if (!React || !ReactDOM || !RF || !htm || !ELK) {
        throw new Error("Missing globals: " + JSON.stringify({
          React: !!React, ReactDOM: !!ReactDOM, ReactFlow: !!RF, htm: !!htm, ELK: !!ELK
        }));
      }

      const { ReactFlow, Background, Controls, MiniMap, Handle, Position, ReactFlowProvider, useEdgesState, useNodesState, MarkerType, BaseEdge, getBezierPath, EdgeLabelRenderer, useReactFlow, Panel } = RF;
      const { useState, useEffect, useMemo, useCallback, useRef } = React;

      const html = htm.bind(React.createElement);
      const elk = new ELK();

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
        const theme = data.theme;
        
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
        
        // --- Render Data Node (Compact) ---
        if (data.nodeType === 'DATA') {
            const isLight = theme === 'light';
            return html`
                <div className=${`px-3 py-1.5 rounded-full border shadow-sm flex items-center gap-2 transition-all duration-200 hover:-translate-y-0.5
                    ${isLight 
                        ? 'bg-white border-slate-200 text-slate-700 shadow-slate-200' 
                        : 'bg-slate-900 border-slate-700 text-slate-300 shadow-black/50'}
                `}>
                     <span className=${isLight ? 'text-slate-400' : 'text-slate-500'}><${Icon} /></span>
                     <span className="text-xs font-mono font-medium">${data.label}</span>
                     <${Handle} type="target" position=${Position.Top} className="!w-2 !h-2 !opacity-0" />
                     <${Handle} type="source" position=${Position.Bottom} className="!w-2 !h-2 !opacity-0" />
                </div>
            `;
        }

        // --- Render Input Node (Compact) ---
        if (data.nodeType === 'INPUT') {
             const isLight = theme === 'light';
             const isBound = Boolean(data.isBound);
             return html`
                <div className=${`px-3 py-2 rounded-lg border-2 flex items-center gap-2
                    ${isBound ? 'border-dashed' : 'border-solid'}
                    ${isLight
                        ? 'bg-cyan-50/50 border-cyan-200 text-cyan-800'
                        : 'bg-cyan-950/20 border-cyan-800/50 text-cyan-200'}
                `}>
                    <!-- Dashed outline is reserved for bound inputs -->
                    <${Icon} />
                    <span className="text-xs font-bold tracking-wide">${data.label}</span>
                    <${Handle} type="source" position=${Position.Bottom} className="!w-2 !h-2 !opacity-0" />
                </div>
             `;
        }

        // --- Render Input Group Node ---
        if (data.nodeType === 'INPUT_GROUP') {
             const isLight = theme === 'light';
             const params = data.params || [];
             const isBound = data.isBound;
             
             return html`
                <div className=${`px-3 py-2 rounded-lg border-2 flex flex-col gap-1 min-w-[120px]
                    ${isBound ? 'border-dashed' : 'border-solid'}
                    ${isLight
                        ? 'bg-cyan-50/50 border-cyan-200 text-cyan-800'
                        : 'bg-cyan-950/20 border-cyan-800/50 text-cyan-200'}
                `}>
                    <div className="flex items-center gap-2 mb-1 pb-1 border-b border-cyan-500/20">
                        <${Icon} className="w-3 h-3" />
                        <span className="text-[10px] font-bold uppercase tracking-wider opacity-70">Inputs</span>
                    </div>
                    ${params.map(p => html`
                        <div className="text-xs font-mono leading-tight">${p}</div>
                    `)}
                    <${Handle} type="source" position=${Position.Bottom} className="!w-2 !h-2 !opacity-0" />
                </div>
             `;
        }

        // --- Render Expanded Pipeline Group ---
        if (data.nodeType === 'PIPELINE' && isExpanded) {
          const isLight = theme === 'light';
          return html`
            <div className=${`relative w-full h-full rounded-2xl border-2 border-dashed p-4 transition-all duration-300
                ${isLight 
                    ? 'border-amber-300 bg-amber-50/30' 
                    : 'border-amber-500/30 bg-amber-500/5'}
            `} onWheel=${(e) => e.stopPropagation()}>
              <div 
                   className=${`absolute -top-3 left-4 px-3 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider flex items-center gap-2 cursor-pointer transition-colors
                        ${isLight
                            ? 'bg-amber-100 text-amber-700 hover:bg-amber-200 border border-amber-200'
                            : 'bg-slate-950 text-amber-400 hover:text-amber-300 border border-amber-500/50'}
                   `}
                   onClick=${data.onToggleExpand}>
                <${Icon} />
                ${data.label}
                <span className="text-[9px] opacity-60 normal-case ml-1">Click to collapse</span>
              </div>
              <${Handle} type="target" position=${Position.Top} className="!w-0 !h-0 !opacity-0" />
              <${Handle} type="source" position=${Position.Bottom} className="!w-0 !h-0 !opacity-0" />
            </div>
          `;
        }

        // --- Render Standard Node ---
        const isLight = theme === 'light';
        const boundInputs = data.inputs ? data.inputs.filter(i => i.is_bound).length : 0;

        return html`
          <div className=${`group relative w-[240px] rounded-lg border shadow-lg backdrop-blur-sm transition-all duration-300 cursor-pointer hover:-translate-y-1 node-function-${theme}
               ${isLight 
                 ? `bg-white/90 border-slate-200 shadow-slate-200 hover:border-${colors.border}-400 hover:shadow-${colors.border}-200`
                 : `bg-slate-950/90 border-slate-800 shadow-black/50 hover:border-${colors.border}-500/50 hover:shadow-${colors.border}-500/10`}
               `}
               onClick=${data.nodeType === 'PIPELINE' ? data.onToggleExpand : undefined}>
            
            <!-- Header -->
            <div className=${`px-3 py-2.5 border-b flex items-center gap-3
                 ${isLight ? 'border-slate-100' : 'border-slate-800/50'}`}>
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

            <!-- Handles -->
            <${Handle} type="target" position=${Position.Top} className=${`!w-2 !h-2 !border-2 transition-colors
                 ${isLight 
                    ? '!bg-white !border-slate-300 group-hover:!border-slate-400' 
                    : '!bg-slate-950 !border-slate-600 group-hover:!border-slate-500'}`} />
            <${Handle} type="source" position=${Position.Bottom} className=${`!w-2 !h-2 !border-2 transition-colors
                 ${isLight 
                    ? '!bg-white !border-slate-300 group-hover:!border-slate-400' 
                    : '!bg-slate-950 !border-slate-600 group-hover:!border-slate-500'}`} />
            
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
                let width = 320;
                let height = 90;
                
                if (n.data?.nodeType === 'DATA') {
                    width = 160;
                    height = 40;
                    if (n.data.label) width = Math.max(120, n.data.label.length * 8 + 40);
                } else if (n.data?.nodeType === 'INPUT') {
                    width = 180;
                    height = 50;
                } else if (n.data?.nodeType === 'INPUT_GROUP') {
                    width = 200;
                    // Dynamic height based on number of inputs
                    const paramCount = n.data.params ? n.data.params.length : 1;
                    height = 40 + (paramCount * 16);
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
                     'elk.padding': '[top=20,left=10,bottom=10,right=10]',
                     'elk.spacing.nodeNode': '40',
                     'elk.layered.spacing.nodeNodeBetweenLayers': '60',
                     'elk.direction': 'DOWN',
                     // Ensure compound nodes are sized by content
                     'elk.resize.fixed': 'false', 
                  }
                };
            };

            const elkGraph = {
              id: 'root',
              layoutOptions: {
                'elk.algorithm': 'layered',
                'elk.direction': 'DOWN',
                'elk.edgeRouting': 'ORTHOGONAL',
                'elk.layered.spacing.nodeNodeBetweenLayers': '50',
                'elk.spacing.nodeNode': '40',
                'elk.layered.nodePlacement.strategy': 'BRANDES_KOEPF',
                'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
                'elk.hierarchyHandling': 'INCLUDE_CHILDREN',
                'elk.resize.fixed': 'false',
                // Separate ports for cleaner routing (Task 7)
                'elk.portConstraints': 'FIXED_ORDER', 
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

        return { layoutedNodes, layoutedEdges, layoutError, graphHeight };
      };

      const initialData = JSON.parse(document.getElementById('graph-data').textContent || '{"nodes":[],"edges":[]}');
      const normalizeThemePref = (pref) => {
        const lower = (pref || '').toLowerCase();
        return ['light', 'dark', 'auto'].includes(lower) ? lower : 'auto';
      };
      const themePreference = normalizeThemePref(initialData.meta?.theme_preference || 'auto');
      const showThemeDebug = Boolean(initialData.meta?.theme_debug);
      const panOnScroll = Boolean(initialData.meta?.pan_on_scroll);

      const parseColorString = (value) => {
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
        const [nodes, setNodes, onNodesChange] = useNodesState(initialData.nodes);
        const [edges, setEdges, onEdgesChange] = useEdgesState(initialData.edges);
        const [theme, setTheme] = useState(themePreference === 'auto' ? 'dark' : themePreference);
        const [showMiniMap, setShowMiniMap] = useState(false);
        const [bgColor, setBgColor] = useState('transparent');
        const [themeDebug, setThemeDebug] = useState({
            source: 'init',
            luminance: null,
            background: 'transparent',
            appliedTheme: themePreference === 'auto' ? 'dark' : themePreference,
        });

        const detectHostTheme = useCallback(() => {
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

        // --- Theme Detection & Background (Task 1) ---
        useEffect(() => {
          const applyTheme = () => {
             const detected = detectHostTheme();
             const appliedTheme = themePreference === 'auto' ? detected.theme : themePreference;
             
             setTheme(appliedTheme);
             setBgColor(detected.background || 'transparent');
             if (showThemeDebug) {
                setThemeDebug({
                    source: detected.source,
                    luminance: detected.luminance,
                    background: detected.background,
                    appliedTheme,
                });
             }
             document.body.classList.toggle('light-mode', appliedTheme === 'light');
             window.__hypernodesVizThemeState = { ...detected, appliedTheme, preference: themePreference };
          };

          applyTheme();

          const observers = [];
          try { 
            const parentDoc = window.parent?.document;
            if (parentDoc) {
                const config = { attributes: true, attributeFilter: ['class', 'data-vscode-theme-kind', 'style'] };
                const observer = new MutationObserver(applyTheme);
                observer.observe(parentDoc.body, config);
                observer.observe(parentDoc.documentElement, config);
                observers.push(observer);
            }
          } catch(e) {}

          const mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
          const mqHandler = () => applyTheme();
          if (mq && mq.addEventListener) mq.addEventListener('change', mqHandler);
          else if (mq && mq.addListener) mq.addListener(mqHandler);

          return () => {
            observers.forEach(o => o.disconnect());
            if (mq && mq.removeEventListener) mq.removeEventListener('change', mqHandler);
            else if (mq && mq.removeListener) mq.removeListener(mqHandler);
          };
        }, [detectHostTheme, themePreference]);

        const toggleTheme = useCallback(() => {
            setTheme(t => {
                const next = t === 'light' ? 'dark' : 'light';
                document.body.classList.toggle('light-mode', next === 'light');
                // If we toggle manually, we override the detected background to a preset
                const manualBg = next === 'light' ? '#f8fafc' : '#020617';
                setBgColor(manualBg);
                if (showThemeDebug) {
                    setThemeDebug(prev => ({
                        ...prev,
                        source: 'manual toggle',
                        appliedTheme: next,
                        background: manualBg,
                        luminance: null,
                    }));
                }
                window.__hypernodesVizThemeState = {
                    ...(window.__hypernodesVizThemeState || {}),
                    appliedTheme: next,
                    background: manualBg,
                    source: 'manual toggle',
                    preference: themePreference,
                };
                return next;
            });
        }, [themePreference]);

        // --- Expansion Logic ---
        const onToggleExpand = useCallback((nodeId) => {
          setNodes((nds) => {
            // Find the node to toggle
            const targetNode = nds.find(n => n.id === nodeId);
            if (!targetNode) return nds;
            
            const isCurrentlyExpanded = targetNode.data.isExpanded;
            const willExpand = !isCurrentlyExpanded;
            
            // If collapsing, we need to find all descendants and collapse them too
            const descendantsToCollapse = new Set();
            if (!willExpand) {
                // Robust descendant finding via parentNode
                const childrenMap = new Map();
                nds.forEach(n => {
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
                
                getDescendants(nodeId).forEach(id => descendantsToCollapse.add(id));
            }

            return nds.map((node) => {
              if (node.id === nodeId) {
                return {
                  ...node,
                  type: willExpand ? 'pipelineGroup' : 'custom',
                  data: { ...node.data, isExpanded: willExpand },
                  style: undefined, // Let ELK handle size
                };
              }
              
              // If we are collapsing the target, also collapse descendants
              if (!willExpand && descendantsToCollapse.has(node.id)) {
                  // Only update if it was expanded
                  if (node.data.isExpanded) {
                      return {
                          ...node,
                          type: 'custom', // Revert to custom (collapsed)
                          data: { ...node.data, isExpanded: false },
                          style: undefined
                      };
                  }
              }
              
              return node;
            });
          });
        }, [setNodes]);

        useEffect(() => {
          setNodes((nds) => nds.map(n => ({
            ...n,
            data: { ...n.data, theme, onToggleExpand: () => onToggleExpand(n.id) }
          })));
        }, [onToggleExpand, setNodes, theme]);

        // --- Visibility Logic with Error Boundary (Task 10 Partial) ---
        useEffect(() => {
           setNodes((nds) => {
             const expansionMap = new Map();
             nds.forEach(n => {
                if (n.data.nodeType === 'PIPELINE') expansionMap.set(n.id, n.data.isExpanded);
             });
             
             // Helper to check if all ancestors are expanded
             const isVisible = (node, map, allNodes) => {
                 if (!node.parentNode) return true;
                 const parent = map.get(node.parentNode);
                 // If parent not in map, it might be a group node that is NOT a pipeline (rare) or just missing.
                 // If parent is a pipeline and not expanded, hidden.
                 if (parent === false) return false;
                 
                 // Recurse up
                 const parentNode = allNodes.find(n => n.id === node.parentNode);
                 if (!parentNode) return true; // Should not happen if consistent
                 return isVisible(parentNode, map, allNodes);
             };
             
             return nds.map(n => {
                const visible = isVisible(n, expansionMap, nds);
                return { ...n, hidden: !visible };
             });
           });
        }, [nodes.map(n => n.data.isExpanded).join(',')]);

        const visibleEdges = useMemo(() => {
            const nodeMap = new Map(nodes.map(n => [n.id, n]));
            const parentMap = new Map();
            const expansionMap = new Map();
            nodes.forEach(n => {
                if (n.parentNode) parentMap.set(n.id, n.parentNode);
                if (n.data.nodeType === 'PIPELINE') expansionMap.set(n.id, n.data.isExpanded);
            });

            const getVisibleAncestor = (nodeId) => {
                let curr = nodeId;
                let candidate = nodeId;
                while (curr) {
                    const parent = parentMap.get(curr);
                    if (!parent) break;
                    if (expansionMap.get(parent) === false) candidate = parent;
                    curr = parent;
                }
                return candidate;
            };

            const newEdges = [];
            const processedEdges = new Set();

            edges.forEach(edge => {
                const sourceNode = nodeMap.get(edge.source);
                const targetNode = nodeMap.get(edge.target);

                // Skip edges that connect directly to an expanded pipeline wrapper.
                const sourceExpandedPipeline = sourceNode?.data?.nodeType === 'PIPELINE' && sourceNode.data.isExpanded;
                const targetExpandedPipeline = targetNode?.data?.nodeType === 'PIPELINE' && targetNode.data.isExpanded;
                if (sourceExpandedPipeline || targetExpandedPipeline) {
                    return;
                }

                const sourceVis = getVisibleAncestor(edge.source);
                const targetVis = getVisibleAncestor(edge.target);

                if (sourceVis && targetVis && sourceVis !== targetVis) {
                    const edgeId = `e_${sourceVis}_${targetVis}`;
                    if (!processedEdges.has(edgeId)) {
                         newEdges.push({ ...edge, id: edgeId, source: sourceVis, target: targetVis });
                         processedEdges.add(edgeId);
                    }
                }
            });
            return newEdges;
        }, [nodes, edges]);

        const { layoutedNodes, layoutedEdges, layoutError, graphHeight } = useLayout(nodes, visibleEdges);
        const { fitView } = useReactFlow();

        // --- Iframe Resize Logic (Task 2) ---
        useEffect(() => {
            if (graphHeight) {
                const desiredHeight = Math.max(600, graphHeight + 100);
                try {
                    // Try to resize the hosting iframe to avoid internal scrollbars
                    if (window.frameElement) {
                        window.frameElement.style.height = desiredHeight + 'px';
                    }
                } catch (e) {
                    // Ignore cross-origin errors or missing frameElement
                }
            }
        }, [graphHeight]);

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
            style=${{ backgroundColor: bgColor, height: Math.max(window.innerHeight - 20, graphHeight + 100) + 'px' }}
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
              fitView
              fitViewOptions=${{ padding: 0.1, minZoom: 0.5, maxZoom: 1 }}
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
            ${(layoutError || (!layoutedNodes.length && nodes.length) || (!nodes.length)) ? html`
                <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                  <div className="px-4 py-2 rounded-lg border text-xs font-mono bg-slate-900/80 text-amber-200 border-amber-500/40 shadow-lg pointer-events-auto">
                    ${layoutError ? `Layout error: ${layoutError}` : (!nodes.length ? 'No graph data' : 'Layout produced no nodes. Showing fallback.')}
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
