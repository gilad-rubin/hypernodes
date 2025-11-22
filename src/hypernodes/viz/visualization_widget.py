import html
import json
from pathlib import Path
from typing import Any, Dict, Optional

import ipywidgets as widgets

from .graph_serializer import GraphSerializer


def transform_to_react_flow(
    serialized_graph: Dict[str, Any],
    theme: str = "CYBERPUNK",
    initial_depth: int = 1,
) -> Dict[str, Any]:
    """Transform serialized graph data to React Flow node/edge structures."""
    nodes = []
    edges = []

    # Helper to find group node for a level
    level_to_group_node = {}

    # Map level_id -> Group Node ID
    for level in serialized_graph.get("levels", []):
        pp_node_id = level.get("parent_pipeline_node_id")
        if pp_node_id:
            level_to_group_node[level["level_id"]] = pp_node_id

    # --- Pre-processing: Identify Grouped Inputs ---
    # Use pre-computed groups from serializer
    grouped_inputs_by_consumer = serialized_graph.get("grouped_inputs", {})

    # Filter groups (min size 2) and track grouped params
    grouped_param_names = set()
    for consumer_id, groups in grouped_inputs_by_consumer.items():
        for group_type in ["bound", "unbound"]:
            params = groups.get(group_type, [])
            if len(params) >= 2:
                grouped_param_names.update(params)

    # Track created data nodes to avoid duplicates: node_id -> { output_name -> data_node_id }
    node_output_map = {}

    # Determine used outputs (only visualize outputs that are actually connected)
    edge_sources = set()
    for edge in serialized_graph.get("edges", []):
        edge_sources.add(edge["source"])

    # Second pass: Create all nodes (Function Nodes + Data Nodes)
    for node in serialized_graph.get("nodes", []):
        node_id = node["id"]
        node_type = node.get("node_type", "STANDARD")

        # Determine initial expansion state based on depth or pre-calculated state
        level_id = node.get("level_id", "root")
        
        if "is_expanded" in node:
            is_expanded = node["is_expanded"]
        else:
            if level_id == "root":
                level_depth = 0
            else:
                level_depth = level_id.count("__nested_")

            should_expand = level_depth < (initial_depth - 1)
            is_expanded = node_type == "PIPELINE" and should_expand

        # Determine parent group
        parent_node_id = level_to_group_node.get(level_id)

        # 1. Create FUNCTION/PIPELINE Node
        rf_node = {
            "id": node_id,
            "type": "pipelineGroup"
            if (node_type == "PIPELINE" and is_expanded)
            else "custom",
            "data": {
                "label": node.get("label", ""),
                "nodeType": node_type,
                "inputs": node.get("inputs", []),
                "outputs": node.get("output_names", []),  # Kept for metadata
                "theme": theme,
                "isExpanded": is_expanded,
            },
            "position": {"x": 0, "y": 0},
        }

        if parent_node_id:
            rf_node["parentNode"] = parent_node_id
            rf_node["extent"] = "parent"

        if node_type == "PIPELINE" and is_expanded:
            rf_node["style"] = {"width": 600, "height": 400}

        nodes.append(rf_node)

        # Create Grouped Input Nodes for this consumer
        if node_id in grouped_inputs_by_consumer:
            groups = grouped_inputs_by_consumer[node_id]
            for group_type in ["bound", "unbound"]:
                params = groups[group_type]
                if not params:
                    continue

                group_node_id = f"group_{node_id}_{group_type}"
                params.sort()

                group_label = "\n".join(params)

                nodes.append(
                    {
                        "id": group_node_id,
                        "type": "custom",
                        "data": {
                            "label": group_label,
                            "nodeType": "INPUT_GROUP",
                            "inputs": [],
                            "outputs": [],
                            "theme": theme,
                            "isBound": (group_type == "bound"),
                            "params": params,
                        },
                        "position": {"x": 0, "y": 0},
                        "parentNode": parent_node_id if parent_node_id else None,
                    }
                )

                # Create edge from Group -> Consumer
                edges.append(
                    {
                        "id": f"e_{group_node_id}_{node_id}",
                        "source": group_node_id,
                        "target": node_id,
                        "animated": False,
                        "style": {"stroke": "#64748b", "strokeWidth": 2},
                        "data": {},
                    }
                )

        # 2. Create DATA Nodes for Outputs
        if not is_expanded:
            output_names = node.get("output_names", [])
            for out_name in output_names:
                data_node_id = f"data_{node_id}_{out_name}"

                # Register
                if node_id not in node_output_map:
                    node_output_map[node_id] = {}
                node_output_map[node_id][out_name] = data_node_id

                # Create Data Node
                data_node = {
                    "id": data_node_id,
                    "type": "custom",
                    "data": {
                        "label": out_name,
                        "nodeType": "DATA",
                        "inputs": [],
                        "outputs": [],
                        "theme": theme,
                    },
                    "position": {"x": 0, "y": 0},
                    "parentNode": parent_node_id if parent_node_id else None,
                }
                nodes.append(data_node)

                # Create Edge: Function -> Data
                edges.append(
                    {
                        "id": f"e_func_data_{node_id}_{out_name}",
                        "source": node_id,
                        "target": data_node_id,
                        "animated": False,
                        "style": {"stroke": "#94a3b8", "strokeWidth": 2},
                        "data": {"isDataLink": True},
                    }
                )

    # 3. Create INPUT Nodes for Pipeline Inputs
    input_levels = serialized_graph.get("input_levels", {})
    input_sources = set()
    for edge in serialized_graph.get("edges", []):
        src = edge.get("source")
        if isinstance(src, str) and src.startswith("input_"):
            input_sources.add(src)

    for source_id in sorted(input_sources):
        input_name = source_id.replace("input_", "", 1)

        # Skip if grouped
        if input_name in grouped_param_names:
            continue

        target_level = input_levels.get(input_name, "root")
        parent_node_id = level_to_group_node.get(target_level)

        nodes.append(
            {
                "id": source_id,
                "type": "custom",
                "data": {
                    "label": input_name,
                    "nodeType": "INPUT",
                    "inputs": [],
                    "outputs": [input_name],
                    "theme": theme,
                },
                "position": {"x": 0, "y": 0},
                "parentNode": parent_node_id if parent_node_id else None,
            }
        )

    # 4. Create Edges (routing through DataNodes)
    for edge in serialized_graph.get("edges", []):
        source = edge["source"]
        target = edge["target"]
        edge_id = edge["id"]

        # Skip if source is a grouped input
        if isinstance(source, str) and source.startswith("input_"):
            param_name = source.replace("input_", "")
            if param_name in grouped_param_names:
                continue

        # Determine if source is a node with DataNode outputs
        source_data_node = None

        # Extract param name from edge ID if present
        prefix = f"e_{source}_{target}_"
        param_name = ""
        if edge_id.startswith(prefix):
            param_name = edge_id[len(prefix) :]

        # Try to resolve source to a DataNode
        if source in node_output_map:
            outputs = node_output_map[source]

            # 1. Try matching mapping_label source name if available (most accurate)
            mapping_label = edge.get("mapping_label", "")
            if mapping_label and "→" in mapping_label:
                src_part = mapping_label.split("→")[0].strip()
                if src_part in outputs:
                    source_data_node = outputs[src_part]

            # 2. Try matching param_name (target input name) to output name
            if not source_data_node and param_name and param_name in outputs:
                source_data_node = outputs[param_name]

            # 3. If single output, default to it
            elif not source_data_node and len(outputs) == 1:
                source_data_node = list(outputs.values())[0]

        # Final Source/Target
        final_source = source_data_node if source_data_node else source
        final_target = target

        rf_edge = {
            "id": edge_id,
            "source": final_source,
            "target": final_target,
            "animated": False,  # Solid lines (Task 3)
            "style": {"stroke": "#64748b", "strokeWidth": 2},
            "data": {},
        }

        # Hide mapping label? (Task 8)
        if edge.get("mapping_label"):
            # rf_edge["label"] = edge["mapping_label"] # Hidden
            rf_edge["data"]["label"] = edge["mapping_label"]

        edges.append(rf_edge)

    return {
        "nodes": nodes,
        "edges": edges,
        "event_index": serialized_graph.get("event_index", {}),
    }


def generate_widget_html(graph_data: Dict[str, Any]) -> str:
    """Generate an HTML document for React Flow rendering.

    Uses local vendored JS/CSS assets (assets/viz/*). Falls back to remote CDNs
    only if a file is missing at runtime.
    """

    graph_json = json.dumps(graph_data)

    def _read_asset(name: str, kind: str) -> Optional[str]:
        """Read an asset file from assets/viz; return None if missing."""
        try:
            path = (
                Path(__file__).resolve().parent.parent.parent.parent
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
        body {{ margin: 0; overflow: hidden; background: transparent; color: #e5e7eb; font-family: 'Inter', system-ui, -apple-system, sans-serif; }}
        .react-flow__attribution {{ display: none; }}
        #root {{ min-height: 100vh; min-width: 100vw; background: transparent; display: flex; align-items: center; justify-content: center; }}
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
             return html`
                <div className=${`px-3 py-2 rounded-lg border-2 border-dashed flex items-center gap-2
                    ${isLight
                        ? 'bg-cyan-50/50 border-cyan-200 text-cyan-800'
                        : 'bg-cyan-950/20 border-cyan-800/50 text-cyan-200'}
                `}>
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
        const boundInputs = data.inputs.filter(i => i.is_bound).length;

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
                let width = 240;
                let height = 60;
                
                if (n.data?.nodeType === 'DATA') {
                    width = 120;
                    height = 30;
                    if (n.data.label) width = Math.max(100, n.data.label.length * 8 + 40);
                } else if (n.data?.nodeType === 'INPUT') {
                    width = 140;
                    height = 40;
                } else if (n.data?.nodeType === 'INPUT_GROUP') {
                    width = 160;
                    // Dynamic height based on number of inputs
                    const paramCount = n.data.params ? n.data.params.length : 1;
                    height = 30 + (paramCount * 16);
                }
                
                if (n.style && n.style.width) width = n.style.width;
                if (n.style && n.style.height) height = n.style.height;

                return {
                  id: n.id,
                  width: width,
                  height: height,
                  children: n.children.length ? n.children.map(mapToElk) : undefined,
                  layoutOptions: {
                     'elk.padding': '[top=40,left=20,bottom=20,right=20]',
                     'elk.spacing.nodeNode': '40',
                     'elk.layered.spacing.nodeNodeBetweenLayers': '60',
                     'elk.direction': 'DOWN',
                  }
                };
            };

            const elkGraph = {
              id: 'root',
              layoutOptions: {
                'elk.algorithm': 'layered',
                'elk.direction': 'DOWN',
                'elk.layered.spacing.nodeNodeBetweenLayers': '60',
                'elk.spacing.nodeNode': '40',
                'elk.hierarchyHandling': 'INCLUDE_CHILDREN',
                'elk.layered.nodePlacement.strategy': 'BRANDES_KOEPF', // Better placement (Task 14)
                'elk.edgeRouting': 'SPLINES', // Better edge routing (Task 7)
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

        return { layoutedNodes, layoutedEdges, layoutError };
      };

      const initialData = JSON.parse(document.getElementById('graph-data').textContent || '{"nodes":[],"edges":[]}');

      const App = () => {
        const [nodes, setNodes, onNodesChange] = useNodesState(initialData.nodes);
        const [edges, setEdges, onEdgesChange] = useEdgesState(initialData.edges);
        const [theme, setTheme] = useState('dark');
        const [showMiniMap, setShowMiniMap] = useState(false);
        const [bgColor, setBgColor] = useState('transparent');

        // --- Theme Detection & Background (Task 1) ---
        useEffect(() => {
          const detectTheme = () => {
             let newTheme = 'dark';
             let detectedBg = 'transparent';
             try {
                 const parentStyle = getComputedStyle(window.parent.document.documentElement);
                 // Use VSCode background variable
                 const parentBg = parentStyle.getPropertyValue('--vscode-editor-background').trim();
                 
                 if (parentBg) {
                     detectedBg = parentBg;
                     const getBrightness = (color) => {
                        const rgb = color.match(/\d+/g);
                        if (rgb && rgb.length >= 3) return (parseInt(rgb[0]) * 299 + parseInt(rgb[1]) * 587 + parseInt(rgb[2]) * 114) / 1000;
                        return 128;
                     };
                     // Decide theme based on brightness of background
                     newTheme = getBrightness(parentBg) > 128 ? 'light' : 'dark';
                 } else {
                     // Fallback
                     const themeKind = window.parent.document.body.getAttribute('data-vscode-theme-kind');
                     if (themeKind === 'vscode-light' || window.parent.document.body.className.includes('vscode-light')) newTheme = 'light';
                 }
             } catch (e) {
                 if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) newTheme = 'light';
             }
             
             setTheme(newTheme);
             // Set specific backgrounds for better contrast if needed, or stick to transparent/detected
             setBgColor(detectedBg); 
             document.body.classList.toggle('light-mode', newTheme === 'light');
          };
          detectTheme();
          const observer = new MutationObserver(detectTheme);
          try { observer.observe(window.parent.document.body, { attributes: true, attributeFilter: ['class', 'data-vscode-theme-kind', 'style'] }); } catch(e) {}
          return () => observer.disconnect();
        }, []);

        const toggleTheme = useCallback(() => {
            setTheme(t => {
                const next = t === 'light' ? 'dark' : 'light';
                document.body.classList.toggle('light-mode', next === 'light');
                // If we toggle manually, we override the detected background to a preset
                setBgColor(next === 'light' ? '#f8fafc' : '#020617');
                return next;
            });
        }, []);

        // --- Expansion Logic ---
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
                const prefix = `${nodeId}__`;
                nds.forEach(n => {
                    if (n.id.startsWith(prefix)) {
                        descendantsToCollapse.add(n.id);
                    }
                });
            }

            return nds.map((node) => {
              if (node.id === nodeId) {
                return {
                  ...node,
                  type: willExpand ? 'pipelineGroup' : 'custom',
                  data: { ...node.data, isExpanded: willExpand },
                  style: willExpand ? { width: 600, height: 400 } : undefined,
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
            // ... (Edge hoisting logic kept same but robustified)
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

        const { layoutedNodes, layoutedEdges, layoutError } = useLayout(nodes, visibleEdges);
        const { fitView } = useReactFlow();

        // --- Resize Handling (Task 2) ---
        useEffect(() => {
            const handleResize = () => {
                fitView({ padding: 0.2, duration: 200 });
            };
            window.addEventListener('resize', handleResize);
            return () => window.removeEventListener('resize', handleResize);
        }, [fitView]);
        
        // Re-fit when layout changes
        useEffect(() => {
            if (layoutedNodes.length > 0) {
                // No animation duration for instant appearance
                window.requestAnimationFrame(() => fitView({ padding: 0.2, duration: 0 }));
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
            className=${`w-full h-screen relative overflow-hidden transition-colors duration-300`}
            style=${{ backgroundColor: bgColor }}
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
              minZoom=${0.1}
              className=${'bg-transparent'}
              panOnScroll=${true}
              zoomOnScroll=${false}
              panOnDrag=${true}
              zoomOnPinch=${true} 
              preventScrolling=${false}
              style=${{ width: '100%', height: '100%' }}
            >
              <${Background} color=${theme === 'light' ? '#94a3b8' : '#334155'} gap=${24} size=${1} variant="dots" />
              <${CustomControls} theme=${theme} onToggleTheme=${toggleTheme} showMiniMap=${showMiniMap} onToggleMiniMap=${() => setShowMiniMap(m => !m)} />
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


class PipelineWidget(widgets.HTML):
    """
    Widget for visualizing pipelines inside Jupyter/VS Code notebooks.
    """

    def __init__(
        self,
        pipeline: Any,
        theme: str = "auto",
        depth: Optional[int] = 1,
        group_inputs: bool = True,
        min_arg_group_size: Optional[int] = 2,
        **kwargs: Any,
    ):
        from .ui_handler import UIHandler
        
        self.pipeline = pipeline
        self.theme = theme
        self.depth = depth
        
        # 1. Get Graph Data once
        handler = UIHandler(
            self.pipeline,
            depth=depth,
            group_inputs=group_inputs,
            min_arg_group_size=min_arg_group_size,
        )
        serialized_graph = handler.get_full_graph_with_state(include_events=True)
        
        # 2. Calculate Height
        estimated_height = self._calculate_initial_height(serialized_graph)
        
        # 3. Generate HTML
        html_content = self._generate_html(serialized_graph, theme=theme, depth=depth)

        # Use srcdoc for better compatibility (VS Code, etc.)
        # We need to escape the HTML for the srcdoc attribute
        escaped_html = html.escape(html_content, quote=True)

        # CSS fix for VS Code white background on ipywidgets
        css_fix = """
        <style>
        .cell-output-ipywidget-background {
           background-color: transparent !important;
        }
        .jp-OutputArea-output {
           background-color: transparent;
        }
        </style>
        """

        iframe_html = (
            f"{css_fix}"
            f'<iframe srcdoc="{escaped_html}" '
            f'width="100%" height="{estimated_height}" frameborder="0" '
            f'style="border: none; width: 100%; height: {estimated_height}px; display: block; background: transparent;" '
            f'sandbox="allow-scripts allow-same-origin allow-popups allow-forms">'
            f"</iframe>"
        )
        super().__init__(value=iframe_html, **kwargs)

    def _calculate_initial_height(self, graph: Dict[str, Any]) -> int:
        """Estimate the required height based on graph structure."""
        try:
            nodes = graph.get("nodes", [])
            # Simple heuristic:
            # 1. Count total nodes
            # 2. Estimate "width" of the graph (max nodes in parallel?) - hard without layout.
            # 3. Estimate "depth" of the graph (longest path) - hard without layout.
            
            # Better heuristic for ELK Layered (DOWN direction):
            # Height depends on the number of ranks (levels).
            # Width depends on the max nodes in a rank.
            
            # Let's just count nodes for now as a proxy.
            num_nodes = len(nodes)
            
            base_height = 500
            row_height = 80 # Approx height per node/row
            
            # If we assume a roughly square aspect ratio or a long chain:
            # Sqrt(N) * row_height might be too small for long chains.
            # N * row_height might be too tall for wide graphs.
            
            # Let's try to find the "levels" in the graph if possible.
            # The graph has "levels" key which describes the hierarchy, not the layout ranks.
            
            # Fallback: Logarithmic/Linear scaling
            # 10 nodes -> 600px
            # 50 nodes -> 1000px
            # 100 nodes -> 1200px
            
            if num_nodes <= 10:
                return 500
            elif num_nodes <= 30:
                return 700
            elif num_nodes <= 50:
                return 900
            else:
                return 1200
                
        except Exception:
            return 600

    def _generate_html(self, graph_data: Dict[str, Any], theme: str = "auto", depth: Optional[int] = 1) -> str:
        react_flow_data = transform_to_react_flow(
            graph_data, theme=theme, initial_depth=depth
        )
        return generate_widget_html(react_flow_data)

    def _repr_html_(self) -> str:
        """Fallback for environments that prefer raw HTML over widgets."""
        return self.value
