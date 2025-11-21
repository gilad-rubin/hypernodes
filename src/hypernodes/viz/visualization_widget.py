import base64
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

    # 1. Create Group Nodes for Levels (Nested Pipelines)
    # We need to map level_ids to React Flow nodes
    # The 'root' level is the canvas, so we skip it.
    # Nested levels (e.g. "root__nested_123") become Group nodes.
    
    level_map = {lvl["level_id"]: lvl for lvl in serialized_graph.get("levels", [])}
    
    # We need to create group nodes for all non-root levels
    # The ID of the group node should match the level_id so children can reference it as parentNode
    for level in serialized_graph.get("levels", []):
        level_id = level["level_id"]
        if level_id == "root":
            continue
            
        # Find the pipeline node that corresponds to this level to get its label
        # The parent_pipeline_node_id tells us which node *contains* this level
        # But we want the visual container for *this* level.
        # Actually, in React Flow, the "Group" node IS the PipelineNode that was expanded.
        # So we should check if we already created a node for this level's container.
        pass

    # Better approach:
    # 1. Iterate over all nodes.
    # 2. If a node is a PIPELINE and is_expanded=True, it becomes a Group node.
    #    Its ID is the node["id"].
    #    The *contents* of this pipeline will have level_id pointing to a level that corresponds to this node.
    #    We need to map level_ids to these Group Node IDs.
    
    # Map level_id -> Group Node ID
    level_to_group_node = {}
    
    # First pass: Identify expanded pipeline nodes and map their inner levels to them
    for node in serialized_graph.get("nodes", []):
        if node.get("node_type") == "PIPELINE" and node.get("is_expanded"):
            # This node is a container.
            # We need to find which level_id corresponds to its *interior*.
            # The serializer doesn't explicitly link node_id -> inner_level_id in the node dict,
            # but the level dict has 'parent_pipeline_node_id'.
            pass

    # Let's build the map from the levels side
    for level in serialized_graph.get("levels", []):
        pp_node_id = level.get("parent_pipeline_node_id")
        if pp_node_id:
            level_to_group_node[level["level_id"]] = pp_node_id

    # Second pass: Create all nodes
    for node in serialized_graph.get("nodes", []):
        node_id = node["id"]
        node_type = node.get("node_type", "STANDARD")
        
        # Determine initial expansion state based on depth
        level_id = node.get("level_id", "root")
        level_depth = level_id.count("__nested_")
        # If the node is a pipeline, should it be expanded?
        # If we want depth=1, we show root nodes. Pipelines at root (depth 0) are collapsed.
        # Wait, nodes in root are at depth 1?
        # Let's say root is depth 0.
        # Nodes in root are visible.
        # If a node in root is a PIPELINE, and we want depth=1, it should be COLLAPSED.
        # If we want depth=2, it should be EXPANDED.
        # So: is_expanded = (level_depth + 1) < initial_depth
        
        should_expand = (level_depth + 1) < initial_depth
        is_expanded = (node_type == "PIPELINE" and should_expand)

        
        # Determine parent group
        level_id = node.get("level_id", "root")
        parent_node_id = level_to_group_node.get(level_id)
        
        # Base node data
        rf_node = {
            "id": node_id,
            "type": "pipelineGroup" if (node_type == "PIPELINE" and is_expanded) else "custom",
            "data": {
                "label": node.get("label", ""),
                "nodeType": node_type,
                "inputs": node.get("inputs", []),
                "outputs": node.get("output_names", []),
                "theme": theme,
                "isExpanded": is_expanded,
            },
            "position": {"x": 0, "y": 0}, # Layout will handle this
        }
        
        if parent_node_id:
            rf_node["parentNode"] = parent_node_id
            rf_node["extent"] = "parent"
            
        if node_type == "PIPELINE" and is_expanded:
            rf_node["style"] = {"width": 600, "height": 400} # Initial size, ELK should resize
            
        nodes.append(rf_node)

    # Add synthetic input nodes for parameter edges
    # These should also be placed in the correct level/group
    input_levels = serialized_graph.get("input_levels", {})
    
    # We need to identify unique inputs (param name + level)
    # The serializer gives us edges starting with "input_X".
    # We need to create a node for "input_X" but potentially multiple times if they appear in different levels?
    # Actually, the serializer's 'input_levels' dict tells us the *definition* level.
    # But edges might originate from 'input_X' in a nested scope.
    # Let's look at the edges to find all input sources.
    
    input_sources = set()
    for edge in serialized_graph.get("edges", []):
        src = edge.get("source")
        if isinstance(src, str) and src.startswith("input_"):
            input_sources.add(src)
            
    for source_id in sorted(input_sources):
        input_name = source_id.replace("input_", "", 1)
        target_level = input_levels.get(input_name, "root")
        parent_node_id = level_to_group_node.get(target_level)
        
        nodes.append({
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
            # "extent": "parent" if parent_node_id else None
        })

    for edge in serialized_graph.get("edges", []):
        rf_edge = {
            "id": edge["id"],
            "source": edge["source"],
            "target": edge["target"],
            "animated": True,
            "style": {"stroke": "#64748b", "strokeWidth": 2},
            "data": {}
        }
        
        if edge.get("mapping_label"):
            rf_edge["label"] = edge["mapping_label"]
            rf_edge["data"]["label"] = edge["mapping_label"]
            
        edges.append(rf_edge)

    return {"nodes": nodes, "edges": edges}


def generate_widget_html(graph_data: Dict[str, Any]) -> str:
    """Generate an HTML document for React Flow rendering.

    Uses local vendored JS/CSS assets (assets/viz/*). Falls back to remote CDNs
    only if a file is missing at runtime.
    """

    graph_json = json.dumps(graph_data)

    def _read_asset(name: str, kind: str) -> Optional[str]:
        """Read an asset file from assets/viz; return None if missing."""
        path = Path(__file__).resolve().parent.parent.parent.parent / "assets" / "viz" / name
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        if kind == "js":
            return f"<script>{text}</script>"
        if kind == "css":
            return f"<style>{text}</style>"
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
        body {{ margin: 0; overflow: hidden; background: #030712; color: #e5e7eb; font-family: 'Inter', system-ui, -apple-system, sans-serif; }}
        .react-flow__attribution {{ display: none; }}
        #root {{ min-height: 100vh; min-width: 100vw; background: #030712; color: #e5e7eb; display: flex; align-items: center; justify-content: center; }}
        #fallback {{ font-size: 13px; letter-spacing: 0.4px; color: #94a3b8; }}
    </style>
    <!-- Prefer local vendored assets; fall back to CDN if missing -->
    {react_js or ''}
    {react_dom_js or ''}
    {htm_js or ''}
    {elk_js or ''}
    {rf_js or ''}
    {fallback_js if not all([react_js, react_dom_js, htm_js, elk_js, rf_js]) else ''}
</head>"""

    # JavaScript body as a raw string (no interpolation needed)
    html_body = """<body>
  <div id="root">
    <div id="fallback">Rendering interactive view…</div>
  </div>
  <script>
    // Surface JS errors inside the iframe so notebook users can see them.
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
        fallback.textContent = msg;
        fallback.style.color = "#f87171";
        fallback.style.fontFamily = "monospace";
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

      const { ReactFlow, Background, Controls, MiniMap, Handle, Position, ReactFlowProvider, useEdgesState, useNodesState, MarkerType, BaseEdge, getBezierPath, EdgeLabelRenderer, useReactFlow } = RF;
      const { useState, useEffect, useMemo, useCallback } = React;

      const html = htm.bind(React.createElement);
      const elk = new ELK();
      const defaultSize = { width: 280, height: 160 };

      // --- Icons ---
      const Icons = {
        Cube: () => html`
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5 text-indigo-400">
            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
            <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
            <line x1="12" y1="22.08" x2="12" y2="12"></line>
          </svg>
        `,
        CheckCircle: () => html`
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4 text-emerald-400">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
            <polyline points="22 4 12 14.01 9 11.01"></polyline>
          </svg>
        `,
        Play: () => html`
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
            <polygon points="5 3 19 12 5 21 5 3"></polygon>
          </svg>
        `,
        Layers: () => html`
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
            <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
            <polyline points="2 17 12 22 22 17"></polyline>
            <polyline points="2 12 12 17 22 12"></polyline>
          </svg>
        `,
        Function: () => html`
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4 text-blue-400">
            <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect>
            <line x1="7" y1="2" x2="7" y2="22"></line>
            <line x1="17" y1="2" x2="17" y2="22"></line>
            <line x1="2" y1="12" x2="22" y2="12"></line>
            <line x1="2" y1="7" x2="7" y2="7"></line>
            <line x1="2" y1="17" x2="7" y2="17"></line>
            <line x1="17" y1="17" x2="22" y2="17"></line>
            <line x1="17" y1="7" x2="22" y2="7"></line>
          </svg>
        `,
        Input: () => html`
           <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4 text-cyan-400">
            <polyline points="4 7 4 4 20 4 20 7"></polyline>
            <line x1="9" y1="20" x2="15" y2="20"></line>
            <line x1="12" y1="4" x2="12" y2="20"></line>
          </svg>
        `,
        Dual: () => html`
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4 text-fuchsia-400">
            <path d="M12 2a10 10 0 1 0 10 10H12V2z"></path>
            <path d="M12 12L2 12"></path>
            <path d="M12 12L12 22"></path>
          </svg>
        `
      };

      // --- Components ---

      const Header = ({ theme, onToggleTheme }) => html`
        <div className="absolute top-4 left-4 z-10 bg-slate-900/90 backdrop-blur border border-slate-700/50 rounded-xl p-3 shadow-xl flex items-center gap-3">
          <div className="p-2 bg-indigo-500/10 rounded-lg border border-indigo-500/20">
            <${Icons.Cube} />
          </div>
          <div>
            <h1 className="text-sm font-bold text-slate-100 leading-tight">HyperNodes DAG</h1>
            <p className="text-[10px] text-slate-400 font-medium">Interactive visualization widget</p>
          </div>
          <div className="h-8 w-[1px] bg-slate-800 mx-1"></div>
          <button 
            onClick=${onToggleTheme}
            className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[10px] text-slate-300 font-mono uppercase tracking-wider transition-colors border border-slate-700"
          >
            ${theme === 'light' ? 'Light' : 'Dark'}
          </button>
        </div>
      `;

      const CustomEdge = ({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, style = {}, markerEnd, label }) => {
        const [edgePath, labelX, labelY] = getBezierPath({
          sourceX,
          sourceY,
          sourcePosition,
          targetX,
          targetY,
          targetPosition,
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
                  className="px-2 py-1 rounded bg-slate-900/90 border border-slate-700 text-[10px] text-slate-300 font-mono shadow-md"
                >
                  ${label}
                </div>
              <//>
            ` : null}
          <//>
        `;
      };

      const CustomNode = ({ data, id }) => {
        const inputs = data.inputs || [];
        const outputs = data.outputs || [];
        const isExpanded = data.isExpanded;
        
        // Determine node style based on type
        let typeColor = "blue";
        let TypeIcon = Icons.Function;
        let typeLabel = "FUNCTION";
        
        if (data.nodeType === 'PIPELINE') {
          typeColor = "amber";
          TypeIcon = Icons.Layers;
          typeLabel = "PIPELINE";
        } else if (data.nodeType === 'DUAL') {
          typeColor = "fuchsia";
          TypeIcon = Icons.Dual;
          typeLabel = "DUAL";
        } else if (data.nodeType === 'INPUT') {
          typeColor = "cyan";
          TypeIcon = Icons.Input;
          typeLabel = "INPUT";
        }

        // Special rendering for expanded pipeline groups
        if (data.nodeType === 'PIPELINE' && isExpanded) {
          return html`
            <div className="relative w-full h-full rounded-2xl border-2 border-dashed border-amber-500/30 bg-amber-500/5 p-4 transition-all duration-300">
              <div className="absolute -top-3 left-4 px-2 bg-slate-950 text-xs font-bold text-amber-400 uppercase tracking-wider flex items-center gap-2 cursor-pointer hover:text-amber-300"
                   onClick=${data.onToggleExpand}>
                <${Icons.Layers} />
                ${data.label}
                <span className="text-[10px] opacity-70">(Click to collapse)</span>
              </div>
              <${Handle} type="target" position=${Position.Top} className="!w-0 !h-0 !opacity-0" />
              <${Handle} type="source" position=${Position.Bottom} className="!w-0 !h-0 !opacity-0" />
            </div>
          `;
        }

        return html`
          <div className="group relative w-[280px] rounded-xl border border-slate-800 bg-slate-950/80 shadow-2xl backdrop-blur-sm transition-all duration-300 hover:border-${typeColor}-500/50 hover:shadow-${typeColor}-500/10 hover:-translate-y-1 cursor-pointer"
               onClick=${data.nodeType === 'PIPELINE' ? data.onToggleExpand : undefined}>
            <!-- Header -->
            <div className="px-4 py-3 border-b border-slate-800/50 flex items-start justify-between gap-3">
              <div className="flex items-center gap-3 overflow-hidden">
                <div className="p-1.5 rounded-lg bg-${typeColor}-500/10 border border-${typeColor}-500/20 shrink-0">
                  <${TypeIcon} />
                </div>
                <div className="min-w-0">
                  <div className="text-[10px] font-bold tracking-wider text-${typeColor}-400 uppercase mb-0.5">${typeLabel}</div>
                  <div className="text-sm font-semibold text-slate-100 truncate" title=${data.label}>${data.label}</div>
                </div>
              </div>
              <div className="text-emerald-400 shrink-0">
                <${Icons.CheckCircle} />
              </div>
            </div>

            <!-- Body -->
            <div className="px-4 py-3 space-y-3">
              <!-- Batch Progress (Mock) -->
              <div>
                <div className="flex justify-between text-[10px] font-medium text-slate-400 mb-1.5">
                  <span className="uppercase tracking-wider">Batch</span>
                  <span className="text-slate-300">5 / 5</span>
                </div>
                <div className="flex gap-1 h-1">
                  <div className="flex-1 rounded-full bg-emerald-500"></div>
                  <div className="flex-1 rounded-full bg-emerald-500"></div>
                  <div className="flex-1 rounded-full bg-emerald-500"></div>
                  <div className="flex-1 rounded-full bg-emerald-500"></div>
                  <div className="flex-1 rounded-full bg-emerald-500"></div>
                </div>
              </div>

              <!-- Outputs -->
              ${outputs.length ? html`
                <div className="flex flex-wrap gap-2 pt-1">
                  ${outputs.map(out => html`
                    <span className="px-2.5 py-1 rounded-md border border-slate-700 bg-slate-800/50 text-xs font-medium text-slate-300 font-mono">
                      ${out}
                    </span>
                  `)}
                </div>
              ` : null}
            </div>

            <!-- Handles -->
            <${Handle} type="target" position=${Position.Top} className="!w-3 !h-3 !bg-slate-950 !border-2 !border-slate-600 group-hover:!border-${typeColor}-400 transition-colors" />
            <${Handle} type="source" position=${Position.Bottom} className="!w-3 !h-3 !bg-slate-950 !border-2 !border-slate-600 group-hover:!border-${typeColor}-400 transition-colors" />
            
            ${data.nodeType === 'PIPELINE' ? html`
               <div className="absolute -bottom-6 left-1/2 -translate-x-1/2 text-[10px] text-slate-500 opacity-0 group-hover:opacity-100 transition-opacity">
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

        useEffect(() => {
          if (!nodes.length) {
            setLayoutedNodes([]);
            setLayoutedEdges([]);
            return;
          }

          // Construct ELK graph with hierarchy
          const buildElkHierarchy = (nodes, edges) => {
            const nodeMap = new Map(nodes.map(n => [n.id, { ...n, children: [], edges: [] }]));
            const rootChildren = [];
            
            // Assign nodes to parents
            // Only assign if parent exists and is expanded (handled by visibility logic elsewhere, 
            // but for layout we need to know structure).
            // Actually, if a node is hidden, we shouldn't layout it?
            // Or ELK handles hidden nodes? No.
            // We should filter nodes passed to ELK.
            
            const visibleNodes = nodes.filter(n => !n.hidden);
            const visibleNodeIds = new Set(visibleNodes.map(n => n.id));
            const visibleEdges = edges.filter(e => visibleNodeIds.has(e.source) && visibleNodeIds.has(e.target));
            
            const visibleNodeMap = new Map(visibleNodes.map(n => [n.id, { ...n, children: [], edges: [] }]));
            
            visibleNodes.forEach(n => {
              if (n.parentNode && visibleNodeIds.has(n.parentNode)) {
                const parent = visibleNodeMap.get(n.parentNode);
                if (parent) parent.children.push(visibleNodeMap.get(n.id));
              } else {
                rootChildren.push(visibleNodeMap.get(n.id));
              }
            });
            
            const mapToElk = (n) => ({
              id: n.id,
              width: n.style?.width || defaultSize.width,
              height: n.style?.height || defaultSize.height,
              children: n.children.length ? n.children.map(mapToElk) : undefined,
              layoutOptions: {
                 'elk.padding': '[top=50,left=20,bottom=20,right=20]',
                 'elk.spacing.nodeNode': '60',
                 'elk.direction': 'DOWN',
              }
            });

            const elkGraph = {
              id: 'root',
              layoutOptions: {
                'elk.algorithm': 'layered',
                'elk.direction': 'DOWN',
                'elk.layered.spacing.nodeNodeBetweenLayers': '80',
                'elk.spacing.nodeNode': '60',
                'elk.hierarchyHandling': 'INCLUDE_CHILDREN',
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

          elk
            .layout(elkGraph)
            .then((graph) => {
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
                
                if (node.children) {
                  node.children.forEach(child => traverse(child, x, y));
                }
              };
              
              (graph.children || []).forEach(n => traverse(n));
              
              setLayoutedNodes(positionedNodes);
              setLayoutedEdges(visibleEdges);
            })
            .catch((err) => console.error('ELK layout error', err));
        }, [nodes, edges]);

        return { layoutedNodes, layoutedEdges };
      };

      const initialData = JSON.parse(document.getElementById('graph-data').textContent || '{"nodes":[],"edges":[]}');

      const App = () => {
        const [nodes, setNodes, onNodesChange] = useNodesState(initialData.nodes);
        const [edges, setEdges, onEdgesChange] = useEdgesState(initialData.edges);
        const [theme, setTheme] = useState('dark');

        // Theme detection
        useEffect(() => {
          const detectTheme = () => {
             let newTheme = 'dark';
             let customBg = null;

             try {
                 // Method: Get exact background color from parent (Trial 5 - worked)
                 const parentStyle = getComputedStyle(window.parent.document.documentElement);
                 const parentBg = parentStyle.getPropertyValue('--vscode-editor-background').trim();
                 
                 if (parentBg) {
                     customBg = parentBg;
                     
                     // Determine theme based on background brightness
                     // Helper to parse color and get brightness (0-255)
                     const getBrightness = (color) => {
                        // Handle rgb/rgba
                        const rgb = color.match(/\d+/g);
                        if (rgb && rgb.length >= 3) {
                            return (parseInt(rgb[0]) * 299 + parseInt(rgb[1]) * 587 + parseInt(rgb[2]) * 114) / 1000;
                        }
                        // Handle hex
                        if (color.startsWith('#')) {
                            let hex = color.substring(1);
                            if (hex.length === 3) hex = hex.split('').map(c => c + c).join('');
                            const r = parseInt(hex.substr(0, 2), 16);
                            const g = parseInt(hex.substr(2, 2), 16);
                            const b = parseInt(hex.substr(4, 2), 16);
                            return (r * 299 + g * 587 + b * 114) / 1000;
                        }
                        return 128; // Default to medium
                     };

                     const brightness = getBrightness(parentBg);
                     newTheme = brightness > 128 ? 'light' : 'dark';
                 } else {
                     // Fallback to class/attribute check if variable not found
                     const parentBody = window.parent.document.body;
                     const themeKind = parentBody.getAttribute('data-vscode-theme-kind');
                     if (themeKind === 'vscode-light') {
                         newTheme = 'light';
                     } else if (themeKind === 'vscode-dark') {
                         newTheme = 'dark';
                     } else if (parentBody.className.includes('vscode-light')) {
                         newTheme = 'light';
                     }
                 }

             } catch (e) {
                 // Fallback if cross-origin access fails
                 console.warn('Cannot access parent frame for theme detection', e);
                 if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
                     newTheme = 'light';
                 }
             }

             setTheme(newTheme);
             
             // Apply custom background if found
             if (customBg) {
                 document.documentElement.style.setProperty('--host-bg-color', customBg);
             }
          };
          
          detectTheme();
          
          // Observer for changes
          try {
              const observer = new MutationObserver(detectTheme);
              observer.observe(window.parent.document.body, { attributes: true, attributeFilter: ['class', 'data-vscode-theme-kind'] });
              return () => observer.disconnect();
          } catch (e) {
              // Fallback observer
              const observer = new MutationObserver(detectTheme);
              observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });
              return () => observer.disconnect();
          }
        }, []);

        const toggleTheme = useCallback(() => {
            setTheme(t => t === 'light' ? 'dark' : 'light');
        }, []);

        // Interactive Expansion Logic
        const onToggleExpand = useCallback((nodeId) => {
          setNodes((nds) => {
            return nds.map((node) => {
              if (node.id === nodeId) {
                const isExpanded = !node.data.isExpanded;
                return {
                  ...node,
                  type: isExpanded ? 'pipelineGroup' : 'custom',
                  data: { ...node.data, isExpanded },
                  style: isExpanded ? { width: 600, height: 400 } : undefined,
                };
              }
              return node;
            });
          });
        }, [setNodes]);

        // Visibility Logic
        useEffect(() => {
           setNodes((nds) => {
             const expansionMap = new Map();
             nds.forEach(n => {
                if (n.data.nodeType === 'PIPELINE') {
                   expansionMap.set(n.id, n.data.isExpanded);
                }
             });

             return nds.map(n => {
                if (n.parentNode) {
                   const parentExpanded = expansionMap.get(n.parentNode);
                   if (parentExpanded === false) {
                      return { ...n, hidden: true };
                   }
                   return { ...n, hidden: false };
                }
                return { ...n, hidden: false };
             });
           });
        }, [nodes.map(n => n.data.isExpanded).join(',')]);

        // Inject toggle handler
        useEffect(() => {
          setNodes((nds) => nds.map(n => ({
            ...n,
            data: {
              ...n.data,
              onToggleExpand: () => onToggleExpand(n.id)
            }
          })));
        }, [onToggleExpand, setNodes]);

        // Edge Hoisting Logic
        const visibleEdges = useMemo(() => {
            // Map nodeId -> parentId
            const parentMap = new Map();
            const expansionMap = new Map();
            nodes.forEach(n => {
                if (n.parentNode) parentMap.set(n.id, n.parentNode);
                if (n.data.nodeType === 'PIPELINE') expansionMap.set(n.id, n.data.isExpanded);
            });

            const getVisibleAncestor = (nodeId) => {
                let curr = nodeId;
                let candidate = nodeId;
                
                // Traverse up
                while (curr) {
                    const parent = parentMap.get(curr);
                    if (!parent) break; // Reached root
                    
                    // Check if parent is collapsed
                    const isParentExpanded = expansionMap.get(parent);
                    if (isParentExpanded === false) {
                        // Parent is collapsed, so IT is the visible representative
                        candidate = parent;
                    }
                    curr = parent;
                }
                return candidate;
            };

            const newEdges = [];
            const processedEdges = new Set();

            edges.forEach(edge => {
                const sourceVis = getVisibleAncestor(edge.source);
                const targetVis = getVisibleAncestor(edge.target);

                if (sourceVis !== targetVis) {
                    const edgeId = `e_${sourceVis}_${targetVis}`;
                    // Avoid duplicates
                    if (!processedEdges.has(edgeId)) {
                         newEdges.push({
                             ...edge,
                             id: edgeId,
                             source: sourceVis,
                             target: targetVis,
                         });
                         processedEdges.add(edgeId);
                    }
                }
            });
            return newEdges;
        }, [nodes, edges]);

        const { layoutedNodes, layoutedEdges } = useLayout(nodes, visibleEdges);

        // Custom edge styling
        const edgeOptions = {
            type: 'custom',
            animated: true,
            style: { stroke: theme === 'light' ? '#94a3b8' : '#64748b', strokeWidth: 2 },
            markerEnd: { type: MarkerType.ArrowClosed, color: theme === 'light' ? '#94a3b8' : '#64748b' },
        };

        const styledEdges = useMemo(() => layoutedEdges.map(e => ({ ...e, ...edgeOptions })), [layoutedEdges, theme]);

        return html`
          <div 
            className=${`w-full h-screen relative overflow-hidden`}
            style=${{ backgroundColor: 'var(--host-bg-color, ' + (theme === 'light' ? '#f8fafc' : '#020617') + ')' }}
          >
            <!-- Background Grid -->
            <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 pointer-events-none mix-blend-overlay"></div>
            
            <${Header} theme=${theme} onToggleTheme=${toggleTheme} />
            
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
            >
              <${Background} color=${theme === 'light' ? '#cbd5e1' : '#334155'} gap=${24} size=${1} variant="dots" />
              <${Controls} className=${theme === 'light' ? '!bg-white !text-slate-700 !border-slate-200 !shadow-xl' : '!bg-slate-900 !text-slate-200 !border-slate-700 !shadow-xl'} />
              <${MiniMap} 
                className=${theme === 'light' ? '!bg-white !border-slate-200 !shadow-xl rounded-lg overflow-hidden' : '!bg-slate-900 !border-slate-700 !shadow-xl rounded-lg overflow-hidden'}
                maskColor=${theme === 'light' ? 'rgba(241, 245, 249, 0.6)' : 'rgba(15, 23, 42, 0.6)'}
                nodeColor=${(n) => theme === 'light' ? '#cbd5e1' : '#475569'}
              />
            <//>
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

    def __init__(self, pipeline: Any, theme: str = "auto", depth: int = 1, **kwargs: Any):
        self.pipeline = pipeline
        self.theme = theme
        self.depth = depth
        html_content = self._generate_html(theme=theme, depth=depth)

        # Use srcdoc for better compatibility (VS Code, etc.)
        # We need to escape the HTML for the srcdoc attribute
        import html
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
            f'{css_fix}'
            f'<iframe srcdoc="{escaped_html}" '
            f'width="100%" height="600" frameborder="0" '
            f'style="border: none; width: 100%; height: 600px; display: block; background: transparent;" '
            f'sandbox="allow-scripts allow-same-origin allow-popups allow-forms">'
            f'</iframe>'
        )
        super().__init__(value=iframe_html, **kwargs)

    def _generate_html(self, theme: str = "auto", depth: int = 1) -> str:
        serializer = GraphSerializer(self.pipeline)
        # Serialize with depth=None to ensure we have all nodes for interactive expansion
        serialized_graph = serializer.serialize(depth=None)
        react_flow_data = transform_to_react_flow(serialized_graph, theme=theme, initial_depth=depth)
        return generate_widget_html(react_flow_data)

    def _repr_html_(self) -> str:
        """Fallback for environments that prefer raw HTML over widgets."""
        return self.value

