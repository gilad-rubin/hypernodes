import base64
import json
from pathlib import Path
from typing import Any, Dict, Optional

import ipywidgets as widgets

from .graph_serializer import GraphSerializer


def transform_to_react_flow(
    serialized_graph: Dict[str, Any],
    theme: str = "CYBERPUNK",
) -> Dict[str, Any]:
    """Transform serialized graph data to React Flow node/edge structures."""
    nodes = []
    edges = []

    for node in serialized_graph.get("nodes", []):
        node_type = node.get("node_type", "STANDARD")
        nodes.append(
            {
                "id": node["id"],
                "type": "pipelineGroup"
                if node_type == "PIPELINE" and node.get("is_expanded")
                else "custom",
                "data": {
                    "label": node.get("label", ""),
                    "nodeType": node_type,
                    "inputs": node.get("inputs", []),
                    "outputs": node.get("output_names", []),
                    "theme": theme,
                },
                "position": {"x": 0, "y": 0},
            }
        )

    # Add synthetic input nodes for parameter edges
    input_sources = {
        edge.get("source")
        for edge in serialized_graph.get("edges", [])
        if isinstance(edge.get("source"), str) and edge.get("source", "").startswith("input_")
    }
    for source_id in sorted(input_sources):
        input_name = source_id.replace("input_", "", 1)
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
            }
        )

    for edge in serialized_graph.get("edges", []):
        rf_edge = {
            "id": edge["id"],
            "source": edge["source"],
            "target": edge["target"],
            "animated": True,
            "style": {"stroke": "#52525b"},
        }
        if edge.get("mapping_label"):
            rf_edge["label"] = edge["mapping_label"]
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

      const { ReactFlow, Background, Controls, MiniMap, Handle, Position, ReactFlowProvider, useEdgesState, useNodesState } = RF;

    const html = htm.bind(React.createElement);
    const elk = new ELK();
    const defaultSize = { width: 280, height: 140 };

    const CustomNode = ({ data }) => {
      const inputs = data.inputs || [];
      const outputs = data.outputs || [];
      const badge = (text, tone) => html`<span className=${`text-xs px-2 py-1 rounded-full border ${tone}`}>${text}</span>`;

      return html`
        <div className="relative rounded-xl border border-slate-700 bg-slate-900/70 shadow-lg backdrop-blur">
          <div className="px-3 py-2 border-b border-slate-800 flex items-center justify-between gap-3">
            <span className="text-sm font-semibold text-slate-100 truncate">${data.label}</span>
            ${badge(
                data.nodeType === 'PIPELINE' ? 'PIPELINE' :
                data.nodeType === 'DUAL' ? 'DUAL' :
                data.nodeType === 'INPUT' ? 'INPUT' : 'NODE',
                data.nodeType === 'PIPELINE' ? 'border-amber-400 text-amber-200 bg-amber-500/10' :
                data.nodeType === 'DUAL' ? 'border-fuchsia-400 text-fuchsia-200 bg-fuchsia-500/10' :
                data.nodeType === 'INPUT' ? 'border-cyan-400 text-cyan-200 bg-cyan-500/10' :
                'border-blue-400 text-blue-200 bg-blue-500/10'
              )}
          </div>
          <div className="px-3 py-3 space-y-2">
            ${inputs.length ? html`
                <div className="flex flex-wrap gap-1 items-center">
                  <span className="text-[11px] uppercase tracking-wide text-slate-400">Inputs</span>
                  ${inputs.map((inp) => badge(inp.name, inp.isBound ? 'border-slate-500 text-slate-300 bg-slate-700/40' : 'border-cyan-400 text-cyan-200 bg-cyan-500/10'))}
                </div>
              ` : null}
            ${outputs.length ? html`
                <div className="flex flex-wrap gap-1 items-center">
                  <span className="text-[11px] uppercase tracking-wide text-slate-400">Outputs</span>
                  ${outputs.map((out) => badge(out, 'border-fuchsia-400 text-fuchsia-200 bg-fuchsia-500/10'))}
                </div>
              ` : null}
          </div>
          <${Handle} type="target" position=${Position.Left} className="!bg-transparent !border !border-slate-500" />
          <${Handle} type="source" position=${Position.Right} className="!bg-transparent !border !border-slate-500" />
        </div>
      `;
    };

    const nodeTypes = { custom: CustomNode, pipelineGroup: CustomNode };

    const useLayout = (nodes, edges) => {
      const [layoutedNodes, setLayoutedNodes] = useState([]);
      const [layoutedEdges, setLayoutedEdges] = useState([]);

      useEffect(() => {
        if (!nodes.length) {
          setLayoutedNodes([]);
          setLayoutedEdges([]);
          return;
        }

        const elkNodes = nodes.map((n) => ({
          id: n.id,
          width: n.width || defaultSize.width,
          height: n.height || defaultSize.height,
        }));

        const elkEdges = edges.map((e) => ({
          id: e.id,
          sources: [e.source],
          targets: [e.target],
        }));

        elk
          .layout({
            id: 'root',
            layoutOptions: {
              'elk.algorithm': 'layered',
              'elk.direction': 'DOWN',
              'elk.layered.spacing.nodeNodeBetweenLayers': '80',
              'elk.spacing.nodeNode': '60',
            },
            children: elkNodes,
            edges: elkEdges,
          })
          .then((graph) => {
            const positioned = (graph.children || []).map((n) => {
              const match = nodes.find((node) => node.id === n.id) || {};
              return {
                ...match,
                position: { x: n.x || 0, y: n.y || 0 },
                width: n.width,
                height: n.height,
                style: { width: n.width, height: n.height },
              };
            });
            setLayoutedNodes(positioned);
            setLayoutedEdges(edges);
          })
          .catch((err) => console.error('ELK layout error', err));
      }, [nodes, edges]);

      return { layoutedNodes, layoutedEdges };
    };

    const initialData = JSON.parse(document.getElementById('graph-data').textContent || '{"nodes":[],"edges":[]}');

    const App = () => {
      const [nodes, , onNodesChange] = useNodesState(initialData.nodes);
      const [edges, , onEdgesChange] = useEdgesState(initialData.edges);
      const { layoutedNodes, layoutedEdges } = useLayout(nodes, edges);

      return html`
        <div className="w-full h-screen bg-slate-950">
          <${ReactFlow}
            nodes=${layoutedNodes}
            edges=${layoutedEdges}
            nodeTypes=${nodeTypes}
            onNodesChange=${onNodesChange}
            onEdgesChange=${onEdgesChange}
            fitView
            minZoom=${0.1}
            className="bg-slate-950 text-slate-50"
          >
            <${Background} color="#1f2937" gap=${28} />
            <${Controls} className="!bg-slate-900 !text-slate-200 !border-slate-700" />
            <${MiniMap} className="!bg-slate-900 !border-slate-800" />
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

    def __init__(self, pipeline: Any, theme: str = "CYBERPUNK", **kwargs: Any):
        self.pipeline = pipeline
        html_content = self._generate_html(theme=theme)

        html_b64 = base64.b64encode(html_content.encode("utf-8")).decode("utf-8")
        iframe_html = (
            f'<iframe src="data:text/html;base64,{html_b64}" '
            f'width="100%" height="600px" frameborder="0"></iframe>'
        )
        super().__init__(value=iframe_html, **kwargs)

    def _generate_html(self, theme: str = "CYBERPUNK") -> str:
        serializer = GraphSerializer(self.pipeline)
        serialized_graph = serializer.serialize()
        react_flow_data = transform_to_react_flow(serialized_graph, theme=theme)
        return generate_widget_html(react_flow_data)
