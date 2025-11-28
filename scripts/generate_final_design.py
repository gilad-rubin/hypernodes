"""Generate final integrated design with both toggles."""

import json
from pathlib import Path

# Sample graph data with outputs including type hints
SAMPLE_GRAPH = {
    "nodes": [
        {"id": "input_x", "position": {"x": 0, "y": 0}, "type": "custom", "data": {"label": "x", "nodeType": "DATA", "typeHint": "int", "theme": "dark"}},
        {"id": "input_y", "position": {"x": 0, "y": 0}, "type": "custom", "data": {"label": "y", "nodeType": "DATA", "typeHint": "int", "theme": "dark"}},
        {"id": "add", "position": {"x": 0, "y": 0}, "type": "custom", "data": {"label": "add", "nodeType": "FUNCTION", "functionName": "add", "theme": "dark"}},
        {"id": "output_sum", "position": {"x": 0, "y": 0}, "type": "custom", "data": {"label": "sum", "nodeType": "DATA", "typeHint": "int", "theme": "dark", "sourceId": "add"}},
        {"id": "multiply", "position": {"x": 0, "y": 0}, "type": "custom", "data": {"label": "multiply", "nodeType": "FUNCTION", "functionName": "multiply", "theme": "dark"}},
        {"id": "output_product", "position": {"x": 0, "y": 0}, "type": "custom", "data": {"label": "product", "nodeType": "DATA", "typeHint": "float", "theme": "dark", "sourceId": "multiply"}},
        {"id": "output_remainder", "position": {"x": 0, "y": 0}, "type": "custom", "data": {"label": "remainder", "nodeType": "DATA", "typeHint": "int", "theme": "dark", "sourceId": "multiply"}},
        {"id": "combine", "position": {"x": 0, "y": 0}, "type": "custom", "data": {"label": "combine", "nodeType": "FUNCTION", "functionName": "combine", "theme": "dark"}},
        {"id": "output_result", "position": {"x": 0, "y": 0}, "type": "custom", "data": {"label": "result", "nodeType": "DATA", "typeHint": "Dict[str, Any]", "theme": "dark", "sourceId": "combine"}},
    ],
    "edges": [
        {"id": "e1", "source": "input_x", "target": "add"},
        {"id": "e2", "source": "input_y", "target": "add"},
        {"id": "e3", "source": "add", "target": "output_sum"},
        {"id": "e4", "source": "output_sum", "target": "multiply"},
        {"id": "e5", "source": "input_y", "target": "multiply"},
        {"id": "e6", "source": "multiply", "target": "output_product"},
        {"id": "e7", "source": "multiply", "target": "output_remainder"},
        {"id": "e8", "source": "output_product", "target": "combine"},
        {"id": "e9", "source": "output_sum", "target": "combine"},
        {"id": "e10", "source": "combine", "target": "output_result"},
    ],
    "meta": {
        "theme_preference": "auto",
        "initial_depth": 1,
        "separate_outputs": False,
        "show_types": True
    }
}

HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>HyperNodes Visualization - Final Design</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reactflow@11.10.1/dist/style.css" />
    <style>
        body { margin: 0; overflow: hidden; background: #0f172a; color: #e5e7eb; font-family: 'Inter', system-ui, -apple-system, sans-serif; }
        .react-flow__attribution { display: none; }
        #root { height: 100vh; width: 100vw; background: transparent; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.production.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.production.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/htm@3.1.1/dist/htm.umd.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/elkjs@0.8.2/lib/elk.bundled.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/reactflow@11.10.1/dist/umd/index.js"></script>
</head>
<body>
  <div id="root"></div>
  <script type="module">
    const React = window.React;
    const ReactDOM = window.ReactDOM;
    const RF = window.ReactFlow;
    const htm = window.htm;
    const ELK = window.ELK;

    const { ReactFlow, Background, Handle, Position, ReactFlowProvider, useEdgesState, useNodesState, MarkerType, BaseEdge, getBezierPath, useReactFlow, Panel } = RF;
    const { useState, useEffect, useMemo, useCallback } = React;

    const html = htm.bind(React.createElement);
    const elk = new ELK();

    // =============================================
    // TOGGLE CONTROLS - Top Right Corner
    // =============================================
    const ViewControls = ({ separateOutputs, showTypes, onToggleSeparate, onToggleTypes, theme }) => {
        const isLight = theme === 'light';
        
        const containerClass = isLight 
            ? "bg-white/95 border-slate-200 shadow-lg" 
            : "bg-slate-900/95 border-slate-700 shadow-xl";
        const labelClass = isLight ? "text-slate-600" : "text-slate-400";
        const activeClass = isLight ? "bg-indigo-500" : "bg-indigo-600";
        const inactiveClass = isLight ? "bg-slate-300" : "bg-slate-600";
        
        return html`
            <div className="absolute top-4 right-4 z-50">
                <div className=${"flex flex-col gap-3 px-4 py-3 rounded-xl border backdrop-blur-sm " + containerClass}>
                    <!-- Separate Outputs Toggle -->
                    <div className="flex items-center justify-between gap-4">
                        <span className=${"text-xs font-medium " + labelClass}>Separate outputs</span>
                        <button
                            onClick=${onToggleSeparate}
                            className=${"relative w-9 h-5 rounded-full transition-all duration-300 " + (separateOutputs ? activeClass : inactiveClass)}
                        >
                            <div className=${"absolute top-0.5 w-4 h-4 rounded-full bg-white shadow-md transition-all duration-300 " + (separateOutputs ? "left-4" : "left-0.5")}></div>
                        </button>
                    </div>
                    
                    <!-- Show Types Toggle -->
                    <div className="flex items-center justify-between gap-4">
                        <span className=${"text-xs font-medium " + labelClass}>Show types</span>
                        <button
                            onClick=${onToggleTypes}
                            className=${"relative w-9 h-5 rounded-full transition-all duration-300 " + (showTypes ? activeClass : inactiveClass)}
                        >
                            <div className=${"absolute top-0.5 w-4 h-4 rounded-full bg-white shadow-md transition-all duration-300 " + (showTypes ? "left-4" : "left-0.5")}></div>
                        </button>
                    </div>
                </div>
            </div>
        `;
    };

    // Icons
    const Icons = {
        Function: () => html`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect><line x1="7" y1="2" x2="7" y2="22"></line><line x1="17" y1="2" x2="17" y2="22"></line><line x1="2" y1="12" x2="22" y2="12"></line></svg>`,
        Data: () => html`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-3 h-3"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>`,
    };

    // =============================================
    // OUTPUT SECTION - Design A (Arrow List) with wider box
    // =============================================
    const OutputsSection = ({ outputs, showTypes, isLight }) => {
        if (!outputs || outputs.length === 0) return null;
        
        const bgClass = isLight ? "bg-slate-50/80" : "bg-slate-900/50";
        const textClass = isLight ? "text-slate-600" : "text-slate-400";
        const arrowClass = isLight ? "text-emerald-500" : "text-emerald-400";
        const typeClass = isLight ? "text-slate-400" : "text-slate-500";
        
        return html`
            <div className=${"px-4 py-3 border-t transition-all duration-300 " + bgClass + " " + (isLight ? "border-slate-100" : "border-slate-800/50")}>
                <div className="flex flex-col items-center gap-1.5">
                    ${outputs.map(out => html`
                        <div key=${out.name} className=${"flex items-center justify-center gap-2 text-xs whitespace-nowrap " + textClass}>
                            <span className=${arrowClass}>→</span>
                            <span className="font-mono font-medium">${out.name}</span>
                            ${showTypes && out.type ? html`<span className=${"font-mono " + typeClass}>: ${out.type}</span>` : null}
                        </div>
                    `)}
                </div>
            </div>
        `;
    };

    // =============================================
    // CUSTOM NODE COMPONENT
    // =============================================
    const CustomNode = ({ data, id }) => {
        const theme = data.theme;
        const isLight = theme === 'light';
        const isOutput = data.sourceId != null;
        const showAsOutput = data.separateOutputs && isOutput;
        const showTypes = data.showTypes;
        
        // DATA nodes (inputs or separate outputs)
        if (data.nodeType === 'DATA') {
            const baseClass = "px-4 py-1.5 w-full relative rounded-full border shadow-sm flex items-center justify-center gap-2 transition-all duration-300 whitespace-nowrap";
            const themeClass = isLight 
                ? "bg-white border-slate-200 text-slate-700" 
                : "bg-slate-900 border-slate-700 text-slate-300";
            const outputHighlight = showAsOutput ? "ring-2 ring-emerald-500/30" : "";
            const iconColor = isLight ? "text-slate-400" : "text-slate-500";
            const typeClass = isLight ? "text-slate-400" : "text-slate-500";
            
            return html`
                <div className=${baseClass + " " + themeClass + " " + outputHighlight}>
                    <span className=${iconColor}><${Icons.Data} /></span>
                    <span className="text-xs font-mono font-medium">${data.label}</span>
                    ${showTypes && data.typeHint ? html`<span className=${"text-[10px] font-mono " + typeClass}>: ${data.typeHint}</span>` : null}
                    <${Handle} type="target" position=${Position.Top} className="!w-2 !h-2 !opacity-0" />
                    <${Handle} type="source" position=${Position.Bottom} className="!w-2 !h-2 !opacity-0" />
                </div>
            `;
        }

        // Function nodes - may show combined outputs
        const outputs = data.outputs || [];
        const showCombined = !data.separateOutputs && outputs.length > 0;
        
        const containerClass = isLight 
            ? "bg-white/90 border-slate-200" 
            : "bg-slate-950/90 border-slate-800";
        const borderClass = isLight ? "border-slate-100" : "border-slate-800/50";
        const iconBgClass = isLight ? "bg-indigo-50 text-indigo-600" : "bg-indigo-500/10 text-indigo-400";
        const labelTypeClass = isLight ? "text-indigo-600" : "text-indigo-400";
        const labelClass = isLight ? "text-slate-800" : "text-slate-100";
        
        return html`
            <div className=${"group relative w-full rounded-lg border shadow-lg transition-all duration-300 overflow-hidden " + containerClass}>
                <div className=${"px-4 py-2.5 flex items-center gap-3 " + (showCombined ? "border-b " + borderClass : "")}>
                    <div className=${"p-1.5 rounded-md shrink-0 " + iconBgClass}>
                        <${Icons.Function} />
                    </div>
                    <div className="min-w-0 flex-1">
                        <div className=${"text-[9px] font-bold tracking-wider uppercase mb-0.5 " + labelTypeClass}>FUNCTION</div>
                        <div className=${"text-sm font-semibold truncate " + labelClass}>${data.label}</div>
                    </div>
                </div>
                
                ${showCombined ? html`<${OutputsSection} outputs=${outputs} showTypes=${showTypes} isLight=${isLight} />` : null}
                
                <${Handle} type="target" position=${Position.Top} className="!w-2 !h-2 !opacity-0" />
                <${Handle} type="source" position=${Position.Bottom} className="!w-2 !h-2 !opacity-0" />
            </div>
        `;
    };

    const nodeTypes = { custom: CustomNode };

    // =============================================
    // LAYOUT HOOK
    // =============================================
    const useLayout = (nodes, edges, showTypes) => {
        const [layoutedNodes, setLayoutedNodes] = useState([]);
        const [layoutedEdges, setLayoutedEdges] = useState(edges);

        useEffect(() => {
            if (!nodes.length) return;
            
            const elkGraph = {
                id: 'root',
                layoutOptions: {
                    'elk.algorithm': 'layered',
                    'elk.direction': 'DOWN',
                    'elk.layered.spacing.nodeNodeBetweenLayers': '60',
                    'elk.spacing.nodeNode': '30',
                },
                children: nodes.filter(n => !n.hidden).map(n => {
                    // Wider nodes to prevent text wrapping
                    let width = 240;  // Increased from 200
                    let height = 72;
                    
                    if (n.data.nodeType === 'DATA') {
                        // Calculate width based on content
                        const labelLen = (n.data.label || '').length;
                        const typeLen = (showTypes && n.data.typeHint) ? n.data.typeHint.length : 0;
                        width = Math.max(180, 60 + (labelLen + typeLen) * 7);
                        height = 36;
                    } else if (!n.data.separateOutputs && n.data.outputs?.length) {
                        // Calculate width for combined outputs
                        const maxOutputLen = Math.max(...n.data.outputs.map(o => {
                            const nameLen = o.name.length;
                            const typeLen = (showTypes && o.type) ? o.type.length + 2 : 0;
                            return nameLen + typeLen;
                        }));
                        width = Math.max(240, 80 + maxOutputLen * 7);
                        height = 90 + (n.data.outputs.length * 24);
                    }
                    
                    return { id: n.id, width, height };
                }),
                edges: edges.filter(e => !nodes.find(n => n.id === e.source)?.hidden && !nodes.find(n => n.id === e.target)?.hidden).map(e => ({
                    id: e.id,
                    sources: [e.source],
                    targets: [e.target],
                })),
            };

            elk.layout(elkGraph).then(graph => {
                const positioned = [];
                (graph.children || []).forEach(child => {
                    const original = nodes.find(n => n.id === child.id);
                    if (original) {
                        positioned.push({
                            ...original,
                            position: { x: child.x || 0, y: child.y || 0 },
                            style: { width: child.width, height: child.height },
                        });
                    }
                });
                setLayoutedNodes(positioned);
                setLayoutedEdges(edges.filter(e => positioned.some(n => n.id === e.source) && positioned.some(n => n.id === e.target)));
            });
        }, [nodes, edges, showTypes]);

        return { layoutedNodes, layoutedEdges };
    };

    // =============================================
    // MAIN APP
    // =============================================
    const initialData = __GRAPH_JSON__;

    const App = () => {
        // Default: separate_outputs=False, show_types=True
        const [separateOutputs, setSeparateOutputs] = useState(initialData.meta?.separate_outputs ?? false);
        const [showTypes, setShowTypes] = useState(initialData.meta?.show_types ?? true);
        const [theme, setTheme] = useState('dark');
        const { fitView } = useReactFlow();

        // Build nodes based on separateOutputs mode
        const { nodes, edges } = useMemo(() => {
            const baseNodes = initialData.nodes;
            const baseEdges = initialData.edges;
            
            if (separateOutputs) {
                // Show all nodes including outputs
                return {
                    nodes: baseNodes.map(n => ({
                        ...n,
                        data: { ...n.data, theme, separateOutputs: true, showTypes },
                    })),
                    edges: baseEdges,
                };
            } else {
                // Combined mode: hide output DataNodes, add outputs to function nodes
                const outputNodes = new Set(baseNodes.filter(n => n.data.sourceId).map(n => n.id));
                const functionOutputs = {};
                baseNodes.forEach(n => {
                    if (n.data.sourceId) {
                        if (!functionOutputs[n.data.sourceId]) functionOutputs[n.data.sourceId] = [];
                        functionOutputs[n.data.sourceId].push({ name: n.data.label, type: n.data.typeHint });
                    }
                });
                
                return {
                    nodes: baseNodes
                        .filter(n => !outputNodes.has(n.id))
                        .map(n => ({
                            ...n,
                            data: { 
                                ...n.data, 
                                theme, 
                                separateOutputs: false,
                                showTypes,
                                outputs: functionOutputs[n.id] || [],
                            },
                        })),
                    edges: baseEdges.filter(e => {
                        if (outputNodes.has(e.target)) return false;
                        return true;
                    }).map(e => {
                        if (outputNodes.has(e.source)) {
                            const outputNode = baseNodes.find(n => n.id === e.source);
                            if (outputNode?.data?.sourceId) {
                                return { ...e, source: outputNode.data.sourceId };
                            }
                        }
                        return e;
                    }),
                };
            }
        }, [separateOutputs, showTypes, theme]);

        const { layoutedNodes, layoutedEdges } = useLayout(nodes, edges, showTypes);

        useEffect(() => {
            if (layoutedNodes.length) {
                setTimeout(() => fitView({ padding: 0.2, duration: 400 }), 100);
            }
        }, [layoutedNodes, fitView]);

        const handleToggleSeparate = useCallback(() => {
            setSeparateOutputs(prev => !prev);
        }, []);

        const handleToggleTypes = useCallback(() => {
            setShowTypes(prev => !prev);
        }, []);

        const edgeOptions = {
            style: { stroke: theme === 'light' ? '#94a3b8' : '#64748b', strokeWidth: 2 },
            markerEnd: { type: MarkerType.ArrowClosed, color: theme === 'light' ? '#94a3b8' : '#64748b' },
        };

        const styledEdges = layoutedEdges.map(e => ({ ...e, ...edgeOptions }));

        return html`
            <div className="w-full h-screen relative" style=${{ backgroundColor: theme === 'light' ? '#f8fafc' : '#0f172a' }}>
                <${ReactFlow}
                    nodes=${layoutedNodes}
                    edges=${styledEdges}
                    nodeTypes=${nodeTypes}
                    fitView
                    minZoom=${0.5}
                    maxZoom=${1.5}
                >
                    <${Background} color=${theme === 'light' ? '#94a3b8' : '#334155'} gap=${24} size=${1} variant="dots" />
                <//>
                <${ViewControls} 
                    separateOutputs=${separateOutputs} 
                    showTypes=${showTypes}
                    onToggleSeparate=${handleToggleSeparate}
                    onToggleTypes=${handleToggleTypes}
                    theme=${theme} 
                />
            </div>
        `;
    };

    const root = ReactDOM.createRoot(document.getElementById('root'));
    root.render(html`<${ReactFlowProvider}><${App} /><//>`);
  </script>
</body>
</html>
'''


def main():
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    html = HTML_TEMPLATE.replace("__GRAPH_JSON__", json.dumps(SAMPLE_GRAPH))
    filepath = output_dir / "final_design_preview.html"
    filepath.write_text(html)
    
    print(f"✓ Final design preview saved to: {filepath}")
    print("\nFeatures:")
    print("  - Separate outputs toggle (default: OFF)")
    print("  - Show types toggle (default: ON)")
    print("  - Wider nodes to prevent text wrapping")
    print("  - Centered outputs with arrow style")
    print("  - Both toggles in top-right corner")


if __name__ == "__main__":
    main()

