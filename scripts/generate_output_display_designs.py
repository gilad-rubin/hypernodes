"""Generate HTML files with different output display designs for combined mode."""

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
    "meta": {"theme_preference": "auto", "initial_depth": 1, "separate_outputs": False}
}

HTML_HEAD = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Output Display - {design_name}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reactflow@11.10.1/dist/style.css" />
    <style>
        body {{ margin: 0; overflow: hidden; background: #0f172a; color: #e5e7eb; font-family: 'Inter', system-ui, -apple-system, sans-serif; }}
        .react-flow__attribution {{ display: none; }}
        #root {{ height: 100vh; width: 100vw; background: transparent; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.production.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.production.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/htm@3.1.1/dist/htm.umd.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/elkjs@0.8.2/lib/elk.bundled.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/reactflow@11.10.1/dist/umd/index.js"></script>
</head>'''

# Toggle component (Design B - Segmented Control)
TOGGLE_COMPONENT = '''
const SeparateOutputsToggle = ({ separateOutputs, onToggle, theme }) => {
    const isLight = theme === 'light';
    
    const containerClass = isLight 
        ? "bg-white/95 border-slate-200" 
        : "bg-slate-900/95 border-slate-700";
    const activeClass = isLight ? "bg-indigo-500 text-white" : "bg-indigo-600 text-white";
    const inactiveClass = isLight 
        ? "text-slate-500 hover:text-slate-700 hover:bg-slate-50" 
        : "text-slate-400 hover:text-slate-200 hover:bg-slate-800";
    const dividerClass = isLight ? "bg-slate-200" : "bg-slate-700";
    
    return html`
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-50">
            <div className=${"flex items-center rounded-xl border shadow-lg backdrop-blur-sm overflow-hidden " + containerClass}>
                <button
                    onClick=${() => separateOutputs && onToggle()}
                    className=${"flex items-center gap-2 px-4 py-2.5 transition-all duration-300 " + (!separateOutputs ? activeClass : inactiveClass)}
                >
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4">
                        <rect x="3" y="3" width="18" height="18" rx="2"/>
                        <line x1="3" y1="12" x2="21" y2="12"/>
                    </svg>
                    <span className="text-sm font-medium">Combined</span>
                </button>
                <div className=${"w-px h-8 " + dividerClass}></div>
                <button
                    onClick=${() => !separateOutputs && onToggle()}
                    className=${"flex items-center gap-2 px-4 py-2.5 transition-all duration-300 " + (separateOutputs ? activeClass : inactiveClass)}
                >
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4">
                        <circle cx="6" cy="12" r="3"/>
                        <circle cx="18" cy="6" r="3"/>
                        <circle cx="18" cy="18" r="3"/>
                        <line x1="8.5" y1="10.5" x2="15.5" y2="7"/>
                        <line x1="8.5" y1="13.5" x2="15.5" y2="17"/>
                    </svg>
                    <span className="text-sm font-medium">Separate</span>
                </button>
            </div>
        </div>
    `;
};
'''

# Output display variation A: Clean List with Arrows
OUTPUT_STYLE_A = '''
// Variation A: Clean centered list with arrows and type hints
const OutputsSection = ({ outputs, isLight }) => {
    if (!outputs || outputs.length === 0) return null;
    
    const bgClass = isLight ? "bg-slate-50/80" : "bg-slate-900/50";
    const textClass = isLight ? "text-slate-600" : "text-slate-400";
    const arrowClass = isLight ? "text-emerald-500" : "text-emerald-400";
    const typeClass = isLight ? "text-slate-400" : "text-slate-500";
    
    return html`
        <div className=${"px-4 py-3 border-t transition-all duration-500 " + bgClass + " " + (isLight ? "border-slate-100" : "border-slate-800/50")}>
            <div className="flex flex-col items-center gap-1.5">
                ${outputs.map(out => html`
                    <div key=${out.name} className=${"flex items-center gap-2 text-xs " + textClass}>
                        <span className=${arrowClass}>→</span>
                        <span className="font-mono font-medium">${out.name}</span>
                        ${out.type ? html`<span className=${"font-mono " + typeClass}>: ${out.type}</span>` : null}
                    </div>
                `)}
            </div>
        </div>
    `;
};
'''

# Output display variation B: Pill Badges
OUTPUT_STYLE_B = '''
// Variation B: Pill badges centered with type hints
const OutputsSection = ({ outputs, isLight }) => {
    if (!outputs || outputs.length === 0) return null;
    
    const bgClass = isLight ? "bg-slate-50/80" : "bg-slate-900/50";
    const pillBg = isLight ? "bg-emerald-100 border-emerald-200" : "bg-emerald-500/20 border-emerald-500/30";
    const nameClass = isLight ? "text-emerald-700" : "text-emerald-300";
    const typeClass = isLight ? "text-emerald-600/70" : "text-emerald-400/70";
    
    return html`
        <div className=${"px-3 py-3 border-t transition-all duration-500 " + bgClass + " " + (isLight ? "border-slate-100" : "border-slate-800/50")}>
            <div className="flex flex-wrap justify-center gap-2">
                ${outputs.map(out => html`
                    <div key=${out.name} className=${"px-2.5 py-1 rounded-full border text-xs font-mono " + pillBg}>
                        <span className=${"font-semibold " + nameClass}>${out.name}</span>
                        ${out.type ? html`<span className=${typeClass}> : ${out.type}</span>` : null}
                    </div>
                `)}
            </div>
        </div>
    `;
};
'''

# Output display variation C: Table/Grid style
OUTPUT_STYLE_C = '''
// Variation C: Two-column table layout centered
const OutputsSection = ({ outputs, isLight }) => {
    if (!outputs || outputs.length === 0) return null;
    
    const bgClass = isLight ? "bg-slate-50/80" : "bg-slate-900/50";
    const headerClass = isLight ? "text-slate-500" : "text-slate-500";
    const nameClass = isLight ? "text-slate-700" : "text-slate-300";
    const typeClass = isLight ? "text-indigo-600" : "text-indigo-400";
    const rowBorder = isLight ? "border-slate-200/50" : "border-slate-700/50";
    
    return html`
        <div className=${"px-3 py-2 border-t transition-all duration-500 " + bgClass + " " + (isLight ? "border-slate-100" : "border-slate-800/50")}>
            <div className="flex justify-center">
                <table className="text-xs">
                    <thead>
                        <tr>
                            <td className=${"pr-4 pb-1 font-semibold uppercase tracking-wider text-[10px] " + headerClass}>Output</td>
                            <td className=${"pb-1 font-semibold uppercase tracking-wider text-[10px] " + headerClass}>Type</td>
                        </tr>
                    </thead>
                    <tbody>
                        ${outputs.map((out, i) => html`
                            <tr key=${out.name} className=${i < outputs.length - 1 ? "border-b " + rowBorder : ""}>
                                <td className=${"pr-4 py-1 font-mono font-medium " + nameClass}>${out.name}</td>
                                <td className=${"py-1 font-mono " + typeClass}>${out.type || "Any"}</td>
                            </tr>
                        `)}
                    </tbody>
                </table>
            </div>
        </div>
    `;
};
'''

# Output display variation D: Stacked cards
OUTPUT_STYLE_D = '''
// Variation D: Mini stacked cards centered
const OutputsSection = ({ outputs, isLight }) => {
    if (!outputs || outputs.length === 0) return null;
    
    const bgClass = isLight ? "bg-slate-50/80" : "bg-slate-900/50";
    const cardBg = isLight ? "bg-white border-slate-200 shadow-sm" : "bg-slate-800/80 border-slate-700/50";
    const nameClass = isLight ? "text-slate-800" : "text-slate-200";
    const typeClass = isLight ? "text-slate-500" : "text-slate-400";
    const iconClass = isLight ? "text-emerald-500" : "text-emerald-400";
    
    return html`
        <div className=${"px-3 py-3 border-t transition-all duration-500 " + bgClass + " " + (isLight ? "border-slate-100" : "border-slate-800/50")}>
            <div className="flex flex-col items-center gap-1.5">
                ${outputs.map(out => html`
                    <div key=${out.name} className=${"w-full max-w-[180px] px-3 py-1.5 rounded-lg border flex items-center gap-2 " + cardBg}>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className=${"w-3 h-3 shrink-0 " + iconClass}>
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            <polyline points="14 2 14 8 20 8"></polyline>
                        </svg>
                        <div className="flex-1 min-w-0">
                            <div className=${"text-xs font-mono font-medium truncate " + nameClass}>${out.name}</div>
                            ${out.type ? html`<div className=${"text-[10px] font-mono truncate " + typeClass}>${out.type}</div>` : null}
                        </div>
                    </div>
                `)}
            </div>
        </div>
    `;
};
'''

# Output display variation E: Horizontal compact
OUTPUT_STYLE_E = '''
// Variation E: Horizontal inline centered with separator dots
const OutputsSection = ({ outputs, isLight }) => {
    if (!outputs || outputs.length === 0) return null;
    
    const bgClass = isLight ? "bg-gradient-to-b from-slate-50 to-slate-100/50" : "bg-gradient-to-b from-slate-900/50 to-slate-950/30";
    const nameClass = isLight ? "text-slate-700" : "text-slate-300";
    const typeClass = isLight ? "text-indigo-500" : "text-indigo-400";
    const dotClass = isLight ? "bg-slate-300" : "bg-slate-600";
    const labelClass = isLight ? "text-slate-400" : "text-slate-500";
    
    return html`
        <div className=${"px-3 py-2.5 border-t transition-all duration-500 " + bgClass + " " + (isLight ? "border-slate-100" : "border-slate-800/50")}>
            <div className="flex flex-col items-center gap-1">
                <div className=${"text-[9px] font-bold uppercase tracking-widest " + labelClass}>Returns</div>
                <div className="flex flex-wrap items-center justify-center gap-x-1 gap-y-1">
                    ${outputs.map((out, i) => html`
                        <${React.Fragment} key=${out.name}>
                            ${i > 0 ? html`<span className=${"w-1 h-1 rounded-full mx-1 " + dotClass}></span>` : null}
                            <span className="text-xs">
                                <span className=${"font-mono font-semibold " + nameClass}>${out.name}</span>
                                ${out.type ? html`<span className=${"font-mono " + typeClass}> : ${out.type}</span>` : null}
                            </span>
                        <//>
                    `)}
                </div>
            </div>
        </div>
    `;
};
'''

# Common app code template
APP_CODE = '''
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

    {toggle_component}

    // Icons
    const Icons = {
        Function: () => html`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect><line x1="7" y1="2" x2="7" y2="22"></line><line x1="17" y1="2" x2="17" y2="22"></line><line x1="2" y1="12" x2="22" y2="12"></line></svg>`,
        Data: () => html`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-3 h-3"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>`,
    };

    {output_style}

    // Custom Node Component
    const CustomNode = ({ data, id }) => {
        const theme = data.theme;
        const isLight = theme === 'light';
        const isOutput = data.sourceId != null;
        const showAsOutput = data.separateOutputs && isOutput;
        
        // DATA nodes
        if (data.nodeType === 'DATA') {
            const baseClass = "px-3 py-1.5 w-full relative rounded-full border shadow-sm flex items-center justify-center gap-2 transition-all duration-300";
            const themeClass = isLight 
                ? "bg-white border-slate-200 text-slate-700" 
                : "bg-slate-900 border-slate-700 text-slate-300";
            const outputHighlight = showAsOutput ? "ring-2 ring-emerald-500/30 scale-105" : "";
            const iconColor = isLight ? "text-slate-400" : "text-slate-500";
            const typeClass = isLight ? "text-slate-400" : "text-slate-500";
            
            return html`
                <div className=${baseClass + " " + themeClass + " " + outputHighlight}>
                    <span className=${iconColor}><${Icons.Data} /></span>
                    <span className="text-xs font-mono font-medium truncate">${data.label}</span>
                    ${data.typeHint ? html`<span className=${"text-[10px] font-mono " + typeClass}>: ${data.typeHint}</span>` : null}
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
                <div className=${"px-3 py-2.5 flex items-center gap-3 " + (showCombined ? "border-b " + borderClass : "")}>
                    <div className=${"p-1.5 rounded-md shrink-0 " + iconBgClass}>
                        <${Icons.Function} />
                    </div>
                    <div className="min-w-0 flex-1">
                        <div className=${"text-[9px] font-bold tracking-wider uppercase mb-0.5 " + labelTypeClass}>FUNCTION</div>
                        <div className=${"text-sm font-semibold truncate " + labelClass}>${data.label}</div>
                    </div>
                </div>
                
                ${showCombined ? html`<${OutputsSection} outputs=${outputs} isLight=${isLight} />` : null}
                
                <${Handle} type="target" position=${Position.Top} className="!w-2 !h-2 !opacity-0" />
                <${Handle} type="source" position=${Position.Bottom} className="!w-2 !h-2 !opacity-0" />
            </div>
        `;
    };

    const nodeTypes = { custom: CustomNode };

    // Layout hook
    const useLayout = (nodes, edges) => {
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
                    let width = 200;
                    let height = 72;
                    if (n.data.nodeType === 'DATA') {
                        width = 180;
                        height = 36;
                    } else if (!n.data.separateOutputs && n.data.outputs?.length) {
                        // Taller for combined outputs
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
        }, [nodes, edges]);

        return { layoutedNodes, layoutedEdges };
    };

    const initialData = {graph_json};

    const App = () => {
        const [separateOutputs, setSeparateOutputs] = useState(initialData.meta?.separate_outputs ?? false);
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
                        data: { ...n.data, theme, separateOutputs: true },
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
        }, [separateOutputs, theme]);

        const { layoutedNodes, layoutedEdges } = useLayout(nodes, edges);

        useEffect(() => {
            if (layoutedNodes.length) {
                setTimeout(() => fitView({ padding: 0.2, duration: 400 }), 100);
            }
        }, [layoutedNodes, fitView]);

        const handleToggle = useCallback(() => {
            setSeparateOutputs(prev => !prev);
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
                <${SeparateOutputsToggle} separateOutputs=${separateOutputs} onToggle=${handleToggle} theme=${theme} />
            </div>
        `;
    };

    const root = ReactDOM.createRoot(document.getElementById('root'));
    root.render(html`<${ReactFlowProvider}><${App} /><//>`);
  </script>
</body>
</html>
'''


def generate_html(design_name: str, output_style: str, graph_data: dict) -> str:
    """Generate complete HTML file for a design."""
    head = HTML_HEAD.format(design_name=design_name)
    body = (APP_CODE
        .replace("{toggle_component}", TOGGLE_COMPONENT)
        .replace("{output_style}", output_style)
        .replace("{graph_json}", json.dumps(graph_data)))
    return head + body


def main():
    output_dir = Path(__file__).parent.parent / "outputs" / "output_display_designs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    designs = [
        ("A_arrow_list", OUTPUT_STYLE_A, "Clean centered list with arrows"),
        ("B_pill_badges", OUTPUT_STYLE_B, "Pill/badge style for each output"),
        ("C_table_grid", OUTPUT_STYLE_C, "Two-column table layout"),
        ("D_stacked_cards", OUTPUT_STYLE_D, "Mini stacked cards"),
        ("E_horizontal_inline", OUTPUT_STYLE_E, "Horizontal inline with dots"),
    ]
    
    print(f"Generating {len(designs)} output display variations...")
    
    for name, style_code, description in designs:
        html = generate_html(name, style_code, SAMPLE_GRAPH)
        filepath = output_dir / f"output_{name}.html"
        filepath.write_text(html)
        print(f"  ✓ {filepath.name}: {description}")
    
    print(f"\nAll designs saved to: {output_dir}")
    print("\nOpen each HTML file in a browser to compare output display designs.")


if __name__ == "__main__":
    main()

