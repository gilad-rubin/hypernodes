"""Generate HTML files with different toggle designs for separate_outputs feature."""

import json
from pathlib import Path

# Sample graph data with outputs that can be combined or separated
SAMPLE_GRAPH = {
    "nodes": [
        {"id": "input_x", "position": {"x": 0, "y": 0}, "type": "custom", "data": {"label": "x", "nodeType": "DATA", "theme": "dark"}},
        {"id": "input_y", "position": {"x": 0, "y": 0}, "type": "custom", "data": {"label": "y", "nodeType": "DATA", "theme": "dark"}},
        {"id": "add", "position": {"x": 0, "y": 0}, "type": "custom", "data": {"label": "add", "nodeType": "FUNCTION", "functionName": "add", "theme": "dark"}},
        {"id": "output_sum", "position": {"x": 0, "y": 0}, "type": "custom", "data": {"label": "sum", "nodeType": "DATA", "theme": "dark", "sourceId": "add"}},
        {"id": "multiply", "position": {"x": 0, "y": 0}, "type": "custom", "data": {"label": "multiply", "nodeType": "FUNCTION", "functionName": "multiply", "theme": "dark"}},
        {"id": "output_product", "position": {"x": 0, "y": 0}, "type": "custom", "data": {"label": "product", "nodeType": "DATA", "theme": "dark", "sourceId": "multiply"}},
        {"id": "combine", "position": {"x": 0, "y": 0}, "type": "custom", "data": {"label": "combine", "nodeType": "FUNCTION", "functionName": "combine", "theme": "dark"}},
        {"id": "output_result", "position": {"x": 0, "y": 0}, "type": "custom", "data": {"label": "result", "nodeType": "DATA", "theme": "dark", "sourceId": "combine"}},
    ],
    "edges": [
        {"id": "e1", "source": "input_x", "target": "add"},
        {"id": "e2", "source": "input_y", "target": "add"},
        {"id": "e3", "source": "add", "target": "output_sum"},
        {"id": "e4", "source": "output_sum", "target": "multiply"},
        {"id": "e5", "source": "input_y", "target": "multiply"},
        {"id": "e6", "source": "multiply", "target": "output_product"},
        {"id": "e7", "source": "output_product", "target": "combine"},
        {"id": "e8", "source": "output_sum", "target": "combine"},
        {"id": "e9", "source": "combine", "target": "output_result"},
    ],
    "meta": {"theme_preference": "auto", "initial_depth": 1, "separate_outputs": False}
}

# Common HTML head
HTML_HEAD = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Separate Outputs Toggle - {design_name}</title>
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

# Design A: Bottom-left pill toggle with icon
DESIGN_A_TOGGLE = '''
const SeparateOutputsToggle = ({ separateOutputs, onToggle, theme }) => {
    const isLight = theme === 'light';
    
    const baseClass = "flex items-center gap-3 px-4 py-2.5 rounded-full border shadow-lg backdrop-blur-sm transition-all duration-300 hover:scale-105 active:scale-95";
    const themeClass = isLight 
        ? "bg-white/95 border-slate-200 text-slate-700 hover:bg-slate-50" 
        : "bg-slate-900/95 border-slate-700 text-slate-200 hover:bg-slate-800";
    const iconRotate = separateOutputs ? "rotate-0" : "rotate-180";
    const iconOpacity = separateOutputs ? "opacity-100" : "opacity-50";
    const toggleBg = separateOutputs 
        ? (isLight ? "bg-indigo-500" : "bg-indigo-600") 
        : (isLight ? "bg-slate-300" : "bg-slate-600");
    const thumbPos = separateOutputs ? "left-5" : "left-0.5";
    
    return html`
        <div className="absolute bottom-4 left-4 z-50">
            <button onClick=${onToggle} className=${baseClass + " " + themeClass}>
                <div className=${"relative w-5 h-5 transition-transform duration-300 " + iconRotate}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" 
                         className=${"w-5 h-5 transition-all duration-300 " + iconOpacity}>
                        <circle cx="6" cy="12" r="3"/>
                        <circle cx="18" cy="6" r="3"/>
                        <circle cx="18" cy="18" r="3"/>
                        <line x1="8.5" y1="10.5" x2="15.5" y2="7"/>
                        <line x1="8.5" y1="13.5" x2="15.5" y2="17"/>
                    </svg>
                </div>
                <span className="text-sm font-medium whitespace-nowrap">
                    ${separateOutputs ? "Separate Outputs" : "Combined Outputs"}
                </span>
                <div className=${"w-10 h-5 rounded-full relative transition-all duration-300 " + toggleBg}>
                    <div className=${"absolute top-0.5 w-4 h-4 rounded-full shadow-md transition-all duration-300 bg-white " + thumbPos}></div>
                </div>
            </button>
        </div>
    `;
};
'''

# Design B: Top-center segmented control
DESIGN_B_TOGGLE = '''
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

# Design C: Icon-only floating button with tooltip
DESIGN_C_TOGGLE = '''
const SeparateOutputsToggle = ({ separateOutputs, onToggle, theme }) => {
    const isLight = theme === 'light';
    const [showTooltip, setShowTooltip] = React.useState(false);
    
    const buttonClass = separateOutputs 
        ? (isLight ? "bg-indigo-500 border-indigo-400 text-white" : "bg-indigo-600 border-indigo-500 text-white")
        : (isLight ? "bg-white/95 border-slate-200 text-slate-600" : "bg-slate-900/95 border-slate-700 text-slate-400");
    const tooltipClass = isLight 
        ? "bg-white border-slate-200 text-slate-700" 
        : "bg-slate-800 border-slate-700 text-slate-200";
    const arrowClass = isLight ? "border-r-white" : "border-r-slate-800";
    
    return html`
        <div className="absolute bottom-4 left-4 z-50" 
             onMouseEnter=${() => setShowTooltip(true)} 
             onMouseLeave=${() => setShowTooltip(false)}>
            <button
                onClick=${onToggle}
                className=${"relative p-3 rounded-xl border shadow-lg backdrop-blur-sm transition-all duration-300 hover:scale-110 active:scale-95 " + buttonClass}
            >
                <div className="relative w-5 h-5">
                    ${separateOutputs ? html`
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
                            <circle cx="6" cy="12" r="3"/>
                            <circle cx="18" cy="6" r="3"/>
                            <circle cx="18" cy="18" r="3"/>
                            <line x1="8.5" y1="10.5" x2="15.5" y2="7"/>
                            <line x1="8.5" y1="13.5" x2="15.5" y2="17"/>
                        </svg>
                    ` : html`
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
                            <rect x="3" y="3" width="18" height="18" rx="2"/>
                            <line x1="3" y1="12" x2="21" y2="12"/>
                        </svg>
                    `}
                </div>
                ${separateOutputs ? html`
                    <div className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-green-400 shadow-lg animate-pulse"></div>
                ` : null}
            </button>
            ${showTooltip ? html`
                <div className=${"absolute left-full ml-3 top-1/2 transform -translate-y-1/2 px-3 py-1.5 rounded-lg whitespace-nowrap text-sm font-medium shadow-lg border transition-all duration-200 " + tooltipClass}>
                    ${separateOutputs ? "Outputs: Separate" : "Outputs: Combined"}
                    <div className=${"absolute right-full mr-0 top-1/2 transform -translate-y-1/2 border-8 border-transparent " + arrowClass}></div>
                </div>
            ` : null}
        </div>
    `;
};
'''

# Design D: Animated slide-in panel
DESIGN_D_TOGGLE = '''
const SeparateOutputsToggle = ({ separateOutputs, onToggle, theme }) => {
    const isLight = theme === 'light';
    const [isExpanded, setIsExpanded] = React.useState(false);
    
    const containerClass = isLight ? "bg-white/95 border-slate-200" : "bg-slate-900/95 border-slate-700";
    const iconClass = separateOutputs 
        ? (isLight ? "text-indigo-600" : "text-indigo-400")
        : (isLight ? "text-slate-500" : "text-slate-400");
    const labelClass = isLight ? "text-slate-700" : "text-slate-200";
    const toggleBg = separateOutputs 
        ? (isLight ? "bg-indigo-500" : "bg-indigo-600") 
        : (isLight ? "bg-slate-300" : "bg-slate-600");
    const thumbState = separateOutputs 
        ? "left-8 bg-white text-indigo-600" 
        : "left-1 bg-white text-slate-500";
    const containerWidth = isExpanded ? "w-64" : "w-12";
    const contentOpacity = isExpanded ? "opacity-100 w-auto" : "opacity-0 w-0";
    
    return html`
        <div className="absolute bottom-4 left-4 z-50">
            <div 
                className=${"flex items-center rounded-2xl border shadow-lg backdrop-blur-sm overflow-hidden transition-all duration-500 ease-out " + containerClass + " " + containerWidth}
                onMouseEnter=${() => setIsExpanded(true)}
                onMouseLeave=${() => setIsExpanded(false)}
            >
                <button
                    onClick=${isExpanded ? undefined : () => setIsExpanded(true)}
                    className=${"p-3 shrink-0 transition-colors duration-200 " + iconClass}
                >
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-5 h-5">
                        ${separateOutputs ? html`
                            <circle cx="6" cy="12" r="3"/>
                            <circle cx="18" cy="6" r="3"/>
                            <circle cx="18" cy="18" r="3"/>
                            <line x1="8.5" y1="10.5" x2="15.5" y2="7"/>
                            <line x1="8.5" y1="13.5" x2="15.5" y2="17"/>
                        ` : html`
                            <rect x="3" y="3" width="18" height="18" rx="2"/>
                            <line x1="3" y1="12" x2="21" y2="12"/>
                        `}
                    </svg>
                </button>
                <div className=${"flex items-center gap-3 pr-4 overflow-hidden transition-all duration-500 " + contentOpacity}>
                    <span className=${"text-sm font-medium whitespace-nowrap " + labelClass}>
                        Output mode
                    </span>
                    <button
                        onClick=${onToggle}
                        className=${"relative w-14 h-7 rounded-full transition-all duration-300 " + toggleBg}
                    >
                        <div className=${"absolute top-1 w-5 h-5 rounded-full shadow-md transition-all duration-300 flex items-center justify-center text-xs font-bold " + thumbState}>
                            ${separateOutputs ? "S" : "C"}
                        </div>
                    </button>
                </div>
            </div>
        </div>
    `;
};
'''

# Design E: Minimal corner chip
DESIGN_E_TOGGLE = '''
const SeparateOutputsToggle = ({ separateOutputs, onToggle, theme }) => {
    const isLight = theme === 'light';
    
    const chipClass = separateOutputs 
        ? (isLight 
            ? "bg-indigo-50 border-indigo-200 text-indigo-700 hover:bg-indigo-100" 
            : "bg-indigo-500/20 border-indigo-500/50 text-indigo-300 hover:bg-indigo-500/30")
        : (isLight 
            ? "bg-white border-slate-200 text-slate-600 hover:bg-slate-50" 
            : "bg-slate-900/90 border-slate-700 text-slate-400 hover:bg-slate-800");
    const dotColor = separateOutputs 
        ? (isLight ? "bg-indigo-500" : "bg-indigo-400")
        : (isLight ? "bg-slate-400" : "bg-slate-500");
    
    return html`
        <div className="absolute top-4 right-4 z-50">
            <button
                onClick=${onToggle}
                className=${"group flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold uppercase tracking-wider border shadow-md backdrop-blur-sm transition-all duration-300 hover:scale-105 active:scale-95 " + chipClass}
            >
                <div className=${"w-1.5 h-1.5 rounded-full transition-all duration-300 " + dotColor}></div>
                ${separateOutputs ? "Outputs: Separate" : "Outputs: Combined"}
            </button>
        </div>
    `;
};
'''

# Common app code template - uses {toggle_component} and {graph_json} as placeholders
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

    // Custom Node Component with separate_outputs support
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
            
            return html`
                <div className=${baseClass + " " + themeClass + " " + outputHighlight}>
                    <span className=${iconColor}><${Icons.Data} /></span>
                    <span className="text-xs font-mono font-medium truncate">${data.label}</span>
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
        const outputsBgClass = isLight ? "bg-slate-50" : "bg-slate-900/50";
        const outputTextClass = isLight ? "text-slate-600" : "text-slate-400";
        const arrowClass = isLight ? "text-emerald-500" : "text-emerald-400";
        
        return html`
            <div className=${"group relative w-full rounded-lg border shadow-lg transition-all duration-300 " + containerClass}>
                <div className=${"px-3 py-2.5 border-b flex items-center gap-3 " + borderClass}>
                    <div className=${"p-1.5 rounded-md shrink-0 " + iconBgClass}>
                        <${Icons.Function} />
                    </div>
                    <div className="min-w-0 flex-1">
                        <div className=${"text-xs font-bold tracking-wider uppercase mb-0.5 " + labelTypeClass}>FUNCTION</div>
                        <div className=${"text-sm font-semibold truncate " + labelClass}>${data.label}</div>
                    </div>
                </div>
                
                ${showCombined ? html`
                    <div className=${"px-3 py-2 space-y-1 transition-all duration-500 " + outputsBgClass}>
                        ${outputs.map(out => html`
                            <div key=${out} className=${"flex items-center gap-2 text-xs " + outputTextClass}>
                                <span className=${arrowClass}>→</span>
                                <span className="font-mono">${out}</span>
                            </div>
                        `)}
                    </div>
                ` : null}
                
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
                children: nodes.filter(n => !n.hidden).map(n => ({
                    id: n.id,
                    width: n.data.nodeType === 'DATA' ? 140 : 200,
                    height: n.data.nodeType === 'DATA' ? 36 : (n.data.separateOutputs || !n.data.outputs?.length ? 72 : 100),
                })),
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
                        functionOutputs[n.data.sourceId].push(n.data.label);
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
                        // Skip edges that go TO output nodes
                        if (outputNodes.has(e.target)) return false;
                        return true;
                    }).map(e => {
                        if (outputNodes.has(e.source)) {
                            // Find the function that produces this output
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


def generate_html(design_name: str, toggle_component: str, graph_data: dict) -> str:
    """Generate complete HTML file for a design."""
    head = HTML_HEAD.format(design_name=design_name)
    # Use string replacement instead of .format() to avoid conflicts with JS curly braces
    body = APP_CODE.replace("{toggle_component}", toggle_component).replace("{graph_json}", json.dumps(graph_data))
    return head + body


def main():
    output_dir = Path(__file__).parent.parent / "outputs" / "separate_outputs_designs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    designs = [
        ("A_pill_toggle", DESIGN_A_TOGGLE, "Bottom-left pill with icon + label + toggle switch"),
        ("B_segmented_control", DESIGN_B_TOGGLE, "Top-center segmented control with icons"),
        ("C_icon_button", DESIGN_C_TOGGLE, "Floating icon button with tooltip"),
        ("D_slide_panel", DESIGN_D_TOGGLE, "Slide-in panel on hover"),
        ("E_minimal_chip", DESIGN_E_TOGGLE, "Minimal corner chip/badge"),
    ]
    
    print(f"Generating {len(designs)} design variations...")
    
    for name, toggle_code, description in designs:
        html = generate_html(name, toggle_code, SAMPLE_GRAPH)
        filepath = output_dir / f"design_{name}.html"
        filepath.write_text(html)
        print(f"  ✓ {filepath.name}: {description}")
    
    print(f"\nAll designs saved to: {output_dir}")
    print("\nOpen each HTML file in a browser to compare designs.")
    print("Click the toggle to see the animation between Combined and Separate modes.")


if __name__ == "__main__":
    main()
