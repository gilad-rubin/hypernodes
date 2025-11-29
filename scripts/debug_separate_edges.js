const utils = require('../assets/viz/state_utils.js');
const fs = require('fs');

const html = fs.readFileSync('../outputs/test_separate_expanded.html', 'utf-8');
const match = html.match(/<script id="graph-data"[^>]*>([\s\S]*?)<\/script>/);
const data = JSON.parse(match[1]);

// Set rag_pipeline as expanded
const expansionState = new Map();
data.nodes.forEach(n => {
    if (n.data.nodeType === 'PIPELINE') {
        expansionState.set(n.id, true);  // All expanded
    }
});

console.log("=== Separate outputs mode, expanded pipelines ===");
const result = utils.applyState(data.nodes, data.edges, {
    expansionState,
    separateOutputs: true,
    showTypes: true,
    theme: 'dark'
});

console.log("\nEdges after applyState:");
result.edges.filter(e => 
    e.source.includes('answer') || e.target.includes('answer') ||
    e.source.includes('generate_answer') || e.target.includes('evaluate')
).forEach(e => console.log(`  ${e.source} -> ${e.target}`));

console.log("\nNodes involving 'answer':");
result.nodes.filter(n => n.id.includes('answer')).forEach(n => 
    console.log(`  ${n.id} (hidden: ${n.hidden})`)
);
