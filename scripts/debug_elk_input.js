const utils = require('../assets/viz/state_utils.js');
const fs = require('fs');

const htmlContent = fs.readFileSync('../outputs/test_viz_comprehensive.html', 'utf-8');
const match = htmlContent.match(/<script id="graph-data" type="application\/json">([\s\S]*?)<\/script>/);
const data = JSON.parse(match[1]);

const expansionState = new Map();
data.nodes.forEach(n => {
  if (n.data.nodeType === 'PIPELINE') {
    expansionState.set(n.id, n.data.isExpanded);
  }
});

const stateResult = utils.applyState(data.nodes, data.edges, {
  expansionState,
  separateOutputs: false,
  showTypes: true,
  theme: 'dark'
});

const nodesWithVis = utils.applyVisibility(stateResult.nodes, expansionState);
const compressed = utils.compressEdges(nodesWithVis, stateResult.edges);

// Simulating what buildElkHierarchy does
const visibleNodes = nodesWithVis.filter(n => !n.hidden);
const visibleNodeIds = new Set(visibleNodes.map(n => n.id));
const visibleEdges = compressed.filter(e => visibleNodeIds.has(e.source) && visibleNodeIds.has(e.target));

console.log("=== Visible nodes (not hidden) ===");
visibleNodes.forEach(n => {
  const parent = n.parentNode || "root";
  const nodeType = n.data?.nodeType;
  console.log(`  ${n.id} (${nodeType}, parent: ${parent})`);
});

console.log("\n=== Visible edges (both endpoints in visibleNodeIds) ===");
visibleEdges.forEach(e => console.log(`  ${e.source} -> ${e.target}`));

console.log("\n=== Check edge: rag_pipeline__generate_answer -> answer ===");
const edge1 = compressed.find(e => e.source === 'rag_pipeline__generate_answer' && e.target === 'answer');
if (edge1) {
  const sourceVis = visibleNodeIds.has('rag_pipeline__generate_answer');
  const targetVis = visibleNodeIds.has('answer');
  console.log(`  Edge exists: ${!!edge1}`);
  console.log(`  Source visible: ${sourceVis}`);
  console.log(`  Target visible: ${targetVis}`);
  console.log(`  Would be included: ${sourceVis && targetVis}`);
}

console.log("\n=== Check edge: answer -> evaluate ===");
const edge2 = compressed.find(e => e.source === 'answer' && e.target === 'evaluate');
if (edge2) {
  const sourceVis = visibleNodeIds.has('answer');
  const targetVis = visibleNodeIds.has('evaluate');
  console.log(`  Edge exists: ${!!edge2}`);
  console.log(`  Source visible: ${sourceVis}`);
  console.log(`  Target visible: ${targetVis}`);
  console.log(`  Would be included: ${sourceVis && targetVis}`);
}
