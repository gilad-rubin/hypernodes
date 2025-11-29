const utils = require('../assets/viz/state_utils.js');
const fs = require('fs');

const htmlContent = fs.readFileSync('outputs/test_viz_comprehensive.html', 'utf-8');
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

// Find answer node at root level
const answerNode = nodesWithVis.find(n => n.id === 'answer');
console.log("=== Root answer node ===");
if (answerNode) {
  console.log("  id:", answerNode.id);
  console.log("  parentNode:", answerNode.parentNode || "root");
  console.log("  hidden:", answerNode.hidden);
  console.log("  data.nodeType:", answerNode.data?.nodeType);
  console.log("  data.sourceId:", answerNode.data?.sourceId);
} else {
  console.log("  NOT FOUND in nodes!");
  console.log("  Checking for nodes containing 'answer':");
  nodesWithVis.filter(n => n.id.includes('answer')).forEach(n => {
    console.log(`    - ${n.id} (parent: ${n.parentNode || 'root'}, hidden: ${n.hidden})`);
  });
}


