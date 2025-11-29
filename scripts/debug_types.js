const utils = require('../assets/viz/state_utils.js');
const fs = require('fs');

const html = fs.readFileSync('../outputs/test_collapsed.html', 'utf-8');
const match = html.match(/<script id="graph-data"[^>]*>([\s\S]*?)<\/script>/);
const data = JSON.parse(match[1]);

// Find input nodes
const inputs = data.nodes.filter(n => n.data?.nodeType === 'DATA' && !n.data?.sourceId);
console.log("=== Input nodes ===");
inputs.forEach(n => {
    console.log(`  ${n.id}: typeHint="${n.data?.typeHint}", showTypes=${n.data?.showTypes}`);
});
