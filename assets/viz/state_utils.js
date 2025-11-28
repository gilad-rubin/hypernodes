/**
 * Shared state helpers for the interactive viz.
 * Exposed as window.HyperNodesVizState in the browser and CommonJS exports for tests.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  if (root) {
    root.HyperNodesVizState = api;
  }
})(typeof window !== "undefined" ? window : globalThis, function () {
  const toMap = (expansionState) => {
    if (expansionState instanceof Map) return expansionState;
    if (expansionState && typeof expansionState === "object") {
      return new Map(Object.entries(expansionState));
    }
    return new Map();
  };

  const applyState = (baseNodes, baseEdges, options) => {
    const { expansionState, separateOutputs, showTypes, theme } = options;
    const expMap = toMap(expansionState);

    const applyMeta = (nodeList) =>
      nodeList.map((n) => {
        const isPipeline = n.data?.nodeType === "PIPELINE";
        const expanded = isPipeline ? Boolean(expMap.get(n.id)) : undefined;
        return {
          ...n,
          type: isPipeline && expanded ? "pipelineGroup" : n.type,
          style: isPipeline && !expanded ? undefined : n.style,
          data: {
            ...n.data,
            theme,
            showTypes,
            isExpanded: expanded,
          },
        };
      });

    if (separateOutputs) {
      return {
        nodes: applyMeta(baseNodes).map((n) => ({
          ...n,
          data: { ...n.data, separateOutputs: true },
        })),
        edges: baseEdges,
      };
    }

    const outputNodes = new Set(
      baseNodes.filter((n) => n.data?.sourceId).map((n) => n.id)
    );
    const functionOutputs = {};
    baseNodes.forEach((n) => {
      if (n.data?.sourceId) {
        if (!functionOutputs[n.data.sourceId]) functionOutputs[n.data.sourceId] = [];
        functionOutputs[n.data.sourceId].push({
          name: n.data.label,
          type: n.data.typeHint,
        });
      }
    });

    const nodes = applyMeta(baseNodes)
      .filter((n) => !outputNodes.has(n.id))
      .map((n) => ({
        ...n,
        data: {
          ...n.data,
          separateOutputs: false,
          outputs: functionOutputs[n.id] || [],
        },
      }));

    const edges = baseEdges
      .filter((e) => !outputNodes.has(e.target))
      .map((e) => {
        if (outputNodes.has(e.source)) {
          const outputNode = baseNodes.find((n) => n.id === e.source);
          if (outputNode?.data?.sourceId) {
            return {
              ...e,
              id: `e_${outputNode.data.sourceId}_${e.target}`,
              source: outputNode.data.sourceId,
            };
          }
        }
        return e;
      });

    return { nodes, edges };
  };

  const applyVisibility = (nodes, expansionState) => {
    const expMap = toMap(expansionState);
    const parentMap = new Map();
    nodes.forEach((n) => {
      if (n.parentNode) parentMap.set(n.id, n.parentNode);
    });

    const isHidden = (nodeId) => {
      let curr = nodeId;
      while (curr) {
        const parent = parentMap.get(curr);
        if (!parent) return false;
        if (expMap.get(parent) === false) return true;
        curr = parent;
      }
      return false;
    };

    return nodes.map((n) => ({
      ...n,
      hidden: isHidden(n.id),
    }));
  };

  const compressEdges = (nodes, edges) => {
    const nodeMap = new Map(nodes.map((n) => [n.id, n]));
    const parentMap = new Map();
    const expansionMap = new Map();
    nodes.forEach((n) => {
      if (n.parentNode) parentMap.set(n.id, n.parentNode);
      if (n.data?.nodeType === "PIPELINE") expansionMap.set(n.id, !!n.data.isExpanded);
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

    edges.forEach((edge) => {
      const sourceNode = nodeMap.get(edge.source);
      const targetNode = nodeMap.get(edge.target);
      if (!sourceNode || !targetNode) return;

      const sourceExpandedPipeline =
        sourceNode.data?.nodeType === "PIPELINE" && sourceNode.data.isExpanded;
      const targetExpandedPipeline =
        targetNode.data?.nodeType === "PIPELINE" && targetNode.data.isExpanded;
      if (sourceExpandedPipeline || targetExpandedPipeline) return;

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
  };

  return { applyState, applyVisibility, compressEdges };
});
