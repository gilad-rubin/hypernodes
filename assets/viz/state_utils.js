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

    // Build lookup maps
    const sourceNodeTypes = new Map();
    const sourceIdMap = new Map(); // nodeId -> sourceId
    baseNodes.forEach((n) => {
      sourceNodeTypes.set(n.id, n.data?.nodeType);
      if (n.data?.sourceId) sourceIdMap.set(n.id, n.data.sourceId);
    });

    // Track which pipelines are expanded/collapsed
    const expandedPipelines = new Set(
      baseNodes
        .filter((n) => n.data?.nodeType === "PIPELINE" && expMap.get(n.id))
        .map((n) => n.id)
    );
    const collapsedPipelines = new Set(
      baseNodes
        .filter((n) => n.data?.nodeType === "PIPELINE" && !expMap.get(n.id))
        .map((n) => n.id)
    );

    // Identify boundary outputs: DATA nodes with sourceId pointing to a PIPELINE
    const boundaryOutputs = new Set(
      baseNodes
        .filter((n) => {
          if (!n.data?.sourceId) return false;
          return sourceNodeTypes.get(n.data.sourceId) === "PIPELINE";
        })
        .map((n) => n.id)
    );

    // Boundary outputs of EXPANDED pipelines should be hidden
    const boundaryOutputsToHide = new Set(
      baseNodes
        .filter((n) => boundaryOutputs.has(n.id) && expandedPipelines.has(n.data?.sourceId))
        .map((n) => n.id)
    );

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

    // Build edge map to find producer chains
    const edgesByTarget = new Map();
    baseEdges.forEach((e) => {
      if (!edgesByTarget.has(e.target)) edgesByTarget.set(e.target, []);
      edgesByTarget.get(e.target).push(e);
    });

    // Find the actual producer for a node (follow sourceId chain)
    const findProducer = (nodeId) => {
      const node = baseNodes.find((n) => n.id === nodeId);
      if (!node?.data?.sourceId) return nodeId;
      const sourceType = sourceNodeTypes.get(node.data.sourceId);
      if (sourceType === "FUNCTION" || sourceType === "DUAL") {
        return node.data.sourceId;
      }
      // For pipeline sources, need to find the internal producer
      // Look at incoming edges to find who produces this node
      const incomingEdges = edgesByTarget.get(nodeId) || [];
      if (incomingEdges.length > 0) {
        return findProducer(incomingEdges[0].source);
      }
      return node.data.sourceId;
    };

    if (separateOutputs) {
      // Separate outputs mode:
      // - Hide boundary outputs for EXPANDED pipelines (internal is visible)
      // - Show boundary outputs for COLLAPSED pipelines
      const filteredNodes = applyMeta(baseNodes)
        .filter((n) => !boundaryOutputsToHide.has(n.id))
        .map((n) => ({
          ...n,
          data: { ...n.data, separateOutputs: true },
        }));

      // Remap edges FROM hidden boundary outputs to their producer
      const remappedEdges = baseEdges
        .filter((e) => !boundaryOutputsToHide.has(e.target))
        .map((e) => {
          if (boundaryOutputsToHide.has(e.source)) {
            const producer = findProducer(e.source);
            return {
              ...e,
              id: `e_${producer}_${e.target}`,
              source: producer,
            };
          }
          return e;
        });

      return { nodes: filteredNodes, edges: remappedEdges };
    }

    // Combined outputs mode:
    // - Always combine FUNCTION/DUAL outputs
    // - Combine PIPELINE outputs only when pipeline is COLLAPSED
    const outputNodes = new Set(
      baseNodes
        .filter((n) => {
          if (!n.data?.sourceId) return false;
          const sourceType = sourceNodeTypes.get(n.data.sourceId);
          // Combine function/dual outputs
          if (sourceType === "FUNCTION" || sourceType === "DUAL") return true;
          // Combine pipeline outputs only if pipeline is collapsed
          if (sourceType === "PIPELINE" && collapsedPipelines.has(n.data.sourceId)) return true;
          return false;
        })
        .map((n) => n.id)
    );

    // Also hide boundary outputs for expanded pipelines
    const nodesToHide = new Set([...outputNodes, ...boundaryOutputsToHide]);

    const functionOutputs = {};
    baseNodes.forEach((n) => {
      if (n.data?.sourceId && outputNodes.has(n.id)) {
        if (!functionOutputs[n.data.sourceId]) functionOutputs[n.data.sourceId] = [];
        functionOutputs[n.data.sourceId].push({
          name: n.data.label,
          type: n.data.typeHint,
        });
      }
    });

    const nodes = applyMeta(baseNodes)
      .filter((n) => !nodesToHide.has(n.id))
      .map((n) => ({
        ...n,
        data: {
          ...n.data,
          separateOutputs: false,
          outputs: functionOutputs[n.id] || [],
        },
      }));

    // Process edges:
    // 1. Filter edges TO hidden nodes
    // 2. Remap edges FROM output nodes to come from their producer
    // 3. Remap edges FROM hidden boundary outputs to their producer
    const edges = baseEdges
      .filter((e) => !nodesToHide.has(e.target))
      .map((e) => {
        // Remap from function output nodes
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
        // Remap from hidden boundary outputs
        if (boundaryOutputsToHide.has(e.source)) {
          const producer = findProducer(e.source);
          return {
            ...e,
            id: `e_${producer}_${e.target}`,
            source: producer,
          };
        }
        return e;
      })
      .filter((e) => {
        // Filter out any remaining edges from hidden nodes
        return !nodesToHide.has(e.source);
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

      // Get visible ancestors - handles collapsed pipeline remapping
      const sourceVis = getVisibleAncestor(edge.source);
      const targetVis = getVisibleAncestor(edge.target);

      // If both endpoints are visible as-is (no remapping needed), keep edge unchanged
      if (sourceVis === edge.source && targetVis === edge.target) {
        const edgeId = edge.id || `e_${edge.source}_${edge.target}`;
        if (!processedEdges.has(edgeId)) {
          newEdges.push(edge);
          processedEdges.add(edgeId);
        }
        return;
      }

      // If endpoints need remapping and result in different visible nodes, create compressed edge
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

  /**
   * Group input nodes that share the same targets and bound state.
   * This runs after edge compression so inputs targeting the same collapsed pipeline get grouped.
   */
  const groupInputs = (nodes, edges) => {
    // Find input nodes: DATA nodes without sourceId, not hidden
    const inputNodes = nodes.filter(
      (n) => n.data?.nodeType === "DATA" && !n.data?.sourceId && !n.hidden
    );
    if (inputNodes.length < 2) return { nodes, edges };

    // Build target map for each input
    const targetMap = new Map();
    inputNodes.forEach((n) => targetMap.set(n.id, new Set()));
    edges.forEach((e) => {
      if (targetMap.has(e.source)) {
        targetMap.get(e.source).add(e.target);
      }
    });

    // Group by (targets, isBound, parentNode)
    const groups = new Map();
    inputNodes.forEach((n) => {
      const targets = Array.from(targetMap.get(n.id) || []).sort().join(",");
      const isBound = Boolean(n.data?.isBound);
      const parent = n.parentNode || "root";
      const key = `${parent}|${targets}|${isBound}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(n);
    });

    // Only group if 2+ nodes share the same key
    const nodesToRemove = new Set();
    const newNodes = [];
    const edgeRewrites = new Map(); // oldSourceId -> newGroupId

    groups.forEach((group, key) => {
      if (group.length < 2) return;

      // Create group node
      const groupId = `group_input_${Math.abs(hashCode(key)) % 100000}`;
      const params = group.map((n) => n.data?.label || n.id);
      const paramTypes = group.map((n) => n.data?.typeHint || null);
      const firstNode = group[0];

      const groupNode = {
        id: groupId,
        position: firstNode.position || { x: 0, y: 0 },
        parentNode: firstNode.parentNode,
        extent: firstNode.extent,
        type: "custom",
        data: {
          nodeType: "INPUT_GROUP",
          label: "Inputs",
          params,
          paramTypes,
          isBound: firstNode.data?.isBound || false,
          theme: firstNode.data?.theme,
          showTypes: firstNode.data?.showTypes,
        },
        sourcePosition: "bottom",
        targetPosition: "top",
      };
      newNodes.push(groupNode);

      // Mark individual nodes for removal and track edge rewrites
      group.forEach((n) => {
        nodesToRemove.add(n.id);
        edgeRewrites.set(n.id, groupId);
      });
    });

    if (newNodes.length === 0) return { nodes, edges };

    // Filter out removed nodes and add group nodes
    const filteredNodes = nodes.filter((n) => !nodesToRemove.has(n.id));
    const finalNodes = [...filteredNodes, ...newNodes];

    // Rewrite edges from removed inputs to group
    const rewrittenEdges = [];
    const seenEdges = new Set();
    edges.forEach((e) => {
      const newSource = edgeRewrites.get(e.source) || e.source;
      const edgeKey = `${newSource}_${e.target}`;
      if (!seenEdges.has(edgeKey)) {
        seenEdges.add(edgeKey);
        rewrittenEdges.push({
          ...e,
          id: `e_${newSource}_${e.target}`,
          source: newSource,
        });
      }
    });

    return { nodes: finalNodes, edges: rewrittenEdges };
  };

  // Simple hash function for consistent group IDs
  const hashCode = (str) => {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash + str.charCodeAt(i)) | 0;
    }
    return hash;
  };

  return { applyState, applyVisibility, compressEdges, groupInputs };
});
