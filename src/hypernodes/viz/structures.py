from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class BaseVizNode:
    id: str
    parent_id: Optional[str]  # For nesting within a pipeline group

@dataclass
class FunctionNode(BaseVizNode):
    """Represents a standard function execution node."""
    label: str
    function_name: str

@dataclass
class DualNode(FunctionNode):
    """Represents a DualNode (batch-optimized)."""
    pass

@dataclass
class PipelineNode(BaseVizNode):
    """Represents a nested pipeline (sub-graph)."""
    label: str
    is_expanded: bool = False

@dataclass
class DataNode(BaseVizNode):
    """Represents a data artifact (input or output)."""
    name: str
    type_hint: Optional[str] = None
    is_bound: bool = False
    source_id: Optional[str] = None  # ID of the node producing this data (None if external input)

@dataclass
class GroupDataNode(BaseVizNode):
    """Represents a group of DataNodes that share source/dest/bound state."""
    nodes: List[DataNode]
    is_bound: bool = False
    source_id: Optional[str] = None

# Union type for all nodes
VizNode = Union[FunctionNode, DualNode, PipelineNode, DataNode, GroupDataNode]

@dataclass
class VizEdge:
    source: str
    target: str
    label: Optional[str] = None

@dataclass
class VisualizationGraph:
    nodes: List[VizNode]
    edges: List[VizEdge]
