"""Pipeline visualization module.

This module provides visualization capabilities for HyperNodes pipelines,
including Graphviz-based static visualizations and interactive IPyWidget-based
visualizations.
"""

from .visualization import (
    DESIGN_STYLES,
    GraphvizStyle,
    visualize,
    visualize_legacy,
)
from .graph_serializer import GraphSerializer
from .visualization_engines import (
    GraphvizEngine,
    IPyWidgetEngine,
    VisualizationEngine,
    get_engine,
)
from .visualization_widget import (
    PipelineWidget,
    generate_widget_html,
    transform_to_react_flow,
)

__all__ = [
    # Main visualization functions
    "visualize",
    "visualize_legacy",
    # Styles
    "GraphvizStyle",
    "DESIGN_STYLES",
    # Serialization
    "GraphSerializer",
    # Engines
    "VisualizationEngine",
    "GraphvizEngine",
    "IPyWidgetEngine",
    "get_engine",
    # Widgets
    "PipelineWidget",
    "generate_widget_html",
    "transform_to_react_flow",
]

