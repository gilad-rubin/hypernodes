"""Pipeline visualization module.

This module provides visualization capabilities for HyperNodes pipelines,
including Graphviz-based static visualizations and interactive IPyWidget-based
visualizations.
"""

from .graphviz_ui import (
    DESIGN_STYLES,
    GraphvizStyle,
    visualize,
    visualize_legacy,
)
from .visualization_engine import VisualizationEngine, get_engine
from .graphviz_ui import GraphvizEngine
from .js_ui import (
    IPyWidgetEngine,
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
