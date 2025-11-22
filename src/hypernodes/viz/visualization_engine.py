"""Visualization engine protocol and registry.

This module defines the pluggable interface for visualization engines.
Concrete renderers live in graphviz_ui.py and js_ui.py; this stays logic-free.
"""
from typing import Any, Dict, Protocol


class VisualizationEngine(Protocol):
    """Protocol for pluggable visualization engines."""

    def render(self, serialized_graph: Dict[str, Any], **options: Any) -> Any:
        """Render a serialized graph with engine-specific options."""
        ...


def get_engine(engine_name: str) -> VisualizationEngine:
    """Resolve a visualization engine by name."""
    from .graphviz_ui import GraphvizEngine
    from .js_ui import IPyWidgetEngine

    engines = {
        "graphviz": GraphvizEngine(),
        "ipywidget": IPyWidgetEngine(),
    }
    if engine_name not in engines:
        raise ValueError(
            f"Unknown engine '{engine_name}'. Choose from: {list(engines.keys())}"
        )
    return engines[engine_name]
