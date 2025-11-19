"""Orchestrator for pipeline execution.

This module provides the "outer loop" logic for executing pipelines, ensuring consistent
behavior across different engines (Sequential, Daft, etc.).

Key Responsibilities:
1. Callback Dispatcher setup
2. Pipeline metadata calculation and notification
3. Start/End lifecycle event notification
4. Common validation
"""

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .callbacks import CallbackContext, CallbackDispatcher, PipelineCallback
from .node_execution import _get_node_id

if TYPE_CHECKING:
    from .pipeline import Pipeline


class ExecutionOrchestrator:
    """Helper class to orchestrate pipeline execution events.

    This class is designed to be used by Engines to handle the common boilerplate
    of firing callbacks, setting up contexts, and tracking execution time.
    """

    def __init__(
        self,
        pipeline: "Pipeline",
        callbacks: List[PipelineCallback],
        context: Optional[CallbackContext] = None,
    ):
        self.pipeline = pipeline
        self.callbacks = callbacks or []
        self.dispatcher = CallbackDispatcher(self.callbacks)

        # Context management
        self.ctx = context or CallbackContext()
        self.is_new_context = context is None

    def __enter__(self):
        if self.is_new_context:
            self.ctx.push_pipeline(self.pipeline.id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_new_context:
            self.ctx.pop_pipeline()

    def notify_start(self, inputs: Dict[str, Any]):
        """Notify pipeline start and set metadata."""
        # Set pipeline metadata
        execution_nodes = self.pipeline.graph.execution_order
        self.ctx.set_pipeline_metadata(
            self.pipeline.id,
            {
                "total_nodes": len(execution_nodes),
                "pipeline_name": self.pipeline.name or self.pipeline.id,
                "node_ids": [_get_node_id(node) for node in execution_nodes],
            },
        )

        self.start_time = time.time()
        self.dispatcher.notify_pipeline_start(self.pipeline.id, inputs, self.ctx)

    def notify_end(self, outputs: Dict[str, Any]):
        """Notify pipeline end."""
        duration = time.time() - getattr(self, "start_time", time.time())
        self.dispatcher.notify_pipeline_end(
            self.pipeline.id, outputs, duration, self.ctx
        )

    def validate_callbacks(self, engine_name: str):
        """Check if callbacks are compatible with the current engine."""
        for callback in self.callbacks:
            if hasattr(callback, "supported_engines"):
                supported = callback.supported_engines
                # If supported_engines is defined and not empty, check compatibility
                if supported and engine_name not in supported:
                    # Only warn for now, or should we raise?
                    # User requested fail early, so let's raise.
                    raise ValueError(
                        f"Callback {callback.__class__.__name__} is not compatible with "
                        f"engine {engine_name}. Supported engines: {supported}"
                    )

    # Map-specific notifications
    def notify_map_start(self, total_items: int):
        self.map_start_time = time.time()
        self.dispatcher.notify_map_start(total_items, self.ctx)

    def notify_map_end(self):
        duration = time.time() - getattr(self, "map_start_time", time.time())
        self.dispatcher.notify_map_end(duration, self.ctx)

    def notify_map_item_start(self, index: int):
        self.dispatcher.notify_map_item_start(index, self.ctx)

    def notify_map_item_end(self, index: int, duration: float):
        self.dispatcher.notify_map_item_end(index, duration, self.ctx)
