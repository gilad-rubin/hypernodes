"""Callback system for pipeline execution lifecycle events."""

from typing import Any, Dict, List, Optional


class CallbackContext:
    """Shared state across all callbacks in a single execution.

    Automatically tracks hierarchical execution to enable seamless nested pipeline handling.
    """

    def __init__(self):
        """Initialize callback context."""
        self.data: Dict[str, Any] = {}
        self._hierarchy_stack: List[str] = []  # Stack of pipeline IDs
        self._depth: int = 0

    def set(self, key: str, value: Any) -> None:
        """Store a value for other callbacks to access.

        Args:
            key: Key to store value under
            value: Value to store
        """
        self.data[key] = value

    def get(self, key: str, default=None) -> Any:
        """Retrieve a value set by another callback.

        Args:
            key: Key to retrieve
            default: Default value if key not found

        Returns:
            Stored value or default
        """
        return self.data.get(key, default)

    def push_pipeline(self, pipeline_id: str) -> None:
        """Track entering a nested pipeline (managed by executor).

        Args:
            pipeline_id: ID of pipeline being entered
        """
        self._hierarchy_stack.append(pipeline_id)
        self._depth += 1

    def pop_pipeline(self) -> str:
        """Track exiting a nested pipeline (managed by executor).

        Returns:
            ID of pipeline being exited
        """
        self._depth -= 1
        return self._hierarchy_stack.pop()

    def get_pipeline_metadata(self, pipeline_id: str) -> Dict:
        """Get metadata about a pipeline (e.g., total_nodes).

        Args:
            pipeline_id: ID of pipeline

        Returns:
            Dictionary of metadata
        """
        return self.data.get(f"_pipeline_metadata:{pipeline_id}", {})

    def set_pipeline_metadata(self, pipeline_id: str, metadata: Dict) -> None:
        """Store metadata about a pipeline (managed by executor).

        Args:
            pipeline_id: ID of pipeline
            metadata: Metadata dictionary
        """
        self.data[f"_pipeline_metadata:{pipeline_id}"] = metadata

    @property
    def current_pipeline_id(self) -> str:
        """Get the currently executing pipeline ID.

        Returns:
            Current pipeline ID or None if at root
        """
        return self._hierarchy_stack[-1] if self._hierarchy_stack else None

    @property
    def parent_pipeline_id(self) -> str:
        """Get the parent pipeline ID (None if at root).

        Returns:
            Parent pipeline ID or None
        """
        return self._hierarchy_stack[-2] if len(self._hierarchy_stack) >= 2 else None

    @property
    def depth(self) -> int:
        """Current nesting depth (0 = root pipeline).

        Returns:
            Nesting depth
        """
        return self._depth

    @property
    def hierarchy_path(self) -> List[str]:
        """Full path from root to current pipeline.

        Returns:
            List of pipeline IDs from root to current
        """
        return self._hierarchy_stack.copy()


class PipelineCallback:
    """Base class for pipeline callbacks.

    Override methods to receive lifecycle events during pipeline execution.
    All methods are optional - only override what you need.
    """

    def on_pipeline_start(
        self, pipeline_id: str, inputs: Dict, ctx: CallbackContext
    ) -> None:
        """Called when pipeline execution starts.

        Args:
            pipeline_id: ID of the pipeline
            inputs: Input dictionary
            ctx: Callback context
        """
        pass

    def on_node_start(self, node_id: str, inputs: Dict, ctx: CallbackContext) -> None:
        """Called before a node executes.

        Args:
            node_id: ID of the node
            inputs: Node input dictionary
            ctx: Callback context
        """
        pass

    def on_node_end(
        self, node_id: str, outputs: Dict, duration: float, ctx: CallbackContext
    ) -> None:
        """Called after a node executes.

        Args:
            node_id: ID of the node
            outputs: Node output dictionary
            duration: Execution duration in seconds
            ctx: Callback context
        """
        pass

    def on_node_cached(
        self, node_id: str, signature: str, ctx: CallbackContext
    ) -> None:
        """Called when a node is skipped due to cache hit.

        Args:
            node_id: ID of the node
            signature: Cache signature
            ctx: Callback context
        """
        pass

    def on_pipeline_end(
        self, pipeline_id: str, outputs: Dict, duration: float, ctx: CallbackContext
    ) -> None:
        """Called when pipeline execution completes.

        Args:
            pipeline_id: ID of the pipeline
            outputs: Pipeline output dictionary
            duration: Execution duration in seconds
            ctx: Callback context
        """
        pass

    def on_error(self, node_id: str, error: Exception, ctx: CallbackContext) -> None:
        """Called when a node raises an error.

        Args:
            node_id: ID of the node
            error: Exception that was raised
            ctx: Callback context
        """
        pass

    def on_nested_pipeline_start(
        self, parent_id: str, child_pipeline_id: str, ctx: CallbackContext
    ) -> None:
        """Called when a nested pipeline starts execution.

        Note: ctx.push_pipeline() is already called by executor before this hook.

        Args:
            parent_id: ID of parent pipeline
            child_pipeline_id: ID of child pipeline
            ctx: Callback context
        """
        pass

    def on_nested_pipeline_end(
        self,
        parent_id: str,
        child_pipeline_id: str,
        duration: float,
        ctx: CallbackContext,
    ) -> None:
        """Called when a nested pipeline completes execution.

        Note: ctx.pop_pipeline() is called by executor after this hook.

        Args:
            parent_id: ID of parent pipeline
            child_pipeline_id: ID of child pipeline
            duration: Execution duration in seconds
            ctx: Callback context
        """
        pass

    # Map operation hooks
    def on_map_start(self, total_items: int, ctx: CallbackContext) -> None:
        """Called when a map operation starts.

        Args:
            total_items: Total number of items to process
            ctx: Callback context
        """
        pass

    def on_map_item_start(self, item_index: int, ctx: CallbackContext) -> None:
        """Called before processing each item in a map operation.

        Args:
            item_index: Index of item being processed
            ctx: Callback context
        """
        pass

    def on_map_item_end(
        self, item_index: int, duration: float, ctx: CallbackContext
    ) -> None:
        """Called after processing each item in a map operation.

        Args:
            item_index: Index of item processed
            duration: Execution duration in seconds
            ctx: Callback context
        """
        pass

    def on_map_item_cached(
        self, item_index: int, signature: str, ctx: CallbackContext
    ) -> None:
        """Called when a map item is retrieved from cache.

        Args:
            item_index: Index of cached item
            signature: Cache signature
            ctx: Callback context
        """
        pass

    def on_map_end(self, total_duration: float, ctx: CallbackContext) -> None:
        """Called when a map operation completes.

        Args:
            total_duration: Total execution duration in seconds
            ctx: Callback context
        """
        pass

    @property
    def supported_engines(self) -> Optional[List[str]]:
        """List of supported engine class names (e.g. ['SeqEngine', 'DaftEngine']).

        If None or empty, compatible with all engines (default).
        """
        return None


class CallbackDispatcher:
    """Dispatches events to a list of callbacks."""

    def __init__(self, callbacks: List[Any]):
        self.callbacks = callbacks or []

    def notify_pipeline_start(
        self, pipeline_id: str, inputs: Dict[str, Any], ctx: CallbackContext
    ) -> None:
        for callback in self.callbacks:
            callback.on_pipeline_start(pipeline_id, inputs, ctx)

    def notify_pipeline_end(
        self,
        pipeline_id: str,
        outputs: Dict[str, Any],
        duration: float,
        ctx: CallbackContext,
    ) -> None:
        for callback in self.callbacks:
            callback.on_pipeline_end(pipeline_id, outputs, duration, ctx)

    def notify_map_start(self, total_items: int, ctx: CallbackContext) -> None:
        for callback in self.callbacks:
            callback.on_map_start(total_items, ctx)

    def notify_map_end(self, duration: float, ctx: CallbackContext) -> None:
        for callback in self.callbacks:
            callback.on_map_end(duration, ctx)

    def notify_map_item_start(self, idx: int, ctx: CallbackContext) -> None:
        for callback in self.callbacks:
            callback.on_map_item_start(idx, ctx)

    def notify_map_item_end(
        self, idx: int, duration: float, ctx: CallbackContext
    ) -> None:
        for callback in self.callbacks:
            callback.on_map_item_end(idx, duration, ctx)

    def notify_node_start(
        self, node_id: str, inputs: Dict[str, Any], ctx: CallbackContext
    ) -> None:
        for callback in self.callbacks:
            callback.on_node_start(node_id, inputs, ctx)

    def notify_node_end(
        self,
        node_id: str,
        outputs: Dict[str, Any],
        duration: float,
        ctx: CallbackContext,
    ) -> None:
        for callback in self.callbacks:
            callback.on_node_end(node_id, outputs, duration, ctx)

    def notify_node_cached(
        self, node_id: str, signature: str, ctx: CallbackContext
    ) -> None:
        for callback in self.callbacks:
            callback.on_node_cached(node_id, signature, ctx)

    def notify_error(
        self, node_id: str, error: Exception, ctx: CallbackContext
    ) -> None:
        for callback in self.callbacks:
            callback.on_error(node_id, error, ctx)
