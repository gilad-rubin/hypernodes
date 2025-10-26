"""Progress visualization callback using tqdm."""

import time
import warnings
from typing import Dict, Optional

from tqdm import TqdmExperimentalWarning

from ..callbacks import CallbackContext, PipelineCallback
from .environment import is_jupyter

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


class ProgressCallback(PipelineCallback):
    """Live progress bars for pipeline execution.

    Automatically detects environment:
    - Jupyter: Uses tqdm.notebook with HTML widgets
    - CLI: Uses tqdm with rich formatting

    Features:
    - Hierarchical display (indented by nesting depth)
    - Per-node progress tracking
    - Map operation statistics (items/sec, cache hits)
    - Time elapsed display

    Example:
        >>> from hypernodes import Pipeline
        >>> from hypernodes.telemetry import ProgressCallback
        >>>
        >>> pipeline = Pipeline(
        ...     nodes=[node1, node2],
        ...     callbacks=[ProgressCallback()]
        ... )
        >>> result = pipeline.run(inputs={...})
    """

    def __init__(self, enable: bool = True):
        """Initialize progress callback.

        Args:
            enable: Whether to enable progress bars (useful for disabling in tests)
        """
        self.enable = enable
        self.use_notebook = is_jupyter()
        self._bars = {}  # Track active progress bars

        # Import appropriate tqdm
        if self.use_notebook:
            try:
                from tqdm.rich import tqdm

                self.tqdm = tqdm
            except ImportError:
                # Fallback to regular tqdm
                from tqdm import tqdm

                self.tqdm = tqdm
        else:
            try:
                from tqdm.rich import tqdm

                self.tqdm = tqdm
            except (ImportError, AttributeError):
                # Fallback to regular tqdm
                from tqdm import tqdm

                self.tqdm = tqdm

    def _create_bar(self, desc: str, total: Optional[int] = None, **kwargs):
        """Create a progress bar with consistent styling.

        Args:
            desc: Description text
            total: Total items (None for indeterminate)
            **kwargs: Additional tqdm arguments

        Returns:
            tqdm progress bar instance
        """
        if not self.enable:
            return None

        return self.tqdm(desc=desc, total=total, leave=True, **kwargs)

    def on_pipeline_start(
        self, pipeline_id: str, inputs: Dict, ctx: CallbackContext
    ) -> None:
        """Create progress bar for pipeline execution."""
        if not self.enable:
            return

        indent = "  " * ctx.depth
        metadata = ctx.get_pipeline_metadata(pipeline_id)
        total_nodes = metadata.get("total_nodes", 0)

        bar = self._create_bar(
            desc=f"{indent}Pipeline: {pipeline_id}",
            total=total_nodes if total_nodes > 0 else None,
        )

        ctx.set(f"progress_bar:{pipeline_id}", bar)
        self._bars[f"pipeline:{pipeline_id}"] = bar

    def on_node_start(self, node_id: str, inputs: Dict, ctx: CallbackContext) -> None:
        """Create progress bar for node execution."""
        if not self.enable:
            return

        indent = "  " * (ctx.depth + 1)
        bar = self._create_bar(desc=f"{indent}├─ {node_id}", total=1)

        ctx.set(f"progress_bar:{node_id}", bar)
        self._bars[f"node:{node_id}"] = bar

    def on_node_end(
        self, node_id: str, outputs: Dict, duration: float, ctx: CallbackContext
    ) -> None:
        """Update node progress bar on completion."""
        if not self.enable:
            return

        bar = ctx.get(f"progress_bar:{node_id}")
        if bar:
            indent = "  " * (ctx.depth + 1)
            bar.set_description(f"{indent}├─ {node_id} ✓ ({duration:.2f}s)")
            bar.update(1)
            bar.close()

        # Update parent pipeline bar
        pipeline_bar = ctx.get(f"progress_bar:{ctx.current_pipeline_id}")
        if pipeline_bar:
            pipeline_bar.update(1)

    def on_node_cached(
        self, node_id: str, signature: str, ctx: CallbackContext
    ) -> None:
        """Handle cached node (instant completion)."""
        if not self.enable:
            return

        indent = "  " * (ctx.depth + 1)
        bar = ctx.get(f"progress_bar:{node_id}")
        if bar:
            bar.set_description(f"{indent}├─ {node_id} ⚡ CACHED")
            bar.update(1)
            bar.close()

        # Update parent pipeline bar
        pipeline_bar = ctx.get(f"progress_bar:{ctx.current_pipeline_id}")
        if pipeline_bar:
            pipeline_bar.update(1)

    def on_pipeline_end(
        self, pipeline_id: str, outputs: Dict, duration: float, ctx: CallbackContext
    ) -> None:
        """Close pipeline progress bar."""
        if not self.enable:
            return

        bar = ctx.get(f"progress_bar:{pipeline_id}")
        if bar:
            indent = "  " * ctx.depth
            bar.set_description(f"{indent}Pipeline: {pipeline_id} ✓ ({duration:.2f}s)")
            bar.close()

            # Clean up
            if f"pipeline:{pipeline_id}" in self._bars:
                del self._bars[f"pipeline:{pipeline_id}"]

    def on_map_start(self, total_items: int, ctx: CallbackContext) -> None:
        """Create progress bar for map operation."""
        if not self.enable:
            return

        indent = "  " * (ctx.depth + 1)
        bar = self._create_bar(desc=f"{indent}Map", total=total_items)

        ctx.set("map_progress_bar", bar)
        ctx.set("map_start_time", time.time())
        ctx.set("map_cache_hits", 0)
        ctx.set("map_processed", 0)
        self._bars["map"] = bar

    def on_map_item_start(self, item_index: int, ctx: CallbackContext) -> None:
        """Track map item start (optional detailed tracking)."""
        pass

    def on_map_item_end(
        self, item_index: int, duration: float, ctx: CallbackContext
    ) -> None:
        """Update map progress bar after each item."""
        if not self.enable:
            return

        bar = ctx.get("map_progress_bar")
        if bar:
            processed = ctx.get("map_processed", 0) + 1
            ctx.set("map_processed", processed)

            # Calculate rate
            start_time = ctx.get("map_start_time", time.time())
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0

            # Calculate cache hit percentage
            cache_hits = ctx.get("map_cache_hits", 0)
            cache_pct = (cache_hits / processed * 100) if processed > 0 else 0

            # Update description with stats
            indent = "  " * (ctx.depth + 1)
            bar.set_description(
                f"{indent}Map [{rate:.1f} items/s, {cache_pct:.1f}% cached]"
            )
            bar.update(1)

    def on_map_item_cached(
        self, item_index: int, signature: str, ctx: CallbackContext
    ) -> None:
        """Track cache hits in map operations."""
        if not self.enable:
            return

        cache_hits = ctx.get("map_cache_hits", 0) + 1
        ctx.set("map_cache_hits", cache_hits)

        # Also update progress
        self.on_map_item_end(item_index, 0.0, ctx)

    def on_map_end(self, total_duration: float, ctx: CallbackContext) -> None:
        """Close map progress bar."""
        if not self.enable:
            return

        bar = ctx.get("map_progress_bar")
        if bar:
            processed = ctx.get("map_processed", 0)
            cache_hits = ctx.get("map_cache_hits", 0)
            cache_pct = (cache_hits / processed * 100) if processed > 0 else 0
            rate = processed / total_duration if total_duration > 0 else 0

            indent = "  " * (ctx.depth + 1)
            bar.set_description(
                f"{indent}Map ✓ ({total_duration:.2f}s, {rate:.1f} items/s, {cache_pct:.1f}% cached)"
            )
            bar.close()

            # Clean up
            if "map" in self._bars:
                del self._bars["map"]

    def on_error(self, node_id: str, error: Exception, ctx: CallbackContext) -> None:
        """Mark progress bar as failed on error."""
        if not self.enable:
            return

        bar = ctx.get(f"progress_bar:{node_id}")
        if bar:
            indent = "  " * (ctx.depth + 1)
            bar.set_description(f"{indent}├─ {node_id} ✗ FAILED")
            bar.close()
