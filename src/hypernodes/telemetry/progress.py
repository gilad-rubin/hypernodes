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

        self._bar_format = (
            "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]"
        )

        # Import appropriate tqdm
        if self.use_notebook:
            try:
                from tqdm.notebook import tqdm

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

    def _create_bar(
        self,
        desc: str,
        total: Optional[int] = None,
        position: int = 0,
        leave: bool = True,
        **kwargs,
    ):
        """Create a progress bar with consistent styling.

        Args:
            desc: Description text
            total: Total items (None for indeterminate)
            position: Vertical position (0=top, 1=nested, etc.)
            leave: Whether to keep bar after completion
            **kwargs: Additional tqdm arguments

        Returns:
            tqdm progress bar instance
        """
        if not self.enable:
            return None

        return self.tqdm(
            desc=desc,
            total=total,
            position=position,
            leave=leave,
            dynamic_ncols=True,
            smoothing=0.1,
            bar_format=self._bar_format,
            **kwargs,
        )

    def on_pipeline_start(
        self, pipeline_id: str, inputs: Dict, ctx: CallbackContext
    ) -> None:
        """Create progress bar for pipeline execution."""
        if not self.enable:
            return

        # Skip creating pipeline bar if we're in a map operation
        # (map bars are created at map_start instead)
        if ctx.get("_in_map", False):
            return

        metadata = ctx.get_pipeline_metadata(pipeline_id)
        total_nodes = metadata.get("total_nodes", 0)

        # Get pipeline name from metadata if available
        pipeline_name = metadata.get("pipeline_name", pipeline_id)

        # Use depth as position, keep all bars visible
        bar = self._create_bar(
            desc=f"{pipeline_name}",
            total=total_nodes if total_nodes > 0 else None,
            position=ctx.depth,
            leave=True,
        )

        ctx.set(f"progress_bar:{pipeline_id}", bar)
        ctx.set(f"pipeline_name:{pipeline_id}", pipeline_name)
        self._bars[f"pipeline:{pipeline_id}"] = bar

    def on_node_start(self, node_id: str, inputs: Dict, ctx: CallbackContext) -> None:
        """Create progress bar for node execution."""
        if not self.enable:
            return

        # Update pipeline bar to show currently running node
        in_map = ctx.get("_in_map", False)
        pipeline_id = ctx.current_pipeline_id
        pipeline_bar = ctx.get(f"progress_bar:{pipeline_id}")

        if pipeline_bar:
            # Show current node in pipeline bar description
            pipeline_name = ctx.get(f"pipeline_name:{pipeline_id}", pipeline_id)
            pipeline_bar.set_description(f"{pipeline_name} → {node_id}")

        # Skip creating node bar if we're in a map operation
        # (map node bars are created at map_start instead)
        if in_map:
            return

        # Skip creating node bar for PipelineNodes - they will create their own
        # progress bars for their internal pipeline/map execution
        # This is marked in context by backend when calling on_nested_pipeline_start
        if ctx.get(f"_is_pipeline_node:{node_id}", False):
            return

        # Regular nodes get a progress bar (depth+1), keep visible after completion
        bar = self._create_bar(
            desc=f"{node_id}",
            total=1,
            position=ctx.depth + 1,
            leave=True,
        )

        ctx.set(f"progress_bar:{node_id}", bar)
        self._bars[f"node:{node_id}"] = bar

    def on_node_end(
        self, node_id: str, outputs: Dict, duration: float, ctx: CallbackContext
    ) -> None:
        """Update node progress bar on completion."""
        if not self.enable:
            return

        # Check if we're in a map operation
        in_map = ctx.get("_in_map", False)

        if in_map:
            # Update the map node bar for this node
            map_bars = ctx.get("map_node_bars", {})
            map_node_progress = ctx.get("map_node_progress", {})

            if node_id in map_bars:
                bar = map_bars[node_id]
                progress = map_node_progress.get(node_id, 0) + 1
                map_node_progress[node_id] = progress
                ctx.set("map_node_progress", map_node_progress)

                # Calculate stats
                start_time = ctx.get("map_start_time", time.time())
                elapsed = time.time() - start_time
                rate = progress / elapsed if elapsed > 0 else 0
                cache_hits = ctx.get("map_node_cache_hits", {}).get(node_id, 0)
                cache_pct = (cache_hits / progress * 100) if progress > 0 else 0

                # Update bar
                bar.set_description(
                    f"{node_id} [{rate:.1f} items/s, {cache_pct:.1f}% cached]"
                )
                bar.update(1)
        else:
            # Regular node execution (not in map)
            bar = ctx.get(f"progress_bar:{node_id}")
            if bar:
                bar.set_description(f"{node_id} ✓ ({duration:.2f}s)")
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

        # Check if we're in a map operation
        in_map = ctx.get("_in_map", False)

        if in_map:
            # Update cache hits and progress for this node
            map_bars = ctx.get("map_node_bars", {})
            map_node_progress = ctx.get("map_node_progress", {})
            map_node_cache_hits = ctx.get("map_node_cache_hits", {})

            if node_id in map_bars:
                bar = map_bars[node_id]
                progress = map_node_progress.get(node_id, 0) + 1
                cache_hits = map_node_cache_hits.get(node_id, 0) + 1
                map_node_progress[node_id] = progress
                map_node_cache_hits[node_id] = cache_hits
                ctx.set("map_node_progress", map_node_progress)
                ctx.set("map_node_cache_hits", map_node_cache_hits)

                # Calculate stats
                start_time = ctx.get("map_start_time", time.time())
                elapsed = time.time() - start_time
                rate = progress / elapsed if elapsed > 0 else 0
                cache_pct = (cache_hits / progress * 100) if progress > 0 else 0

                # Update bar
                bar.set_description(
                    f"{node_id} [{rate:.1f} items/s, {cache_pct:.1f}% cached]"
                )
                bar.update(1)
        else:
            # Regular cached node (not in map)
            bar = ctx.get(f"progress_bar:{node_id}")
            if bar:
                bar.set_description(f"{node_id} ⚡ CACHED")
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

        # Skip if in map (handled by map_end)
        if ctx.get("_in_map", False):
            return

        bar = ctx.get(f"progress_bar:{pipeline_id}")
        if bar:
            pipeline_name = ctx.get(f"pipeline_name:{pipeline_id}", pipeline_id)
            bar.set_description(f"{pipeline_name} ✓ ({duration:.2f}s)")
            bar.close()

            # Clean up
            if f"pipeline:{pipeline_id}" in self._bars:
                del self._bars[f"pipeline:{pipeline_id}"]

    def on_map_start(self, total_items: int, ctx: CallbackContext) -> None:
        """Create per-node progress bars for map operation."""
        if not self.enable:
            return

        # Get pipeline metadata to know which nodes we're mapping over
        pipeline_id = ctx.current_pipeline_id
        metadata = ctx.get_pipeline_metadata(pipeline_id)
        node_ids = metadata.get("node_ids", [])
        pipeline_name = metadata.get("pipeline_name", pipeline_id)

        # Create main pipeline bar for map operation
        pipeline_bar = self._create_bar(
            desc=f"Running {pipeline_name} with {total_items} examples...",
            total=total_items,
            position=ctx.depth,
            leave=True,
        )
        ctx.set(f"progress_bar:{pipeline_id}", pipeline_bar)
        ctx.set(f"pipeline_name:{pipeline_id}", pipeline_name)

        # Create one progress bar for each node with unique positions
        map_bars = {}

        for idx, node_id in enumerate(node_ids):
            bar = self._create_bar(
                desc=f"{node_id}",
                total=total_items,
                position=ctx.depth + 1 + idx,  # Each node gets unique position
                leave=True,
            )
            map_bars[node_id] = bar

        # Store bars and metadata
        ctx.set("map_node_bars", map_bars)
        ctx.set("map_start_time", time.time())
        ctx.set("map_total_items", total_items)
        ctx.set("map_node_progress", {node_id: 0 for node_id in node_ids})
        ctx.set("map_node_cache_hits", {node_id: 0 for node_id in node_ids})
        self._bars["map_bars"] = map_bars
        self._bars[f"pipeline:{pipeline_id}"] = pipeline_bar

    def on_map_item_start(self, item_index: int, ctx: CallbackContext) -> None:
        """Track map item start."""
        # No-op: node-level tracking happens in on_node_end
        pass

    def on_map_item_end(
        self, item_index: int, duration: float, ctx: CallbackContext
    ) -> None:
        """Update pipeline bar when a map item completes."""
        if not self.enable:
            return

        # Update the main pipeline bar for map progress
        pipeline_id = ctx.current_pipeline_id
        pipeline_bar = ctx.get(f"progress_bar:{pipeline_id}")
        if pipeline_bar:
            pipeline_bar.update(1)

    def on_map_item_cached(
        self, item_index: int, signature: str, ctx: CallbackContext
    ) -> None:
        """Track cache hits in map operations (no-op, handled per-node)."""
        # No-op: cache tracking happens in on_node_cached
        pass

    def on_map_end(self, total_duration: float, ctx: CallbackContext) -> None:
        """Close all map node progress bars."""
        if not self.enable:
            return

        map_bars = ctx.get("map_node_bars", {})
        map_node_progress = ctx.get("map_node_progress", {})
        map_node_cache_hits = ctx.get("map_node_cache_hits", {})
        total_items = ctx.get("map_total_items", 0)
        pipeline_id = ctx.current_pipeline_id

        # Update final descriptions and close bars
        rate = total_items / total_duration if total_duration > 0 else 0

        for node_id, bar in map_bars.items():
            progress = map_node_progress.get(node_id, 0)
            cache_hits = map_node_cache_hits.get(node_id, 0)
            cache_pct = (cache_hits / progress * 100) if progress > 0 else 0

            bar.set_description(
                f"{node_id} ✓ ({total_duration:.2f}s, {rate:.1f} items/s, {cache_pct:.1f}% cached)"
            )
            bar.close()

        # Update and close pipeline bar
        pipeline_bar = ctx.get(f"progress_bar:{pipeline_id}")
        pipeline_name = ctx.get(f"pipeline_name:{pipeline_id}", pipeline_id)
        if pipeline_bar:
            pipeline_bar.set_description(
                f"{pipeline_name} ✓ ({total_duration:.2f}s, {rate:.1f} items/s)"
            )
            pipeline_bar.close()

        # Clean up
        if "map_bars" in self._bars:
            del self._bars["map_bars"]
        if f"pipeline:{pipeline_id}" in self._bars:
            del self._bars[f"pipeline:{pipeline_id}"]

    def on_nested_pipeline_start(
        self, parent_id: str, child_pipeline_id: str, ctx: CallbackContext
    ) -> None:
        """Track nested pipeline start (bar created by on_pipeline_start)."""
        # Don't create a separate placeholder bar - the nested pipeline will create its own
        # bar via on_pipeline_start, which provides clearer hierarchy
        pass

    def on_nested_pipeline_end(
        self,
        parent_id: str,
        child_pipeline_id: str,
        duration: float,
        ctx: CallbackContext,
    ) -> None:
        """Update parent progress bar when nested pipeline completes."""
        if not self.enable:
            return

        # Update parent pipeline bar
        pipeline_bar = ctx.get(f"progress_bar:{parent_id}")
        if pipeline_bar:
            pipeline_bar.update(1)

    def on_error(self, node_id: str, error: Exception, ctx: CallbackContext) -> None:
        """Mark progress bar as failed on error."""
        if not self.enable:
            return

        bar = ctx.get(f"progress_bar:{node_id}")
        if bar:
            bar.set_description(f"{node_id} ✗ FAILED")
            bar.close()
