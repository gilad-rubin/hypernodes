"""Progress visualization callback using tqdm."""

import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

try:
    from tqdm import TqdmExperimentalWarning

    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
except ImportError:
    pass

from ..callbacks import CallbackContext, PipelineCallback
from .environment import is_jupyter


def _env_flag(name: str, default: bool = False) -> bool:
    """Read boolean-like environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ProgressConfig:
    """Configuration for progress bars."""

    enable: bool = True
    leave: bool = True
    bar_format: str = (
        "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}]"
    )
    debug_enabled: bool = False


class ProgressBackend(Protocol):
    """Protocol for progress bar backends."""

    @property
    def name(self) -> str:
        """Backend name."""
        ...

    def create_bar(
        self,
        desc: str,
        total: Optional[int] = None,
        position: int = 0,
        leave: bool = True,
        **kwargs,
    ) -> Any:
        """Create a progress bar."""
        ...

    def close_bar(self, bar: Any) -> None:
        """Close a progress bar."""
        ...

    def refresh_bar(self, bar: Any) -> None:
        """Refresh a progress bar."""
        ...


class BaseTqdmBackend:
    """Base class for tqdm-based backends."""

    def __init__(self, config: ProgressConfig, tqdm_module: Any, name: str):
        self.config = config
        self.tqdm = tqdm_module
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def create_bar(
        self,
        desc: str,
        total: Optional[int] = None,
        position: int = 0,
        leave: bool = True,
        **kwargs,
    ) -> Any:
        return self.tqdm(
            desc=desc,
            total=total,
            position=position,
            leave=leave,
            dynamic_ncols=True,
            smoothing=0.1,
            bar_format=self.config.bar_format,
            **kwargs,
        )

    def close_bar(self, bar: Any) -> None:
        bar.close()

    def refresh_bar(self, bar: Any) -> None:
        bar.refresh()


class StandardBackend(BaseTqdmBackend):
    """Standard tqdm backend."""

    def __init__(self, config: ProgressConfig):
        from tqdm import tqdm

        super().__init__(config, tqdm, "std")


class NotebookBackend(BaseTqdmBackend):
    """Jupyter notebook backend."""

    def __init__(self, config: ProgressConfig):
        try:
            from tqdm.notebook import tqdm
        except ImportError:
            from tqdm import tqdm
        super().__init__(config, tqdm, "notebook")


class RichBackend(BaseTqdmBackend):
    """Rich console backend."""

    def __init__(self, config: ProgressConfig):
        try:
            from tqdm.rich import tqdm
        except (ImportError, AttributeError):
            from tqdm import tqdm
        super().__init__(config, tqdm, "rich")

    def refresh_bar(self, bar: Any) -> None:
        # Rich backend often doesn't need explicit refresh or it can cause artifacts
        pass


def create_backend(config: ProgressConfig) -> ProgressBackend:
    """Factory to create appropriate backend."""
    if is_jupyter():
        return NotebookBackend(config)

    try:
        import rich  # noqa: F401
        from tqdm.rich import tqdm  # noqa: F401

        return RichBackend(config)
    except (ImportError, AttributeError):
        return StandardBackend(config)


class ProgressCallback(PipelineCallback):
    """Live progress bars for pipeline execution.

    Refactored to adhere to SOLID principles and cleaner architecture.
    """

    def __init__(self, enable: bool = True):
        self.config = ProgressConfig(
            enable=enable,
            debug_enabled=_env_flag("HN_PROGRESS_DEBUG"),
            leave=True,  # Force linger on
        )
        self.backend = create_backend(self.config)
        self._bars: Dict[str, Any] = {}

        self._debug(f"Init enable={self.config.enable} backend={self.backend.name}")

    def _debug(self, message: str) -> None:
        if self.config.debug_enabled:
            print(f"[ProgressCallback] {message}")

    def _create_bar(
        self,
        desc: str,
        total: Optional[int] = None,
        position: int = 0,
        leave: bool = True,
        **kwargs,
    ) -> Any:
        if not self.config.enable:
            return None

        self._debug(
            f"Create bar desc={desc!r} total={total} position={position} leave={leave}"
        )

        return self.backend.create_bar(
            desc=desc, total=total, position=position, leave=leave, **kwargs
        )

    # -------------------------------------------------------------------------
    # Pipeline (non-map)
    # -------------------------------------------------------------------------

    def on_pipeline_start(
        self, pipeline_id: str, inputs: Dict, ctx: CallbackContext
    ) -> None:
        if not self.config.enable:
            return

        if ctx.get("_in_map", False):
            return

        metadata = ctx.get_pipeline_metadata(pipeline_id)
        total_nodes = metadata.get("total_nodes", 0)
        pipeline_name = metadata.get("pipeline_name", pipeline_id)

        bar = self._create_bar(
            desc=f"{pipeline_name}",
            total=total_nodes if total_nodes > 0 else None,
            position=ctx.depth,
            leave=self.config.leave,
        )

        ctx.set(f"progress_bar:{pipeline_id}", bar)
        ctx.set(f"pipeline_name:{pipeline_id}", pipeline_name)
        self._bars[f"pipeline:{pipeline_id}"] = bar

    def on_pipeline_end(
        self, pipeline_id: str, outputs: Dict, duration: float, ctx: CallbackContext
    ) -> None:
        if not self.config.enable:
            return

        if ctx.get("_in_map", False):
            return

        bar = ctx.get(f"progress_bar:{pipeline_id}")
        if bar:
            pipeline_name = ctx.get(f"pipeline_name:{pipeline_id}", pipeline_id)

            if bar.total is not None and bar.n < bar.total:
                bar.update(bar.total - bar.n)

            bar.set_description(f"{pipeline_name} ✓ ({duration:.2f}s)")
            bar.leave = self.config.leave

            self.backend.refresh_bar(bar)
            self.backend.close_bar(bar)

            if f"pipeline:{pipeline_id}" in self._bars:
                del self._bars[f"pipeline:{pipeline_id}"]

    # -------------------------------------------------------------------------
    # Nodes (non-map)
    # -------------------------------------------------------------------------

    def on_node_start(self, node_id: str, inputs: Dict, ctx: CallbackContext) -> None:
        if not self.config.enable:
            return

        if ctx.get("_in_map", False):
            return

        if ctx.get(f"_is_pipeline_node:{node_id}", False):
            return

        bar = self._create_bar(
            desc=f"{node_id}",
            total=1,
            position=ctx.depth + 1,
            leave=self.config.leave,
        )

        ctx.set(f"progress_bar:{node_id}", bar)
        self._bars[f"node:{node_id}"] = bar

    def on_node_end(
        self, node_id: str, outputs: Dict, duration: float, ctx: CallbackContext
    ) -> None:
        if not self.config.enable:
            return

        in_map = ctx.get("_in_map", False)

        if in_map:
            self._update_map_node_progress(node_id, outputs, ctx)
        else:
            self._update_regular_node_progress(node_id, duration, ctx)

    def _update_map_node_progress(
        self, node_id: str, outputs: Dict, ctx: CallbackContext
    ) -> None:
        map_bars = ctx.get("map_node_bars", {})
        map_node_progress = ctx.get("map_node_progress", {})

        if node_id in map_bars:
            bar = map_bars[node_id]
            increment = outputs.get("_progress_increment", 1) if outputs else 1
            progress = map_node_progress.get(node_id, 0) + increment
            map_node_progress[node_id] = progress
            ctx.set("map_node_progress", map_node_progress)
            bar.update(increment)

    def _update_regular_node_progress(
        self, node_id: str, duration: float, ctx: CallbackContext
    ) -> None:
        bar = ctx.get(f"progress_bar:{node_id}")
        if bar:
            bar.set_description(f"{node_id} ✓ ({duration:.2f}s)")
            bar.update(1)
            bar.leave = self.config.leave

            self.backend.refresh_bar(bar)
            self.backend.close_bar(bar)

        pipeline_bar = ctx.get(f"progress_bar:{ctx.current_pipeline_id}")
        if pipeline_bar:
            pipeline_bar.update(1)

    def on_node_cached(
        self, node_id: str, signature: str, ctx: CallbackContext
    ) -> None:
        if not self.config.enable:
            return

        in_map = ctx.get("_in_map", False)

        if in_map:
            self._update_map_node_cached(node_id, ctx)
        else:
            self._update_regular_node_cached(node_id, ctx)

    def _update_map_node_cached(self, node_id: str, ctx: CallbackContext) -> None:
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
            bar.update(1)

    def _update_regular_node_cached(self, node_id: str, ctx: CallbackContext) -> None:
        bar = ctx.get(f"progress_bar:{node_id}")
        if bar:
            bar.set_description(f"{node_id} ⚡ CACHED")
            bar.update(1)
            bar.leave = self.config.leave

            self.backend.refresh_bar(bar)
            self.backend.close_bar(bar)

        pipeline_bar = ctx.get(f"progress_bar:{ctx.current_pipeline_id}")
        if pipeline_bar:
            pipeline_bar.update(1)

    # -------------------------------------------------------------------------
    # Map operations
    # -------------------------------------------------------------------------

    def on_map_start(self, total_items: int, ctx: CallbackContext) -> None:
        if not self.config.enable:
            return

        pipeline_id = ctx.current_pipeline_id
        metadata = ctx.get_pipeline_metadata(pipeline_id)
        node_ids = metadata.get("node_ids", [])
        pipeline_name = metadata.get("pipeline_name", pipeline_id)

        self._debug(
            f"Map start pipeline={pipeline_id} name={pipeline_name} "
            f"total_items={total_items} nodes={node_ids}"
        )

        pipeline_bar = self._create_bar(
            desc=f"{pipeline_name}",
            total=total_items,
            position=ctx.depth,
            leave=self.config.leave,
        )
        ctx.set(f"progress_bar:{pipeline_id}", pipeline_bar)
        ctx.set(f"pipeline_name:{pipeline_id}", pipeline_name)

        map_bars = {}
        for idx, node_id in enumerate(node_ids):
            bar = self._create_bar(
                desc=f"{node_id}",
                total=total_items,
                position=ctx.depth + 1 + idx,
                leave=self.config.leave,  # Always linger as requested
            )
            map_bars[node_id] = bar

        ctx.set("map_node_bars", map_bars)
        ctx.set("map_start_time", time.time())
        ctx.set("map_total_items", total_items)
        ctx.set("map_node_progress", {node_id: 0 for node_id in node_ids})
        ctx.set("map_node_cache_hits", {node_id: 0 for node_id in node_ids})

        self._bars["map_bars"] = map_bars
        self._bars[f"pipeline:{pipeline_id}"] = pipeline_bar

    def on_map_item_end(
        self, item_index: int, duration: float, ctx: CallbackContext
    ) -> None:
        if not self.config.enable:
            return

        pipeline_id = ctx.current_pipeline_id
        pipeline_bar = ctx.get(f"progress_bar:{pipeline_id}")
        if pipeline_bar:
            pipeline_bar.update(1)

    def on_map_end(self, total_duration: float, ctx: CallbackContext) -> None:
        if not self.config.enable:
            return

        pipeline_id = ctx.current_pipeline_id
        total_items = ctx.get("map_total_items", 0)
        rate = total_items / total_duration if total_duration > 0 else 0.0

        # Update node bars FIRST (bottom-up or just children before parent)
        map_bars = ctx.get("map_node_bars", {})
        map_node_progress = ctx.get("map_node_progress", {})
        map_node_cache_hits = ctx.get("map_node_cache_hits", {})

        for node_id, bar in map_bars.items():
            if not bar:
                continue

            progress = map_node_progress.get(node_id, 0)
            cache_hits = map_node_cache_hits.get(node_id, 0)
            cache_pct = (cache_hits / progress * 100.0) if progress > 0 else 0.0

            summary = (
                f"{node_id} ✓ ({total_duration:.2f}s, "
                f"{rate:.1f} items/s, {cache_pct:.1f}% cached)"
            )

            self._finalize_map_bar(bar, total_items, summary)

        # Update pipeline bar LAST (parent)
        pipeline_bar = ctx.get(f"progress_bar:{pipeline_id}")
        if pipeline_bar:
            self._finalize_map_bar(
                pipeline_bar,
                total_items,
                f"{ctx.get(f'pipeline_name:{pipeline_id}', pipeline_id)} ✓ "
                f"({total_duration:.2f}s, {rate:.1f} items/s)",
            )

        # Cleanup
        if "map_bars" in self._bars:
            del self._bars["map_bars"]
        if f"pipeline:{pipeline_id}" in self._bars:
            del self._bars[f"pipeline:{pipeline_id}"]

    def _finalize_map_bar(self, bar: Any, total: int, desc: str) -> None:
        """Helper to finalize a map progress bar."""
        if total and bar.n < total:
            bar.update(total - bar.n)

        bar.set_description(desc)
        bar.leave = self.config.leave

        self.backend.refresh_bar(bar)
        self.backend.close_bar(bar)

    def on_nested_pipeline_end(
        self,
        parent_id: str,
        child_pipeline_id: str,
        duration: float,
        ctx: CallbackContext,
    ) -> None:
        if not self.config.enable:
            return

        pipeline_bar = ctx.get(f"progress_bar:{parent_id}")
        if pipeline_bar:
            pipeline_bar.update(1)

    def on_error(self, node_id: str, error: Exception, ctx: CallbackContext) -> None:
        if not self.config.enable:
            return

        bar = ctx.get(f"progress_bar:{node_id}")
        if bar:
            bar.set_description(f"{node_id} ✗ FAILED")
            bar.leave = self.config.leave
            self.backend.refresh_bar(bar)
            self.backend.close_bar(bar)
