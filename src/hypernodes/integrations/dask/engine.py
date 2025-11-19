"""Dask-based execution engine for HyperNodes.

This engine provides parallel execution using Dask Bag for map operations
while maintaining sequential execution for regular pipeline runs.

The engine automatically determines optimal parallelism based on:
- Number of items to process
- Available CPU cores
- Workload type (I/O, CPU, or mixed)
"""

import multiprocessing
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import dask
import dask.bag as db
import numpy as np

from ...callbacks import CallbackContext, PipelineCallback
from ...map_planner import MapPlanner
from ...node_execution import execute_single_node
from ...orchestrator import ExecutionOrchestrator

if TYPE_CHECKING:
    from ...cache import Cache
    from ...pipeline import Pipeline


class DaskEngine:
    """Dask-based execution engine with automatic optimization.

    This engine uses Dask Bag for parallel map operations while maintaining
    sequential execution for regular pipeline runs (to avoid overhead).

    Features:
    - Automatic npartitions calculation based on dataset size and workload
    - Configurable scheduler (threads, processes, synchronous)
    - Inherits configuration in nested pipelines
    - Zero overhead for non-map operations

    Args:
        scheduler: Dask scheduler to use ("threads", "processes", "synchronous")
            Default is "threads" which works best for mixed I/O/CPU workloads
        num_workers: Number of workers (defaults to CPU count)
        workload_type: "io", "cpu", or "mixed" - affects npartitions calculation
        npartitions: Manual override for number of partitions (None = auto)
        cache: Cache instance for caching results
        callbacks: List of callbacks for observability

    Example:
        >>> from hypernodes import Pipeline
        >>> from hypernodes.engines import DaskEngine
        >>>
        >>> # Auto-optimized for your workload
        >>> engine = DaskEngine()
        >>> pipeline = Pipeline(nodes=[...], engine=engine)
        >>>
        >>> # Regular run (sequential, no overhead)
        >>> result = pipeline.run(inputs={"x": 5})
        >>>
        >>> # Map operation (parallel via Dask Bag)
        >>> results = pipeline.map(
        ...     inputs={"x": [1, 2, 3, 4, 5]},
        ...     map_over="x"
        ... )
        >>>
        >>> # Custom configuration for CPU-bound workload
        >>> engine = DaskEngine(
        ...     scheduler="processes",
        ...     workload_type="cpu"
        ... )
    """

    def __init__(
        self,
        scheduler: str = "threads",
        num_workers: Optional[int] = None,
        workload_type: str = "mixed",
        npartitions: Optional[int] = None,
        cache: Optional["Cache"] = None,
        callbacks: Optional[List[PipelineCallback]] = None,
    ):
        """Initialize DaskEngine.

        Args:
            scheduler: Dask scheduler ("threads", "processes", "synchronous")
            num_workers: Number of workers (None = CPU count)
            workload_type: "io" (I/O bound), "cpu" (CPU bound), or "mixed"
            npartitions: Manual npartitions override (None = auto-calculate)
            cache: Cache instance for caching results
            callbacks: List of callbacks for observability
        """
        self.scheduler = scheduler
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.workload_type = workload_type
        self.npartitions = npartitions
        self.cache = cache
        self.callbacks = callbacks or []

    def run(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        output_name: Optional[Union[str, List[str]]] = None,
        _ctx: Optional[CallbackContext] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute pipeline sequentially (no parallelism for single runs).

        For single pipeline runs, we use sequential execution to avoid
        Dask overhead. Parallelism is only beneficial for map operations.

        Args:
            pipeline: Pipeline to execute
            inputs: Input values
            output_name: Optional output name(s) to filter results
            _ctx: Internal callback context (for nested pipelines)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary of output values
        """
        # Use orchestrator for lifecycle management
        with ExecutionOrchestrator(pipeline, self.callbacks, _ctx) as orchestrator:
            orchestrator.validate_callbacks("DaskEngine")
            orchestrator.notify_start(inputs)

            # Execute nodes sequentially (same as SeqEngine)
            available_values = dict(inputs)
            outputs = {}
            node_signatures = {}

            for node in pipeline.graph.execution_order:
                # Gather inputs for this node
                node_inputs = {
                    param: available_values[param] for param in node.root_args
                }

                # Execute the node
                result, signature = execute_single_node(
                    node,
                    node_inputs,
                    pipeline,
                    self.cache,
                    self.callbacks,
                    orchestrator.ctx,
                    node_signatures,
                )

                # Store outputs
                self._store_node_outputs(
                    node, result, outputs, available_values, node_signatures, signature
                )

            orchestrator.notify_end(outputs)

            # Filter outputs if requested
            if output_name:
                names = [output_name] if isinstance(output_name, str) else output_name
                return {k: outputs[k] for k in names}

            return outputs

    def map(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        map_over: Union[str, List[str]],
        map_mode: str = "zip",
        output_name: Optional[Union[str, List[str]]] = None,
        _ctx: Optional[CallbackContext] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Execute pipeline for each item using Dask Bag for parallelism.

        This is where Dask shines - parallel map operations across many items.
        We use Dask Bag which is optimized for this use case.

        Args:
            pipeline: Pipeline to execute
            inputs: Input values (lists for map_over params, scalars for fixed)
            map_over: Parameter name(s) to map over
            map_mode: "zip" (parallel iteration) or "product" (all combinations)
            output_name: Optional output name(s) to filter results
            _ctx: Internal callback context
            **kwargs: Additional arguments (ignored)

        Returns:
            List of output dictionaries, one per item
        """
        # Use orchestrator for lifecycle management
        with ExecutionOrchestrator(pipeline, self.callbacks, _ctx) as orchestrator:
            orchestrator.validate_callbacks("DaskEngine")

            # Normalize map_over to list
            map_over_list = [map_over] if isinstance(map_over, str) else map_over

            # Use MapPlanner to create items
            planner = MapPlanner()
            items = planner.plan_execution(inputs, map_over_list, map_mode)

            # Calculate optimal npartitions
            if self.npartitions is None:
                nparts = self._calculate_npartitions(len(items))
            else:
                nparts = self.npartitions

            # Notify map start
            orchestrator.notify_map_start(len(items))

            # Set map context
            ctx = orchestrator.ctx
            ctx.set("_in_map", True)
            ctx.set("_map_total_items", len(items))

            # Execute using Dask Bag for parallelism
            with dask.config.set(scheduler=self.scheduler):
                bag = db.from_sequence(items, npartitions=nparts)

                # Define the function that processes each item
                def process_item(item_inputs: Dict[str, Any]) -> Dict[str, Any]:
                    """Process a single item through the pipeline."""
                    return self.run(pipeline, item_inputs, output_name, _ctx=ctx)

                # Map over all items in parallel
                results = bag.map(process_item).compute()

            # Clear map context
            ctx.set("_in_map", False)

            orchestrator.notify_map_end()

            return list(results)

    def _calculate_npartitions(self, num_items: int) -> int:
        """Calculate optimal number of partitions using heuristic.

        Based on empirical analysis and Dask best practices:
        - I/O bound: 4x oversubscription (more parallelism helps)
        - CPU bound: 2x oversubscription (match core count)
        - Mixed: 3x oversubscription (moderate parallelism)
        - Aim for 10-1000 items per partition
        - Clamp to reasonable bounds

        Args:
            num_items: Number of items to process

        Returns:
            Recommended number of partitions
        """
        total_parallelism = self.num_workers

        # Heuristic 1: Based on workload type
        if self.workload_type == "io":
            parallelism_based = total_parallelism * 4
        elif self.workload_type == "cpu":
            parallelism_based = total_parallelism * 2
        else:  # mixed
            parallelism_based = total_parallelism * 3

        # Heuristic 2: Based on item count and granularity
        min_items_per_partition = 10
        max_items_per_partition = 1000

        granularity_based_min = max(1, num_items // max_items_per_partition)
        granularity_based_max = max(1, num_items // min_items_per_partition)

        # Heuristic 3: Avoid extremes
        min_partitions = max(2, total_parallelism)
        max_partitions = min(num_items, total_parallelism * 10)

        # Combine heuristics
        recommended = int(parallelism_based)

        # Clamp to reasonable bounds
        recommended = max(min_partitions, min(recommended, max_partitions))
        recommended = max(
            granularity_based_min, min(recommended, granularity_based_max)
        )

        # Prefer power of 2 for better distribution
        power_of_2 = 2 ** round(np.log2(recommended))
        if abs(power_of_2 - recommended) / recommended < 0.3:
            recommended = power_of_2

        return recommended

    def _store_node_outputs(
        self,
        node,
        result: Any,
        outputs: Dict[str, Any],
        available_values: Dict[str, Any],
        node_signatures: Dict[str, str],
        signature: str,
    ) -> None:
        """Store node outputs in results dicts.

        Handles both regular nodes and PipelineNodes (which return dicts).
        """
        if hasattr(node, "pipeline"):
            # PipelineNode - result is dict of outputs
            outputs.update(result)
            available_values.update(result)
            for output_name_key in result.keys():
                node_signatures[output_name_key] = signature
        else:
            # Regular node
            if isinstance(node.output_name, tuple):
                # Multiple outputs
                for name, val in zip(node.output_name, result):
                    outputs[name] = val
                    available_values[name] = val
                    node_signatures[name] = signature
            else:
                # Single output
                outputs[node.output_name] = result
                available_values[node.output_name] = result
                node_signatures[node.output_name] = signature
