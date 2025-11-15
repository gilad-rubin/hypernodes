from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Union

from .hypernode import HyperNode

if TYPE_CHECKING:
    from .pipeline import Pipeline


class PipelineNode(HyperNode):
    """Wraps a Pipeline to behave like a Node with custom input/output mapping.

    This class adapts a Pipeline interface to work as a node in another pipeline,
    with support for parameter renaming and internal mapping.

    Implements HyperNode interface to ensure compatibility when used in Pipeline.nodes lists.

    Attributes:
        pipeline: The wrapped pipeline
        input_mapping: Maps outer parameter names to inner pipeline parameters
        output_mapping: Maps inner pipeline outputs to outer names
        map_over: Optional parameter name(s) to map over (from outer perspective)
        map_mode: Mode for map operation ("zip" or "product")
        cache: Whether to cache the node. Not to be confused with the pipeline's cache.
        name: Optional name for this node (displayed in visualizations)
    """

    def __init__(
        self,
        pipeline: "Pipeline",
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        map_over: Optional[Union[str, List[str]]] = None,
        map_mode: Literal["zip", "product"] = "zip",
        cache: bool = True,  # Whether to cache the node. Not to be confused with the pipeline's cache.
        name: Optional[str] = None,
    ):
        """Initialize a PipelineNode wrapper.

        Args:
            pipeline: Pipeline to wrap
            input_mapping: Maps {outer_name: inner_name} for inputs
            output_mapping: Maps {inner_name: outer_name} for outputs
            map_over: Parameter name(s) to map over (from outer perspective)
            map_mode: Mode for map operation ("zip" or "product")
            cache: Whether to cache the node. Not to be confused with the pipeline's cache.
            name: Optional name for this node (displayed in visualizations)
        """
        self._pipeline = pipeline
        self.input_mapping = input_mapping or {}
        self.output_mapping = output_mapping or {}
        self.map_mode = map_mode
        self.map_over = [map_over] if isinstance(map_over, str) else map_over

        self.cache = cache
        self.name = name if name is not None else pipeline.name

        # Pre-compute and cache aggregated code hash to avoid expensive recomputation
        # This aggregates hashes from all inner nodes (including nested pipelines)
        from .cache import compute_pipeline_code_hash

        self._code_hash = compute_pipeline_code_hash(pipeline)

    @property
    def root_args(self) -> tuple:
        """Get outer parameter names (after applying reverse input mapping).

        Returns:
            Tuple of parameter names from outer pipeline's perspective
        """
        # Special case: map_over without input_mapping exposes only map_over params
        if self.map_over and not self.input_mapping:
            return tuple(self.map_over)

        # Get inner pipeline's root parameters and apply reverse mapping
        inner_params = self._pipeline.graph.root_args
        reverse_mapping = self._create_reverse_input_mapping()
        outer_params = [reverse_mapping.get(p, p) for p in inner_params]

        # Add any map_over params that aren't already included
        if self.map_over:
            for param in self.map_over:
                if param not in outer_params:
                    outer_params.append(param)

        return tuple(outer_params)

    @property
    def output_name(self) -> Union[str, tuple]:
        """Get outer output names (after output mapping).

        Returns all possible outputs. The actual required outputs are determined
        at the parent pipeline level (stored in pipeline.graph.required_outputs).

        Returns:
            Output name(s) from outer pipeline's perspective
        """
        # Get inner pipeline outputs and apply output mapping
        inner_outputs = self._pipeline.graph.available_output_names
        outer_outputs = self._map_names(inner_outputs, self.output_mapping)

        # Return single string or tuple (matching Node convention)
        if len(outer_outputs) == 1:
            return outer_outputs[0]
        return tuple(outer_outputs)

    @property
    def code_hash(self) -> str:
        """Get cached aggregated code hash for this pipeline node.

        The hash is computed once at node creation by aggregating all inner
        node hashes. This avoids expensive recomputation on every execution.
        The cached value persists through pickling/unpickling.

        Returns:
            SHA256 hash of all inner node code
        """
        return self._code_hash

    @property
    def pipeline(self) -> "Pipeline":
        """Get the wrapped pipeline.

        Returns:
            The Pipeline instance wrapped by this PipelineNode
        """
        return self._pipeline

    def _create_reverse_input_mapping(self) -> Dict[str, str]:
        """Create reverse mapping from inner to outer parameter names.

        Returns:
            Dictionary mapping inner names to outer names
        """
        return {inner: outer for outer, inner in self.input_mapping.items()}

    def _apply_input_mapping(self, inputs: Dict) -> Dict:
        """Map outer input names to inner parameter names.

        Args:
            inputs: Input parameters from outer perspective

        Returns:
            Mapped inputs ready for inner pipeline
        """
        inner_inputs = {}
        for outer_name, value in inputs.items():
            inner_name = self.input_mapping.get(outer_name, outer_name)
            inner_inputs[inner_name] = value
        return inner_inputs

    def _translate_map_over_to_inner(self) -> Union[str, List[str]]:
        """Translate map_over parameters from outer to inner names.

        Returns:
            Inner parameter name(s) to map over
        """
        if not self.map_over:
            return None

        map_over_inner = [
            self.input_mapping.get(outer_param, outer_param)
            for outer_param in self.map_over
        ]
        # Return as list if multiple, otherwise return single string
        return map_over_inner if len(map_over_inner) > 1 else map_over_inner[0]

    def _map_names(self, names: List[str], mapping: Dict[str, str]) -> List[str]:
        """Apply a name mapping to a list of names.

        Args:
            names: List of names to map
            mapping: Dictionary mapping original names to new names

        Returns:
            List of mapped names
        """
        return [mapping.get(name, name) for name in names]

    def _apply_output_mapping(self, result: Dict) -> Dict:
        """Map inner output names to outer names for a single result.

        Args:
            result: Output dictionary from inner pipeline

        Returns:
            Mapped output ready for outer perspective
        """
        outer_result = {}
        for inner_name, value in result.items():
            outer_name = self.output_mapping.get(inner_name, inner_name)
            outer_result[outer_name] = value
        return outer_result

    def _collect_mapped_results(self, results: List[Dict]) -> Dict[str, List]:
        """Collect and map results from pipeline.map() into outer format.

        Args:
            results: List of result dictionaries from inner pipeline

        Returns:
            Dictionary with outer names mapping to lists of values
        """
        outer_results: Dict[str, List] = {}
        for result_dict in results:
            mapped_result = self._apply_output_mapping(result_dict)
            for outer_name, value in mapped_result.items():
                if outer_name not in outer_results:
                    outer_results[outer_name] = []
                outer_results[outer_name].append(value)
        return outer_results

    def _get_required_inner_outputs(
        self, required_outputs: Optional[List[str]] = None
    ) -> Optional[Union[str, List[str]]]:
        """Determine which inner outputs to request based on required_outputs parameter.

        Args:
            required_outputs: List of required outer output names, or None for all

        Returns:
            Inner output name(s) to pass to inner pipeline, or None for all outputs
        """
        if required_outputs is None:
            # No pruning - request all outputs
            return None

        # Map outer required outputs back to inner output names
        reverse_output_mapping = {
            outer: inner for inner, outer in self.output_mapping.items()
        }

        inner_outputs = []
        for outer_name in required_outputs:
            # Get the inner name (reverse the output mapping)
            inner_name = reverse_output_mapping.get(outer_name, outer_name)
            inner_outputs.append(inner_name)

        # Return as string if single output, list if multiple, None if empty
        if len(inner_outputs) == 0:
            return None
        elif len(inner_outputs) == 1:
            return inner_outputs[0]
        else:
            return inner_outputs

    def __call__(self, required_outputs: Optional[List[str]] = None, **inputs):
        """Execute the wrapped pipeline with input/output mapping.

        Args:
            required_outputs: Optional list of required outer output names.
                            If provided, only these outputs will be computed.
            **inputs: Input parameters from outer perspective

        Returns:
            Dictionary of outputs (with output_mapping applied)
        """
        inner_inputs = self._apply_input_mapping(inputs)

        # Determine which inner outputs to request based on required_outputs
        # This enables the inner pipeline to skip unnecessary computation
        inner_output_name = self._get_required_inner_outputs(required_outputs)

        if self.map_over:
            map_over_inner = self._translate_map_over_to_inner()
            results = self._pipeline.map(
                inputs=inner_inputs,
                map_over=map_over_inner,
                map_mode=self.map_mode,
                output_name=inner_output_name,
            )
            return self._collect_mapped_results(results)
        else:
            result = self._pipeline.run(
                inputs=inner_inputs,
                output_name=inner_output_name,
            )
            return self._apply_output_mapping(result)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PipelineNode({self._pipeline})"

    def __hash__(self) -> int:
        """Make PipelineNode hashable."""
        return hash(id(self))

    def __eq__(self, other) -> bool:
        """Check equality based on identity."""
        return self is other

    def __getstate__(self):
        """Custom pickle support to preserve code hash cache."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Custom unpickle support to restore code hash cache."""
        self.__dict__.update(state)
        # _code_hash is preserved through pickling
