"""Map operation planning and execution plan generation."""

import itertools
from typing import Any, Dict, List


class MapPlanner:
    """Plans execution strategies for map operations.

    Handles input validation, separation of varying/fixed parameters,
    and generation of execution plans for both 'zip' and 'product' modes.
    """

    def plan_execution(
        self,
        inputs: Dict[str, Any],
        map_over: List[str],
        map_mode: str,
    ) -> List[Dict[str, Any]]:
        """Plan execution by preparing inputs and building execution plans.

        Args:
            inputs: All input parameters
            map_over: List of parameters that vary across executions
            map_mode: "zip" or "product"

        Returns:
            List of input dictionaries, one per execution

        Raises:
            ValueError: If validation fails
        """
        varying_params, fixed_params, zip_lengths = self._prepare_inputs(
            inputs, map_over, map_mode
        )
        return self._build_plans(varying_params, fixed_params, map_mode, zip_lengths)

    def _prepare_inputs(
        self, inputs: Dict[str, Any], map_over: List[str], map_mode: str
    ) -> tuple[Dict[str, Any], Dict[str, Any], List[int]]:
        """Prepare inputs for map operation.

        Separates varying and fixed parameters, validates inputs.

        Args:
            inputs: All input parameters
            map_over: List of parameters that vary
            map_mode: "zip" or "product"

        Returns:
            Tuple of (varying_params, fixed_params, zip_lengths)

        Raises:
            ValueError: If validation fails
        """
        # Separate varying and fixed parameters
        varying_params = {}
        fixed_params = {}

        for key, value in inputs.items():
            if key in map_over:
                if not isinstance(value, list):
                    raise ValueError(
                        f"Parameter '{key}' is in map_over but value is not a list"
                    )
                varying_params[key] = value
            else:
                fixed_params[key] = value

        # Validate that all map_over parameters are present
        for param in map_over:
            if param not in varying_params:
                raise ValueError(f"Parameter '{param}' in map_over not found in inputs")

        # Validate zip mode lengths
        zip_lengths: List[int] = []
        if map_mode == "zip":
            zip_lengths = [len(lst) for lst in varying_params.values()]
            if zip_lengths and not all(
                length == zip_lengths[0] for length in zip_lengths
            ):
                length_info = {k: len(v) for k, v in varying_params.items()}
                raise ValueError(
                    f"In zip mode, all lists must have the same length. "
                    f"Got lengths: {length_info}"
                )

        return varying_params, fixed_params, zip_lengths

    def _build_plans(
        self,
        varying_params: Dict[str, Any],
        fixed_params: Dict[str, Any],
        map_mode: str,
        zip_lengths: List[int],
    ) -> List[Dict[str, Any]]:
        """Build list of execution plans based on map_mode.

        Args:
            varying_params: Parameters that vary across items
            fixed_params: Parameters that stay constant
            map_mode: "zip" or "product"
            zip_lengths: List lengths (for empty detection in zip mode)

        Returns:
            List of input dictionaries, one per execution
        """
        if map_mode == "zip":
            # Create execution plans by zipping
            if not varying_params or not zip_lengths or zip_lengths[0] == 0:
                # Empty case
                return []
            else:
                # Zip the varying parameters together
                param_names = list(varying_params.keys())
                param_lists = [varying_params[name] for name in param_names]
                return [
                    {**fixed_params, **{k: v for k, v in zip(param_names, values)}}
                    for values in zip(*param_lists)
                ]
        else:  # product mode
            # Create all combinations
            if not varying_params:
                return [fixed_params]
            else:
                param_names = list(varying_params.keys())
                param_lists = [varying_params[name] for name in param_names]
                return [
                    {**fixed_params, **{k: v for k, v in zip(param_names, values)}}
                    for values in itertools.product(*param_lists)
                ]
