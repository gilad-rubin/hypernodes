import inspect
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional

from hamilton.driver import Builder, Driver
from hypster.core import Hypster


def _remove_driver_config_keys(
    driver_config: Dict[str, Any], inputs: Dict[str, Any]
) -> Dict[str, Any]:
    return {k: v for k, v in inputs.items() if k not in driver_config}


def _get_all_possible_outputs(driver: Driver) -> List[str]:
    return [n for n in driver.list_available_variables() if not n.is_external_input]


def _get_upstream_args(driver: Driver, node_name: str) -> List[str]:
    node = driver.graph.nodes.get(node_name)
    if node is None:
        raise ValueError(f"Node {node_name} not found in driver graph")

    if node.originating_functions is None:
        original_func = _get_original_func(node.callable)
    else:
        original_func = node.originating_functions[0]

    upstream_args = _get_func_arg_list(original_func)
    return [arg for arg in upstream_args]


def _get_original_func(func: Callable) -> Callable:
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    return func


def _get_func_arg_list(func: Callable) -> List[str]:
    signature = inspect.signature(func)
    return list(signature.parameters.keys())


class HyperNode:
    def __init__(
        self,
        name: str,
        hamilton_dags: List[ModuleType] = None,
        hypster_config: Optional[Hypster] = None,
        builder_param_name: str = "builder",
    ):
        self.name = name
        self.hamilton_dags = hamilton_dags or []
        self.hypster_config = hypster_config
        self.builder_param_name = builder_param_name or "builder"
        self._instantiated_config = None
        self._driver: Driver = None

    def _check_hamilton_dags(self):
        if len(self.hamilton_dags) == 0:
            raise ValueError(f"No Hamilton DAGs have been added to node {self.name}.")

    def _check_driver_initialized(self):
        if self._driver is None:
            raise RuntimeError("Driver not initialized, call instantiate() first.")

    def instantiate(self, selections: Dict[str, Any] = {}, overrides: Dict[str, Any] = {}) -> None:
        self._instantiate_config(selections, overrides)
        self._init_driver()

    def _instantiate_config(self, selections: Dict[str, Any] = {}, overrides: Dict[str, Any] = {}):
        if self.hypster_config is None and (selections or overrides):
            raise ValueError(f"No hypster config found for node {self.name}.\
                             Please add a hypster config or the remove selections and overrides.")

        self._instantiated_config = self.hypster_config(
            selections=selections,
            overrides=overrides,
        )

    def _init_driver(self) -> None:
        self._check_hamilton_dags()

        if self._instantiated_config is None:
            raise ValueError("You must instantiate inputs before initializing the driver")

        builder = self._instantiated_config.get(self.builder_param_name, Builder()).copy()
        self._driver = builder.with_modules(*self.hamilton_dags).build()

    def execute(
        self, final_vars: List[Any] = [], additional_inputs: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        self._check_driver_initialized()

        inputs = {**self._instantiated_config, **additional_inputs}
        inputs = _remove_driver_config_keys(self._driver.config, inputs)

        if len(final_vars) == 0:
            final_vars = _get_all_possible_outputs(self._driver)

        return self._driver.execute(final_vars=final_vars, inputs=inputs)

    def get_hamilton_node_inputs(self, node_name: str) -> Dict[str, Any]:
        self._check_hamilton_dags()
        self._check_driver_initialized()

        upstream_args = _get_upstream_args(self._driver, node_name)
        return self.execute(final_vars=upstream_args)

    def set_instantiated_config(self, config: Dict[str, Any]) -> None:
        """
        Set the instantiated config directly and initialize the driver.

        Args:
            config (Dict[str, Any]): The instantiated configuration to set.
        """
        self._instantiated_config = config
        self._init_driver()
