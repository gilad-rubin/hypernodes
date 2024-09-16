import inspect
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional

from hamilton.driver import Builder, Driver
from hypster.core import Hypster


class HyperNode:
    def __init__(
        self,
        name: str,
        hamilton_dags: List[ModuleType] = None,
        hypster_config: Optional[Hypster] = None,
    ):
        self.name = name
        self.hamilton_dags = hamilton_dags or []
        self.hypster_config = hypster_config
        self._instantiated_inputs: Optional[Dict[str, Any]] = None
        self._driver: Driver = None

    def _check_hamilton_dags(self):
        if not self.hamilton_dags:
            raise ValueError(
                f"No Hamilton DAGs have been added to node {self.name}."
            )

    def instantiate_inputs(
        self,
        selections: Dict[str, Any] = {},
        overrides: Dict[str, Any] = {},
        return_config_snapshot=False,
    ) -> None:
        self._check_hamilton_dags()
        if self.hypster_config is None:
            self._instantiated_inputs = {}
        else:
            if return_config_snapshot:
                self._instantiated_inputs, snapshot = self.hypster_config(
                    selections=selections,
                    overrides=overrides,
                    return_config_snapshot=True,
                )
                self._instantiated_inputs.update(snapshot)
            else:
                self._instantiated_inputs = self.hypster_config(
                    selections=selections,
                    overrides=overrides,
                    return_config_snapshot=False,
                )

    def init_driver(
        self
    ) -> None:  # TODO: make it instant after instantiated_inputs is set?
        self._check_hamilton_dags()
        if self._driver is not None:
            return

        if self._instantiated_inputs is None:
            raise ValueError(
                "You must instantiate inputs before initializing the driver"
            )

        builder = self._instantiated_inputs.get("builder", Builder()).copy()

        try:
            self._driver = builder.with_modules(*self.hamilton_dags).build()
        except Exception as e:
            print(f"Failed to build driver: {e}")
            raise RuntimeError(f"Failed to build driver: {e}")

    def ensure_driver_initialized(self) -> None:
        self._check_hamilton_dags()
        if self._driver is None:
            try:
                self.init_driver()
            except Exception as e:
                print(f"Failed to initialize driver: {e}")
                raise RuntimeError(f"Failed to initialize driver: {e}")

    def execute(
        self, final_vars: List[Any] = [], inputs: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        self._check_hamilton_dags()
        self.ensure_driver_initialized()

        if self._driver is None:
            raise RuntimeError("Driver initialization failed")

        inputs = inputs.copy()
        {k: inputs.pop(k) for k in self._driver.config}

        if len(final_vars) == 0:
            final_vars = [
                n
                for n in self._driver.list_available_variables()
                if not n.is_external_input
            ]

        if len(inputs) == 0:
            inputs = self._instantiated_inputs

        return self._driver.execute(final_vars=final_vars, inputs=inputs)

    def get_node_inputs(self, node_name: str) -> Dict[str, Any]:
        self._check_hamilton_dags()
        self.ensure_driver_initialized()

        if self._driver is None:
            raise RuntimeError("Driver initialization failed")

        upstream_args = get_upstream_args(self._driver, node_name)
        return self.execute(
            final_vars=upstream_args, inputs=self._instantiated_inputs
        )


def get_upstream_args(driver: Driver, node_name: str) -> List[str]:
    node = driver.graph.nodes.get(node_name)
    if node is None:
        raise ValueError(f"Node {node_name} not found in driver graph")

    if node.originating_functions is None:
        original_func = get_original_func(node.callable)
    else:
        original_func = node.originating_functions[0]

    upstream_args = get_func_arg_list(original_func)
    return [arg for arg in upstream_args if not arg.startswith(node_name)]


def get_original_func(func: Callable) -> Callable:
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    return func


def get_func_arg_list(func: Callable) -> List[str]:
    signature = inspect.signature(func)
    return list(signature.parameters.keys())
