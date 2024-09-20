import importlib
import inspect
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import List, Optional

import hypster
from hypster.core import Hypster

from .hypernode import HyperNode
from .registry import NodeInfo, NodeRegistry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def write_module_to_file(module: ModuleType, file_path: Path) -> None:
    """
    Write a module's source code to a file.

    Args:
        module (ModuleType): The module to write.
        file_path (Path): The path where the module should be written.

    Raises:
        IOError: If there's an error writing the file.
    """
    try:
        file_path.write_text(inspect.getsource(module))
    except IOError as e:
        logger.error(f"Failed to write module to file {file_path}: {e}")
        raise


def import_module_from_path(module_path: Path) -> Optional[ModuleType]:
    """
    Import a module from a given path.

    Args:
        module_path (Path): Path to the module file.

    Returns:
        Optional[ModuleType]: Imported module or None if import fails.
    """
    folder = module_path.parent
    module_name = module_path.stem

    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        logger.error(f"Failed to import {module_name}. Error: {e}")
        logger.debug(f"sys.path: {sys.path}")
        logger.debug(f"Available modules: {list(sys.modules.keys())}")
        return None


class NodeHandler:
    """
    Handles the management of a single node, including its Hamilton DAGs and Hypster configuration.
    """

    def __init__(self, node_info: NodeInfo, registry: NodeRegistry):
        """
        Initialize a NodeHandler instance.

        Args:
            node_info (NodeInfo): The NodeInfo instance to handle.
            registry (NodeRegistry): The parent NodeRegistry instance.
        """
        self.node_info = node_info
        self.name = node_info.name
        self.registry = registry
        self.hamilton_dags: List[ModuleType] = []
        self.hypster_config: Optional[Hypster] = None
        self._load_existing_data()

    def _load_existing_data(self) -> None:
        """Load existing DAGs and Hypster config."""
        self.load_dags()
        self.load_hypster()

    def load_dags(self) -> None:
        """Load Hamilton DAGs for this node from the registry."""
        self.hamilton_dags = []
        for dag_path in self.node_info.hamilton_dag_paths:
            dag_path = Path(dag_path)
            if dag_path.exists():
                module = import_module_from_path(dag_path)
                if module:
                    self.hamilton_dags.append(module)
            else:
                logger.warning(f"DAG file not found: {dag_path}")

    def load_hypster(self) -> None:
        """Load the Hypster configuration for this node."""
        hypster_config_path = self.node_info.hypster_config_path
        if hypster_config_path:
            config_path = Path(hypster_config_path)
            if config_path.exists():
                self.hypster_config = hypster.load(str(config_path))
            else:
                logger.warning(f"Hypster config file not found: {config_path}")
                self.hypster_config = None
        else:
            self.hypster_config = None

    def save_dag(self, module: ModuleType, overwrite: bool = True) -> None:
        """
        Save a Hamilton DAG module to file and update the registry.

        Args:
            module (ModuleType): The module to save.
            overwrite (bool): Whether to overwrite existing file. Defaults to True.

        Raises:
            ValueError: If the DAG already exists and overwrite is False.
            IOError: If there's an error writing the file.
        """
        module_name = module.__name__
        file_name = f"{self.name}_{module_name}.py"
        module_path = Path(self.node_info.folder) / file_name

        if not overwrite and module_path.exists():
            raise ValueError(
                f"DAG '{module_name}' already exists for node '{self.name}'. "
                "Use overwrite=True to replace it."
            )

        write_module_to_file(module, module_path)
        self._update_hamilton_dags(module, str(module_path))

        logger.info(f'`{module_name}` is saved as a Hamilton DAG for "{self.name}"')

    def _update_hamilton_dags(self, module: ModuleType, module_path: str) -> None:
        """
        Update the list of Hamilton DAGs and the registry.

        Args:
            module (ModuleType): The module to update.
            module_path (str): The path where the module is saved.
        """
        self.hamilton_dags = [dag for dag in self.hamilton_dags if dag.__name__ != module.__name__]
        self.hamilton_dags.append(module)
        if module_path not in self.node_info.hamilton_dag_paths:
            self.node_info.hamilton_dag_paths.append(module_path)
        self.registry.update_node(self.name, self.node_info)

    def save_hypster_config(
        self, hypster_config: Hypster, builder_param_name: str = "builder", overwrite: bool = True
    ) -> None:
        """
        Save the Hypster configuration for this node.

        Args:
            hypster_config (Hypster): The Hypster configuration to save.
            builder_param_name (str): The name of the builder parameter. Defaults to "builder".
            overwrite (bool): Whether to overwrite an existing configuration. Defaults to True.

        Raises:
            ValueError: If the configuration already exists and overwrite is False.
        """
        config_path = Path(self.node_info.folder) / f"{self.name}_hypster_config.py"

        if not overwrite and config_path.exists():
            raise ValueError(
                f"Hypster config already exists for node '{self.name}'. "
                "Use overwrite=True to replace it."
            )

        self.hypster_config = hypster_config
        self.node_info.builder_param_name = builder_param_name
        hypster.save(hypster_config, str(config_path))
        self.node_info.hypster_config_path = str(config_path)
        self.registry.update_node(self.name, self.node_info)

        logger.info(f"`{hypster_config.name}` is saved as a hypster config for {self.name}")

    def save(
        self,
        hamilton_dags: List[ModuleType],
        hypster_config: Optional[Hypster] = None,
        overwrite: bool = True,
    ) -> None:
        """
        Save Hamilton DAGs and optionally a Hypster configuration for this node.

        Args:
            hamilton_dags (List[ModuleType]): List of Hamilton DAG modules to save.
            hypster_config (Optional[Hypster]): Hypster configuration to save. Defaults to None.
            overwrite (bool): Whether to overwrite existing files. Defaults to True.
        """
        for module in hamilton_dags:
            self.save_dag(module, overwrite)
        if hypster_config:
            self.save_hypster_config(hypster_config, overwrite=overwrite)
        logger.info(
            f'Saved {len(hamilton_dags)} DAG{"s" if len(hamilton_dags) > 1 else ""}'
            f'{" and hypster config" if hypster_config else ""} to node "{self.name}"'
        )
        self.registry.update_node(self.name, self.node_info)

    def to_hypernode(self) -> HyperNode:
        """
        Convert this NodeHandler to a HyperNode.

        Returns:
            HyperNode: A HyperNode instance representing this node.
        """
        return HyperNode(
            name=self.name,
            hamilton_dags=self.hamilton_dags,
            hypster_config=self.hypster_config,
            builder_param_name=self.node_info.builder_param_name,
        )
