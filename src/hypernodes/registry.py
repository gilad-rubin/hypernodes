import logging
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

import yaml

from . import mock_dag
from .hypernode import HyperNode


def load_hypernodes_config():
    config_path = "hypernodes.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    return {}


hypernodes_config = load_hypernodes_config()
DEFAULT_REGISTRY_PATH = hypernodes_config.get("node_registry_path", "conf/node_registry.yaml")
DEFAULT_FOLDER_TEMPLATE = hypernodes_config.get(
    "node_folder_template", "src/nodes/{node_name}/modules"
)

logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """
    Represents a node in the registry.

    Attributes:
        name (str): The name of the node.
        folder (str): The folder path for the node.
        hamilton_dag_paths (List[str]): List of Hamilton DAG paths associated with the node.
        hypster_config_path (Optional[str]): Path to the Hypster configuration file.
        builder_param_name (Optional[str]): Name of the Builder for the hamilton driver
          in the Hypster config.
    """

    name: str
    folder: str
    hamilton_dag_paths: List[str] = field(default_factory=list)
    hypster_config_path: Optional[str] = None
    builder_param_name: Optional[str] = None

    @classmethod
    def get_default_value(cls, field_name: str) -> Any:
        """
        Get the default value for a given field.

        Args:
            field_name (str): Name of the field.

        Returns:
            Any: Default value of the field, or None if not found.
        """
        for f in fields(cls):
            if f.name == field_name:
                return f.default
        return None

    def to_dict(self) -> dict:
        """
        Convert the Node instance to a dictionary.

        Returns:
            dict: Dictionary representation of the Node, excluding 'name' and default values.
        """
        return {
            k: v for k, v in asdict(self).items() if k != "name" and v != self.get_default_value(k)
        }

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "NodeInfo":
        """
        Create a Node instance from a dictionary.

        Args:
            name (str): Name of the node.
            data (dict): Dictionary containing node data.

        Returns:
            Node: A new Node instance.
        """
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(name=name, **filtered_data)


class StoreHandler:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self.file_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}

    def save(self, data: Dict[str, Dict[str, Any]]) -> None:
        with open(self.file_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


class NodeRegistry:
    """
    Manages a registry of nodes, including creation, retrieval, and persistence.
    """

    def __init__(self, store_handler: StoreHandler, folder_template: str):
        """
        Initialize the NodeRegistry.

        Args:
            store_handler (StoreHandler): Handler for loading and saving node data.
            folder_template (str): Template for node folder paths.
        """
        self.store_handler = store_handler
        self.folder_template = folder_template
        self.nodes = self._load_nodes()

    def _load_nodes(self) -> Dict[str, NodeInfo]:
        """
        Load nodes from the store handler.

        Returns:
            Dict[str, Node]: Dictionary of node names to Node instances.
        """
        data = self.store_handler.load()
        nodes = {}
        for name, node_data in data.items():
            nodes[name] = NodeInfo.from_dict(name, node_data)
        return nodes

    def _save_nodes(self) -> None:
        """
        Save all nodes to the store handler.
        """
        data = {}
        for node in self.nodes.values():
            data[node.name] = node.to_dict()
        self.store_handler.save(data)

    def list_nodes(
        self,
        require_hamilton_dags: bool = True,
        require_hypster_config: bool = False,
    ) -> List[str]:
        """
        List names of nodes meeting specified criteria.

        Args:
            require_hamilton_dags (bool): If True, only include nodes with Hamilton DAGs.
            require_hypster_config (bool): If True, only include nodes with a Hypster configuration.

        Returns:
            List[str]: List of node names meeting the criteria.
        """
        return [
            name
            for name, node in self.nodes.items()
            if (not require_hypster_config or node.hypster_config_path)
            and (not require_hamilton_dags or node.hamilton_dag_paths)
        ]

    def create_or_get(
        self, node_name: str, folder: Optional[str] = None, overwrite: bool = False
    ) -> "NodeHandler":
        """
        Create a new node or get an existing one.

        Args:
            node_name (str): Name of the node.
            folder (Optional[str]): Folder path for the node.
            overwrite (bool): If True, overwrite existing node's folder.

        Returns:
            NodeHandler: Created or retrieved NodeHandler instance.
        """
        node_info = self._create_or_get_node_info(node_name, folder, overwrite)
        from .node_handler import NodeHandler

        return NodeHandler(node_info, self)

    def load(self, node_name: str) -> "HyperNode":
        """
        Load a HyperNode instance for the given node name.

        Args:
            node_name (str): Name of the node to load.

        Returns:
            HyperNode: A read-only HyperNode instance.

        Raises:
            ValueError: If the node is not found in the registry.
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in registry")

        node_info = self.nodes[node_name]
        from .node_handler import NodeHandler

        node_handler = NodeHandler(node_info, self)
        return node_handler.to_hypernode()

    def _create_or_get_node_info(
        self, node_name: str, folder: Optional[str] = None, overwrite: bool = False
    ) -> NodeInfo:
        if node_name in self.nodes:
            return self._handle_existing_node(node_name, folder, overwrite)
        return self._create_new_node(node_name, folder)

    def _handle_existing_node(
        self, node_name: str, folder: Optional[str], overwrite: bool
    ) -> NodeInfo:
        """
        Handle an existing node when creating or getting a node.

        Args:
            node_name (str): Name of the node.
            folder (Optional[str]): Folder path for the node.
            overwrite (bool): If True, overwrite existing node's folder.

        Returns:
            Node: Existing Node instance, potentially updated.

        Raises:
            ValueError: If node exists in a different folder and overwrite is False.
        """
        existing_node = self.nodes[node_name]
        if folder and existing_node.folder != folder:
            if not overwrite:
                raise ValueError(
                    f"Node '{node_name}' already exists with a different folder. "
                    f"Existing: {existing_node.folder}, Requested: {folder}. "
                    f"Use overwrite=True to update the node."
                )
            logger.info(f"Updating existing node '{node_name}' with new folder: {folder}")
            existing_node.folder = folder
            self._save_nodes()
        logger.info(f'Loaded existing node "{node_name}" from {existing_node.folder}')
        return existing_node

    def _create_new_node(self, node_name: str, folder: Optional[str]) -> NodeInfo:
        """
        Create a new node.

        Args:
            node_name (str): Name of the node.
            folder (Optional[str]): Folder path for the node.

        Returns:
            Node: Newly created Node instance.
        """
        folder = folder or self.folder_template.format(node_name=node_name)
        new_node = NodeInfo(name=node_name, folder=folder)
        self.nodes[node_name] = new_node
        self._save_nodes()
        logger.info(f'Created new node "{node_name}" with folder {folder}')
        return new_node

    def mock(self, node_name: str) -> HyperNode:
        from hypster import HP, config

        @config
        def mock_config(hp: HP):
            mock_param = hp.select([1, 2], default=1)

        return HyperNode(node_name, [mock_dag], mock_config)

    def set_hypster_config_for_node(
        self, node_name: str, config_path: str, builder_param_name: Optional[str] = None
    ) -> None:
        """
        Set the Hypster configuration for a node.

        Args:
            node_name (str): Name of the node.
            config_path (str): Path to the Hypster configuration file.
            builder_param_name (Optional[str]): Name of the builder parameter.

        Raises:
            ValueError: If the node is not found in the registry.
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in registry")

        node = self.nodes[node_name]
        node.hypster_config_path = config_path
        node.builder_param_name = builder_param_name
        self._save_nodes()

    def add_dag_to_node(self, node_name: str, dag_path: str) -> None:
        """
        Add a Hamilton DAG to a node.

        Args:
            node_name (str): Name of the node.
            dag_path (str): Path to the Hamilton DAG file.

        Raises:
            ValueError: If the node is not found in the registry.
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in registry")

        node = self.nodes[node_name]
        if dag_path not in node.hamilton_dag_paths:
            node.hamilton_dag_paths.append(dag_path)
        self._save_nodes()

    def update_node(self, node_name: str, node: NodeInfo) -> None:
        """
        Update a node in the registry.

        Args:
            node_name (str): Name of the node.
            node (Node): Updated Node instance.

        Raises:
            ValueError: If the node is not found in the registry.
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in registry")

        self.nodes[node_name] = node
        self._save_nodes()

    def delete(self, node_name: str) -> None:
        """
        Delete a node from the registry.

        Args:
            node_name (str): Name of the node to delete.

        Raises:
            ValueError: If the node is not found in the registry.
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in registry")

        del self.nodes[node_name]
        self._save_nodes()
        logger.info(f"Deleted node '{node_name}' from registry")


def create_registry(
    registry_path: str = DEFAULT_REGISTRY_PATH, folder_template: str = DEFAULT_FOLDER_TEMPLATE
) -> NodeRegistry:
    """
    Create or get a NodeRegistry instance with default or custom settings.

    Args:
        store_path (str): Path to the YAML file for storing node data.
        folder_template (str): Template for node folder paths.

    Returns:
        NodeRegistry: Initialized NodeRegistry instance.
    """
    directory = os.path.dirname(registry_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    store_handler = StoreHandler(registry_path)
    return NodeRegistry(store_handler, folder_template)


# Add this at the end of the file
registry = create_registry()
