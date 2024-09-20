import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from hypster import HP, config

from hypernodes import HyperNode
from hypernodes.node_handler import NodeHandler
from hypernodes.registry import NodeInfo, NodeRegistry, StoreHandler, create_registry


@pytest.fixture
def temp_store_path():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as temp_file:
        yield temp_file.name
    os.unlink(temp_file.name)


@pytest.fixture
def node_registry(temp_store_path):
    print(temp_store_path)
    return create_registry(temp_store_path)


# @pytest.fixture
# def node_handler(node_registry):
#     node_info = NodeInfo(name="test_node", folder="src/nodes/test_node/modules")
#     return NodeHandler(node_info, node_registry)


def test_create_or_get_node(node_registry):
    node = node_registry.create_or_get("test_node")
    assert node.name == "test_node"


def test_create_or_get_existing_node(node_registry):
    node1 = node_registry.create_or_get("test_node")
    node2 = node_registry.create_or_get("test_node")
    assert node1.node_info is node2.node_info


def test_create_or_get_node_with_custom_folder(node_registry):
    node = node_registry.create_or_get("test_node", folder="custom/folder")
    assert node.node_info.folder == "custom/folder"


def test_set_hypster_config_for_node(node_registry):
    node = node_registry.create_or_get("test_node")
    node_registry.set_hypster_config_for_node("test_node", "config.py", "builder")
    assert node.node_info.hypster_config_path == "config.py"
    assert node.node_info.builder_param_name == "builder"


def test_add_dag_to_node(node_registry):
    node = node_registry.create_or_get("test_node")
    node_registry.add_dag_to_node("test_node", "dag.py")
    assert "dag.py" in node.node_info.hamilton_dag_paths


def test_create_or_get_node_with_different_folder_fails(node_registry):
    node_registry.create_or_get("test_node")
    with pytest.raises(ValueError):
        node_registry.create_or_get("test_node", folder="different/folder", overwrite=False)


def test_list_nodes(node_registry):
    node_registry.create_or_get("node1")
    node_registry.create_or_get("node2")
    node_registry.add_dag_to_node("node1", "dag1.py")
    node_registry.set_hypster_config_for_node("node2", "config2.py")

    assert set(
        node_registry.list_nodes(require_hamilton_dags=True, require_hypster_config=False)
    ) == {"node1"}
    assert set(
        node_registry.list_nodes(require_hamilton_dags=False, require_hypster_config=True)
    ) == {"node2"}


@pytest.mark.parametrize(
    "store_path,folder_template",
    [
        ("test_store.yaml", "custom/{node_name}"),
        ("another_store.yaml", "custom2/{node_name}"),
    ],
)
def test_create_registry(store_path, folder_template):
    registry = create_registry(store_path, folder_template)
    assert isinstance(registry, NodeRegistry)
    assert registry.store_handler.file_path == store_path
    assert registry.folder_template == (folder_template or "src/nodes/{node_name}/modules")
