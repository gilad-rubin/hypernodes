import importlib
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Union

import hypster
import tomli
import yaml
from hypster.core import Hypster

from .hypernode import HyperNode


class NodeHandler:
    def __init__(self, name: str, folder: str, registry: "NodeRegistry"):
        self.name = name
        self.folder = Path(folder)
        self.registry = registry
        self.hamilton_dags: List[ModuleType] = []
        self.hypster_config: Optional[Hypster] = None
        self._load_existing_data()

    def _load_existing_data(self):
        self.load_dags()
        self.load_hypster()

    def save_dag(self, module: ModuleType, overwrite: bool = True):
        module_name = module.__name__
        module_path = self.folder / f"{self.name}_{module_name}.py"
        
        if not overwrite and module_path.exists():
            raise ValueError(f"DAG '{module_name}' already exists for node '{self.name}'. Use overwrite=True to replace it.")

        with open(module_path, "w") as f:
            f.write(inspect.getsource(module))

        # Update the registry with the new DAG path
        # if the module is in the list of Hamilton DAGs, remove it first:
        self.hamilton_dags = [dag for dag in self.hamilton_dags if dag.__name__ != f"{self.name}_{module_name}"]
        self.hamilton_dags.append(module)
        self.registry.add_dag_to_node(self.name, str(module_path))

        print(f'"{module_name}" is saved as a Hamilton DAG for "{self.name}"')

    def save_hypster_config(self, hypster_config: Hypster, overwrite: bool = True):
        config_path = self.folder / f"{self.name}_hypster_config.py"
        
        if not overwrite and config_path.exists():
            raise ValueError(f"Hypster config already exists for node '{self.name}'. Use overwrite=True to replace it.")

        hypster.save(hypster_config, str(config_path))
        self.hypster_config = hypster_config
        self.registry.set_hypster_config_for_node(self.name, str(config_path))

        print(f'Hypster config is saved for "{self.name}"')
    
    def load_dags(self):
        self.hamilton_dags = []
        node_info = self.registry.nodes.get(self.name, {})
        dag_paths = node_info.get("hamilton_dags", [])

        # Ensure the folder containing DAGs is in sys.path
        if str(self.folder) not in sys.path:
            sys.path.insert(0, str(self.folder))

        for dag_path in dag_paths:
            dag_path = Path(dag_path)
            if dag_path.exists():
                # Convert file path to module path
                module_name = dag_path.stem

                try:
                    # Import the module
                    module = importlib.import_module(module_name)
                    self.hamilton_dags.append(module)
                except ImportError as e:
                    print(f"Failed to import {module_name}. Error: {e}")
                    print("sys.path:", sys.path)
                    print("Available modules:", list(sys.modules.keys()))
                    raise
            else:
                print(f"Warning: DAG file not found: {dag_path}")

    def save_hypster(self, hypster_config: Hypster, overwrite: bool = True):
        hypster_config_path = self.folder / f"{self.name}_hypster_config.py"
        if not overwrite and hypster_config_path.exists():
            raise ValueError(
                f"Hypster config already exists for node '{self.name}'. Use overwrite=True to replace it."
            )
        hypster.save(hypster_config, str(hypster_config_path))
        self.hypster_config = hypster_config
        print(f'Hypster config is saved for "{self.name}"')
        self.registry.update_node(self.name, self)

    def load_hypster(self):
        node_info = self.registry.nodes.get(self.name, {})
        hypster_config_path = node_info.get("hypster_config")
        if hypster_config_path and Path(hypster_config_path).exists():
            # Ensure the folder containing the hypster config is in sys.path
            if str(self.folder) not in sys.path:
                sys.path.insert(0, str(self.folder))

            hypster_config_module_name = Path(hypster_config_path).stem
            try:
                hypster_config_module = importlib.import_module(
                    hypster_config_module_name
                )
                self.hypster_config = hypster.config(hypster_config_module)
            except ImportError as e:
                print(
                    f"Failed to import hypster config from {hypster_config_module_name}. Error: {e}"
                )
                raise
        else:
            self.hypster_config = None

    def save(
        self,
        hamilton_dags: List[ModuleType],
        hypster_config: Optional[Hypster] = None,
        overwrite: bool = True,
    ):
        for module in hamilton_dags:
            self.save_dag(module, overwrite)
        if hypster_config:
            self.save_config(hypster_config, overwrite)
        print(
            f'Saved {len(hamilton_dags)} DAG{"s" if len(hamilton_dags) > 1 else ""}'
            f'{" and hypster config" if hypster_config else ""} to node "{self.name}"'
        )
        self.registry.update_node(self.name, self)

    def load_config(self):
        node_info = self.registry.nodes.get(self.name, {})
        hypster_config_path = node_info.get("hypster_config")
        if hypster_config_path:
            spec = importlib.util.spec_from_file_location(
                f"{self.name}_hypster_config", hypster_config_path
            )
            hypster_config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hypster_config_module)
            self.hypster_config = hypster.config(hypster_config_module)
        else:
            self.hypster_config = None
    
    def to_hypernode(self) -> HyperNode:
        return HyperNode(self.name, self.hamilton_dags, self.hypster_config)



class NodeRegistry:
    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = Path(registry_path or self._get_default_registry_path())
        self.nodes: Dict[str, Dict[str, Any]] = self._load_registry()

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    def create_or_get(self, node_name: str, folder: str, overwrite: bool = False) -> 'NodeHandler':
        if node_name in self.nodes:
            existing_folder = self.nodes[node_name]['folder']
            if existing_folder != folder:
                if not overwrite:
                    raise ValueError(f"Node '{node_name}' already exists with a different folder. "
                                     f"Existing: {existing_folder}, Requested: {folder}. "
                                     f"Use overwrite=True to create a new node.")
                else:
                    print(f"Overwriting existing node '{node_name}' with new folder: {folder}")
                    self.nodes[node_name] = {'folder': folder}
                    self._save_registry()
            else:
                print(f'Loaded existing node "{node_name}" from {existing_folder}')
        else:
            self.nodes[node_name] = {'folder': folder}
            self._save_registry()
            print(f'Created new node "{node_name}" in {folder}')
        
        return NodeHandler(node_name, folder, self)

    def set_hypster_config_for_node(self, node_name: str, config_path: str):
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in registry")
        
        self.nodes[node_name]['hypster_config'] = config_path
        self._save_registry()
    
    def add_dag_to_node(self, node_name: str, dag_path: str):
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in registry")
        
        if 'hamilton_dags' not in self.nodes[node_name]:
            self.nodes[node_name]['hamilton_dags'] = []
        
        if dag_path not in self.nodes[node_name]['hamilton_dags']:
            self.nodes[node_name]['hamilton_dags'].append(dag_path)
        self._save_registry()
    
    def update_node(self, node_name: str, handler: 'NodeHandler'):
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in registry")
        
        self.nodes[node_name] = {
            'folder': str(handler.folder),
            'hamilton_dags': [str(handler.folder / f"{node_name}_{module.__name__}.py") for module in handler.hamilton_dags],
            'hypster_config': str(handler.folder / f"{node_name}_hypster_config.py") if handler.hypster_config else None
        }
        self._save_registry()

    def load(self, node_name: str) -> HyperNode:
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in registry")
        
        node_info = self.nodes[node_name]
        node_handler = NodeHandler(node_name, node_info['folder'], self)
        return node_handler.to_hypernode()

    def _save_registry(self):
        with open(self.registry_path, "w") as f:
            yaml.dump(self.nodes, f, default_flow_style=False, sort_keys=False)

    def delete(self, node_name: str):
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in registry")

        del self.nodes[node_name]
        self._save_registry()
        print(f"Deleted node '{node_name}' from registry")

    def _get_default_registry_path(self) -> str:
        try:
            # Look for pyproject.toml in the current directory and parent directories
            current_dir = Path.cwd()
            while current_dir != current_dir.parent:
                pyproject_path = current_dir / "pyproject.toml"
                if pyproject_path.exists():
                    with open(pyproject_path, "rb") as f:
                        pyproject_data = tomli.load(f)
                    registry_path = (
                        pyproject_data.get("tool", {})
                        .get("hypernodes", {})
                        .get("node_registry_path")
                    )
                    if registry_path:
                        registry_path = Path(registry_path)
                        if registry_path.is_absolute():
                            if registry_path.exists():
                                return str(registry_path)
                            else:
                                print(
                                    f"Warning: Specified registry path '{registry_path}' does not exist. Using default."
                                )
                        else:
                            # If it's a relative path, make it relative to the pyproject.toml location
                            full_path = pyproject_path.parent / registry_path
                            if full_path.exists():
                                return str(full_path)
                            else:
                                print(
                                    f"Warning: Specified registry path '{full_path}' does not exist. Using default."
                                )
                current_dir = current_dir.parent

            print(
                "No pyproject.toml found or no valid registry_path specified. Using default."
            )
        except Exception as e:
            print(f"Error reading pyproject.toml: {e}. Using default registry path.")

        raise ValueError("No valid registry path found")
