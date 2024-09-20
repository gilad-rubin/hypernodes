import glob
import os
import sys
from typing import Any, Dict, List, Optional

import mlflow
import yaml

from .registry import NodeInfo


def log_artifacts(artifact_files, artifact_folders):
    # Log individual files
    for artifact_name, file_path in artifact_files.items():
        mlflow.log_artifact(file_path, artifact_name)

    # Log folders and their contents
    for folder_path in artifact_folders:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, start=os.path.dirname(folder_path))
                mlflow.log_artifact(full_path, relative_path)


def get_dags_from_artifacts(node_name, artifacts):
    dag_paths = []
    for artifact_name, artifact_path in artifacts.items():
        if artifact_name.startswith(f"{node_name}_dag_"):
            dag_paths.append(artifact_path)
    return dag_paths


class HyperNodeMLFlow(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        node_name,
        dynamic_inputs,
        final_vars,
        final_outputs=[],
        selections={},
        overrides={},
        env_filename=None,
    ):
        self.node_name = node_name
        self.dynamic_inputs = dynamic_inputs
        self.final_vars = [final_vars] if isinstance(final_vars, str) else final_vars
        self.final_outputs = final_outputs if len(final_outputs) > 0 else self.final_vars
        self.selections = selections
        self.overrides = overrides
        self.env_filename = env_filename
        self.context_loaded = False
        self.node = None

    def _load_dotenv(self, artifacts, env_filename):
        from dotenv import load_dotenv

        if env_filename in artifacts:
            dotenv_path = artifacts.get(env_filename)
            load_dotenv(dotenv_path=dotenv_path)
        else:
            raise ValueError(f"Environment file {env_filename} not found in artifacts")

    def load_context(self, context):
        if "node_registry_path" in context.artifacts:
            self.node_registry_path = context.artifacts["node_registry_path"]

        from hypernodes import create_registry

        registry = create_registry(registry_path=self.node_registry_path)

        new_nodes = {}
        for node_name, node_info in registry.nodes.items():
            new_node_info = NodeInfo(name=node_info.name, folder=node_info.folder)

            for field in ["hamilton_dag_paths", "hypster_config_path", "builder_param_name"]:
                value = getattr(node_info, field)
                if value:
                    if field == "hamilton_dag_paths":
                        new_value = get_dags_from_artifacts(node_name, context.artifacts)
                    else:
                        artifact_key = f"{node_name}_{field}"
                        new_value = context.artifacts.get(artifact_key, value)
                    setattr(new_node_info, field, new_value)

            new_nodes[node_name] = new_node_info

        registry.nodes = new_nodes
        registry.store_handler.file_path = "node_registry_updated.yaml"
        context.artifacts["node_registry_path"] = registry.store_handler.file_path
        registry._save_nodes()

        self.node = registry.load(node_name=self.node_name)
        if self.env_filename:
            self._load_dotenv(context.artifacts, self.env_filename)

        overrides = self.overrides.copy()
        overrides.update(context.artifacts)

        self.node.instantiate(selections=self.selections, overrides=overrides)
        self.context_loaded = True
        print(self.node._instantiated_config)

    def add_cols_to_df(self, df, input_df):
        for col in input_df.columns:
            if col not in df.columns:
                df[col] = input_df[col].values[0]
        return df

    def predict(self, context, model_input, params=None):
        import pandas as pd

        if not self.context_loaded:
            self.load_context(context)

        dynamic_df = model_input[self.dynamic_inputs]
        dynamic_inputs_dct = dynamic_df.to_dict(orient="records")[0]

        results = self.node.execute(
            final_vars=self.final_vars, additional_inputs=dynamic_inputs_dct
        )

        if len(self.final_outputs) == 1:
            results = results[self.final_outputs[0]]
            if isinstance(results, pd.DataFrame):
                results = self.add_cols_to_df(results, model_input)

        return results


class EnvironmentGenerator:
    def __init__(
        self,
        env_name: str,
        python_version: Optional[str] = None,
        dependency_file: Optional[str] = None,
        extra_dependencies: Optional[List[str]] = None,
    ):
        self.env_name = env_name
        self.python_version = python_version or self.detect_python_version()
        self.dependency_file = dependency_file
        self.extra_dependencies = extra_dependencies or []
        self.dependencies: List[str] = []

        self._load_dependencies()

    @staticmethod
    def detect_python_version() -> str:
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    def _load_dependencies(self):
        if self.dependency_file:
            self._parse_dependency_file()
        self.dependencies.extend(self.extra_dependencies)

    def _parse_dependency_file(self):
        if not os.path.exists(self.dependency_file):
            raise FileNotFoundError(f"Dependency file not found: {self.dependency_file}")

        # Simple parsing for requirements.txt
        # TODO: Add support for pyproject.toml and setup.py
        with open(self.dependency_file, "r") as f:
            self.dependencies.extend(
                line.strip() for line in f if line.strip() and not line.startswith("#")
            )

    def to_conda(self) -> Dict:
        return {
            "name": self.env_name,
            "channels": ["defaults"],
            "dependencies": [
                f"python={self.python_version}",
                "pip",
                {"pip": self.dependencies},
            ],
        }

    def get_pip_requirements_list(self) -> List[str]:
        return self.dependencies.copy()

    def export_conda_yaml(self, filepath: str = "environment.yml"):
        conda_env = self.get_conda_environment_dict()
        with open(filepath, "w") as f:
            yaml.dump(conda_env, f, default_flow_style=False)

    def export_pip_requirements(self, filepath: str = "requirements.txt"):
        with open(filepath, "w") as f:
            f.write("\n".join(self.get_pip_requirements_list()))


def find_files_with_suffix(directory, suffix):
    pattern = os.path.join(directory, f"**/*{suffix}")
    return glob.glob(pattern, recursive=True)


def get_code_paths(base_dir=".", folders=["src"], suffixes=[".py"]):
    if isinstance(suffixes, str):
        suffixes = [suffixes]

    all_paths = []
    base_dir = os.path.abspath(base_dir)

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        for suffix in suffixes:
            paths = find_files_with_suffix(folder_path, suffix)
            # Convert absolute paths to relative paths
            relative_paths = [os.path.relpath(path, base_dir) for path in paths]
            all_paths.extend(relative_paths)

    return all_paths


# def registry_to_artifacts(node_registry):
#     registry_path = node_registry.registry_path
#     if os.path.exists(registry_path):
#         with open(registry_path, "r") as file:
#             data = yaml.safe_load(file) or {}
#     else:
#         print(f"Input file {registry_path} does not exist. Starting with an empty dictionary.")
#         data = {}

#     artifacts = {}
#     for node_name, node_info in data.items():
#         if "hamilton_dags" in node_info:
#             for dag_path in node_info["hamilton_dags"]:
#                 artifacts[f"{node_name}_dag_{os.path.basename(dag_path)}"] = dag_path

#         if "hypster_config" in node_info:
#             artifacts[f"{node_name}_hypster_config"] = node_info["hypster_config"]

#     return artifacts


def registry_to_artifacts(node_registry) -> Dict[str, Any]:
    artifacts = {}
    for node_name, node_info in node_registry.nodes.items():
        for field in ["hamilton_dag_paths", "hypster_config_path"]:
            value = getattr(node_info, field)
            if value:
                if field == "hamilton_dag_paths":
                    for dag_path in value:
                        artifact_key = f"{node_name}_dag_{os.path.basename(dag_path)}"
                        artifacts[artifact_key] = dag_path
                else:
                    artifact_key = f"{node_name}_{field}"
                    artifacts[artifact_key] = value

    return artifacts


def get_existing_dependencies_files():
    optional_files = [
        "requirements.txt",
        "environment.yml",
        "environment.yaml",
        "pyproject.toml",
        "*.toml",
    ]

    project_root = os.path.abspath(os.path.join(os.path.abspath("")))
    existing_files = []

    for file_pattern in optional_files:
        matches = glob.glob(os.path.join(project_root, file_pattern))
        existing_files.extend([os.path.basename(f) for f in matches])

    existing_files = list(set(existing_files))  # Remove duplicates
    return existing_files


class MLFlowSetup:
    def __init__(self, registry, artifacts, dotenv_file, code_paths, conda_env):
        self.registry = registry
        self.artifacts = artifacts
        self.dotenv_file = dotenv_file
        self.code_paths = code_paths
        self.conda_env = conda_env

    def log_model(self, model, artifact_path="model"):
        artifacts = self.artifacts.copy()
        artifacts.update(registry_to_artifacts(self.registry))
        artifacts["node_registry_path"] = self.registry.store_handler.file_path
        with mlflow.start_run():
            model_info = mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=model,
                artifacts=artifacts,
                conda_env=self.conda_env,
                code_paths=self.code_paths,
            )
            model_uri = f"runs:/{model_info.run_id}/model"
            return model_uri

    def register_model(self, model_uri, model_name):
        model_reg = mlflow.register_model(model_uri, model_name)
        return model_reg

    def load_model(self, model_uri):
        model = mlflow.pyfunc.load_model(model_uri)
        return model

    # .log_register
    # .register
    # .test_local() #example
    # .deploy(...)
    # .test_remote() #example
