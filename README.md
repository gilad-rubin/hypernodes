<div align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/dark_background_logo.svg">
  <img alt="hypernodes" src="assets/light_background_logo.svg" width=700">
</picture></div>

<p align="center">
  <a href="#installation">[Installation]</a> |
  <a href="#quick-start">[Quick Start]</a> |
  <a href="#license">[License]</a>
</p>

**`hypernodes`** is a lightweight Python package designed to bind together Hamilton DAGs and Hypster configurations to create modular, extensible and highly optimized AI workflows.

```{warning}
This package is currently in active development and should not be used in production environments.
```

## Installation

Install hypernodes using pip:

```bash
pip install hypernodes
```

## Quick Start

Here's a simple example of how to use hypernodes:

```python
from hypernodes import registry

# Create or get a HyperNode
node = registry.create_or_get("example_node")

# Define Hypster configuration

from hypster import HP, config

@config
def my_config(hp: HP):
  data_path = hp.text_input("data")
  env = hp.select(["dev", "prod"], default="dev")
  llm_model = hp.select({"haiku": "claude-3-haiku-20240307",
                         "sonnet": "claude-3-5-sonnet-20240620"}, default="haiku")

# Save Hypster configuration
node.save_hypster_config(my_config)

# Define Hamilton DAG
def query(llm_model: str) -> str:
  return f"Querying {llm_model}..."

# Save Hamilton DAG
node.save_dag(dag)

# Load and execute
node = registry.load("example_node")
node.instantiate(selections={"llm_model": "sonnet"},
                 overrides={"data_path": "data_folder"})

results = node.execute()
print(results) # {'query': 'Querying claude-3-5-sonnet-20240620...'}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.