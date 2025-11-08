## General
- use uv run X to run scripts
- use trash X instead of rm X. This allows me to rescue deleted files if I need.
- when making changes in the codebase - run them to verify everything works
- if an API key is needed, first check in the .env to make sure it exists. use dotenv to load if needed.
- prefer to search online and in documentations before acting
- tests go to tests folder. scripts go to scripts folder

## Coding Principles
- When designing and implementing features - always prefer using SOLID principles.
- Use simple, human readable functions rather than massive long indented functions.
- Split classes functions into helper functions if needed

## API and Architecture

### Current API (Post-Refactoring)
- **Engine-based**: Use `HypernodesEngine` (NOT Backend-based)
- **Executors**: `node_executor` and `map_executor` parameters (NOT `node_execution` or `map_execution`)
- **Executor types**: `"sequential"`, `"async"`, `"threaded"`, `"parallel"`
- **Import paths**:
  - Main: `from hypernodes import Pipeline, node, HypernodesEngine, DiskCache`
  - Engines: `from hypernodes.engines import DaftEngine`
  - Telemetry: `from hypernodes.telemetry import ProgressCallback, TelemetryCallback`

### Architecture Components
- **Node** (`node.py`): Function wrapper with output_name and parameters
- **Pipeline** (`pipeline.py`): DAG manager with `.run()` and `.map()` methods
- **Engine** (`engine.py`): Execution orchestrator (HypernodesEngine is default)
- **Executors** (`executors.py`): SequentialExecutor, AsyncExecutor (auto-wraps sync functions)
- **Node Execution** (`node_execution.py`): Single node execution with caching and callbacks
- **Cache** (`cache.py`): DiskCache with content-addressed signatures
- **Callbacks** (`callbacks.py`): Lifecycle hooks (ProgressCallback, TelemetryCallback, etc.)

### Example Usage
```python
from hypernodes import Pipeline, node, HypernodesEngine, DiskCache

@node(output_name="result")
def process(text: str) -> str:
    return text.upper()

pipeline = Pipeline(
    nodes=[process],
    engine=HypernodesEngine(node_executor="async"),
    cache=DiskCache(path=".cache")
)

result = pipeline(text="hello")
```

## Tools
- use tavily web search and context7 MCP servers whenever you're stuck or want to understand how a library works

## Jupyter
- Use concise, human readable cells
- avoid redundancy in notebooks. keep the cells and notebook as a whole concise.
- avoid using "special" emojis in jupyter, it can crash the notebook. you can use basic ones, like X, V etc...
- remember that jupyter has its own async handling. remember to use the correct syntax.
- If you're editing a module while reviewing the output in jupyter, remember to either restart the kernel or reload the module to see changes

- jupyter notebook's working directory is the project's working directory, so no need to do sys.path.insert(0, '/Users/...')
- run cells after you create them to verify things work as expected. read the output and decide what to do next
- when trying to figure out something about an object - iterate over it by running the cell and examining the output and refining