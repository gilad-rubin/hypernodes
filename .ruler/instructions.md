## General
- use uv run X to run scripts
- use trash X instead of rm X. This allows me to rescue deleted files if I need.
- when making changes in the codebase - run them to verify everything works
- if an API key is needed, first check in the .env to make sure it exists. use dotenv to load if needed.
- prefer to search online and in documentations before acting
- tests go to tests folder. scripts go to scripts folder

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