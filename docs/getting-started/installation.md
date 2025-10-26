# Installation

- Python 3.12+

Install core:

```bash
pip install hypernodes
```

Optional extras:

- Visualization: `pip install 'graphviz' 'rich' 'ipywidgets' 'tqdm' 'plotly'`
- Modal (remote backend): `pip install 'modal' 'cloudpickle'` or `pip install 'hypernodes[modal]'`
- Telemetry: `pip install 'logfire'` or `pip install 'hypernodes[telemetry]'`

Extras via project metadata (if using uv/pip):
- `pip install "hypernodes[viz]"`
- `pip install "hypernodes[modal]"`
- `pip install "hypernodes[telemetry]"`
