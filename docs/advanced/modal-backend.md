# Modal Backend

Requires optional dependencies. Example:

```python
# pip install 'hypernodes[modal]'
import modal
from hypernodes import Pipeline, ModalBackend

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy")
)

backend = ModalBackend(image=image, gpu="A100", memory="32GB")
p = Pipeline(nodes=[...], backend=backend)
res = p.run(inputs={...})
```
