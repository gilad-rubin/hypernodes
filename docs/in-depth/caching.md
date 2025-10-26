# Caching

HyperNodes computes a content-addressed signature per node:

```
sig(node) = hash(code_hash + env_hash + inputs_hash + deps_hash)
```

- `code_hash`: from function source (and closure) via `hash_code`
- `inputs_hash`: hashed values, nested structures supported
- `deps_hash`: upstream node signatures for fine-grained invalidation

Use `DiskCache`:

```python
from hypernodes import Pipeline
from hypernodes.cache import DiskCache

cache = DiskCache(path=".cache")
p = Pipeline(nodes=[...], cache=cache)
result = p.run(inputs={...})
```
