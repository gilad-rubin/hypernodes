# Reproducibility

HyperNodes uses stable hashing of code, inputs, and dependency signatures to ensure deterministic caching.

Key functions (from `hypernodes.cache`):
- `hash_code(func)`
- `hash_inputs(dict)`
- `compute_signature(code_hash, inputs_hash, deps_hash, env_hash)`

Use a persistent cache directory (e.g., `.cache`) and version control your pipeline code to preserve signatures.
