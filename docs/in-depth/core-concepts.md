# Core Concepts

- Functions become nodes via `@node(output_name=...)`. Dependencies are inferred from parameter names.
- Pipelines are DAGs of nodes. `Pipeline(nodes=[...])` computes topological order and runs nodes.
- Pipelines can nest, and a pipeline can be adapted as a node with `pipeline.as_node(...)`.

```python
from hypernodes import Pipeline, node

@node(output_name="a")
def make_a(x: int) -> int: return x + 1

@node(output_name="b")
def make_b(a: int) -> int: return a * 2

p = Pipeline(nodes=[make_a, make_b])
print(p.run(inputs={"x": 5}))  # {"a": 6, "b": 12}
```
