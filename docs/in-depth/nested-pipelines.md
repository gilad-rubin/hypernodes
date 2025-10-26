# Nested Pipelines

Wrap a pipeline as a node with interface adaptation and mapping:

```python
inner = Pipeline(nodes=[clean_text, encode_text])
adapted = inner.as_node(
  input_mapping={"corpus": "passage"},
  output_mapping={"embedding": "encoded_corpus"},
  map_over="corpus",
)
outer = Pipeline(nodes=[adapted, build_index])
result = outer.run(inputs={"corpus": ["Hello", "World"]})
```

Configuration (backend, cache, callbacks) is inherited unless overridden.
