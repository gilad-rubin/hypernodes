# Quick Start

```python
from hypernodes import Pipeline, node

@node(output_name="cleaned_text")
def clean_text(passage: str) -> str:
    return passage.strip().lower()

@node(output_name="word_count")
def count_words(cleaned_text: str) -> int:
    return len(cleaned_text.split())

pipeline = Pipeline(nodes=[clean_text, count_words])

# Single input (two options)
print(pipeline(passage="Hello World"))
print(pipeline.run(inputs={"passage": "Hello World"}))

# Map over many inputs
results = pipeline.map(inputs={"passage": ["Hello", "World"]}, map_over="passage")
print(results)
```

Enable caching:

```python
from hypernodes import DiskCache

pipeline = Pipeline(nodes=[clean_text, count_words], cache=DiskCache(path=".cache"))
print(pipeline.run(inputs={"passage": "Hello World"}))
print(pipeline.run(inputs={"passage": "Hello World"}))  # Cached
```
