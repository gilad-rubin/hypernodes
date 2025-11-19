# Stateful Objects

**Optimize expensive initialization with the `@stateful` decorator.**

When your pipeline uses objects that are expensive to initialize (ML models, database connections, API clients), use the `@stateful` decorator to enable lazy initialization and automatic reuse.

## What are Stateful Objects?

Stateful objects are resources that:

- **Take time to initialize** (loading ML models, connecting to databases)
- **Should be reused** across multiple operations
- **Maintain state** between operations

The `@stateful` decorator provides:

- **Lazy Initialization**: `__init__` is not called when creating the instance, only on first use
- **Efficient Serialization**: Only init arguments are pickled, not heavy state (models, connections)
- **Automatic Cache Keys**: Generated from init arguments for correct caching
- **Engine Optimization**: SequentialEngine caches instances, DaftEngine handles per-worker init

## Basic Usage

### Mark a Class as Stateful

```python
from hypernodes import stateful, node, Pipeline

@stateful
class ExpensiveModel:
    def __init__(self, model_path: str):
        # This expensive operation is LAZY - only happens on first use
        self.model = load_model(model_path)
    
    def predict(self, text: str) -> str:
        return self.model(text)

@node(output_name="prediction")
def predict(text: str, model: ExpensiveModel) -> str:
    return model.predict(text)

# Create model instance - __init__ is NOT called yet!
model = ExpensiveModel(model_path="./model.pkl")

# Use in pipeline
pipeline = Pipeline(nodes=[predict])

# On first use, __init__ is called ONCE, then reused for all items
results = pipeline.map(
    inputs={"text": ["hello", "world"], "model": model},
    map_over="text"
)
# __init__ called on first "hello", same instance used for "world"
```

### How It Works: Lazy Initialization

When you decorate a class with `@stateful`, it returns a **wrapper** that delays initialization:

1. **Creating instance**: `model = ExpensiveModel("path.pkl")` → stores args, **doesn't call `__init__`**
2. **First access**: `model.predict(...)` → **now calls `__init__`**, creates real instance
3. **Subsequent access**: `model.predict(...)` → reuses same instance

This enables:
- **Efficient serialization**: Only init args pickled (not 10GB model weights!)
- **Cache-friendly**: Cache keys based on init args, not instance state
- **Engine optimization**: 
  - **SequentialEngine**: One instance for entire `.map()` operation
  - **DaftEngine**: Daft calls `__init__` once per worker (future work)

## Caching with Stateful Objects

### Automatic Cache Keys

**Good news: You don't need to do anything!** The `@stateful` decorator automatically generates cache keys from your initialization arguments:

```python
@stateful
class Model:
    def __init__(self, model_path: str, version: str = "v1"):
        self.model = load_model(model_path, version)
    
    def predict(self, x: int) -> int:
        return self.model.predict(x)

# These create DIFFERENT cache keys (different init args)
model_v1 = Model("model.pkl", version="v1")
model_v2 = Model("model.pkl", version="v2")

# These create the SAME cache key (same init args)
model_a = Model("model.pkl", version="v1")
model_b = Model("model.pkl", version="v1")  # Same as model_a
```

The cache key is automatically computed as:
```
hash(class_name + init_args + init_kwargs)
```

### Custom Cache Keys (Optional)

For advanced cases, implement `__cache_key__()` on the **wrapped class**:

```python
@stateful
class Model:
    def __init__(self, model_path: str, version: str = "v1"):
        self.model_path = model_path
        self.version = version
        self.model = load_model(model_path, version)
    
    def __cache_key__(self) -> str:
        """Custom cache key - only use model_path, ignore version."""
        return f"Model(path={self.model_path})"
    
    def predict(self, x: int) -> int:
        return self.model.predict(x)
```

**When to use custom `__cache_key__()`:**
- Ignore certain init parameters (e.g., logging config)
- Use file hashes instead of paths
- Complex objects that need special handling

### Cache Key Best Practices

1. **Include all initialization parameters** that affect behavior:

```python
def __cache_key__(self) -> str:
    return f"Model(path={self.model_path}, version={self.version}, threshold={self.threshold})"
```

2. **Be deterministic** - same params should always produce same key:

```python
# Good: deterministic
def __cache_key__(self) -> str:
    return f"Config(batch_size={self.batch_size}, lr={self.lr})"

# Bad: non-deterministic (time-dependent)
def __cache_key__(self) -> str:
    return f"Config(created_at={time.time()})"  # ❌
```

3. **Handle complex types** by serializing them:

```python
def __cache_key__(self) -> str:
    import json
    return f"Processor(config={json.dumps(self.config, sort_keys=True)})"
```

## Common Use Cases

### Machine Learning Models

```python
@stateful  # Automatic cache key from (model_name, device)
class BERTModel:
    def __init__(self, model_name: str, device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer
        # Expensive initialization - only happens on first use!
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
    
    def encode(self, text: str) -> list[float]:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

@node(output_name="embedding")
def encode_text(text: str, model: BERTModel) -> list[float]:
    return model.encode(text)

# Create model (doesn't load transformers yet - lazy!)
model = BERTModel(model_name="bert-base-uncased", device="cuda")

# Process many documents (model loads on first doc, reused for all)
pipeline = Pipeline(nodes=[encode_text])
results = pipeline.map(
    inputs={"text": documents, "model": model},
    map_over="text"
)
```

### Database Connections

```python
@stateful  # Automatic cache key from connection_string
class DatabaseConnection:
    def __init__(self, connection_string: str):
        import psycopg2
        # Connection established lazily on first query
        self.conn = psycopg2.connect(connection_string)
    
    def query(self, sql: str) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute(sql)
        return cursor.fetchall()
    
    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()

@node(output_name="user_data")
def fetch_user(user_id: int, db: DatabaseConnection) -> dict:
    result = db.query(f"SELECT * FROM users WHERE id = {user_id}")
    return result[0] if result else None

# Create connection (not opened yet - lazy!)
db = DatabaseConnection("postgresql://localhost/mydb")

pipeline = Pipeline(nodes=[fetch_user])

# Connection opened on first query, reused for all
results = pipeline.map(
    inputs={"user_id": [1, 2, 3, 4, 5], "db": db},
    map_over="user_id"
)
```

### API Clients

```python
@stateful  # Automatic cache key from (api_key, base_url)
class APIClient:
    def __init__(self, api_key: str, base_url: str = "https://api.example.com"):
        import httpx
        # Client created lazily on first request
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url)
    
    def get(self, endpoint: str) -> dict:
        response = self.client.get(
            endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return response.json()
    
    def __del__(self):
        if hasattr(self, 'client'):
            self.client.close()

@node(output_name="data")
def fetch_data(item_id: str, client: APIClient) -> dict:
    return client.get(f"/items/{item_id}")

# Create client (not connected yet - lazy!)
client = APIClient(api_key="secret-key")

pipeline = Pipeline(nodes=[fetch_data])

# Client connects on first request, connection pooled for all
results = pipeline.map(
    inputs={"item_id": ["A", "B", "C"], "client": client},
    map_over="item_id"
)
```

## Engine-Specific Behavior

### SequentialEngine

In `SequentialEngine`, stateful objects are **cached for the entire `.map()` operation**:

```python
# Object lifecycle in SequentialEngine.map()
model = ExpensiveModel("path.pkl")  # ← __init__ called here

results = pipeline.map(
    inputs={"x": [1, 2, 3], "model": model},
    map_over="x"
)
# model is passed to each iteration (not recreated)
```

**Benefits:**
- Simple, predictable behavior
- No serialization overhead
- Same instance used sequentially

### DaftEngine

In `DaftEngine`, stateful objects are **wrapped with `@daft.cls`** automatically:

```python
# Object lifecycle in DaftEngine
model = ExpensiveModel("path.pkl")  # ← __init__ NOT called yet (lazy)

results = pipeline.map(
    inputs={"x": [1, 2, 3], "model": model},
    map_over="x"
)
# DaftEngine detects stateful parameter
# Extracts original class and init args from StatefulWrapper
# Wraps node function with @daft.cls
# On worker: __init__ called ONCE, instance reused for all rows
```

**This is equivalent to manually writing:**
```python
@daft.cls
class Wrapper:
    def __init__(self):
        self.model = ExpensiveModel("path.pkl")  # Once per worker
    
    def process(self, x: int) -> int:
        return x * self.model.predict(...)

wrapper = Wrapper()
df = df.with_column("result", wrapper(df["x"]))
```

**Benefits:**
- Automatic `@daft.cls` wrapping (you just use `@stateful`)
- Per-worker initialization (parallel execution)
- Resource control (gpus, max_concurrency via engine config)
- Works seamlessly with Daft's distributed execution

## Best Practices

### 1. Use `@stateful` for Expensive Resources

**Do:**
```python
@stateful
class HeavyModel:
    def __init__(self, path: str):
        self.model = load_100gb_model(path)  # Expensive!
```

**Don't:**
```python
@stateful  # ❌ Unnecessary
class SimpleCounter:
    def __init__(self):
        self.count = 0  # Cheap to create
```

### 2. Cache Keys are Automatic

**Do (simple case):**
```python
@stateful  # ✅ Auto cache key from init args
class Model:
    def __init__(self, path: str):
        self.model = load_model(path)
```

**Do (custom cache key for advanced cases):**
```python
@stateful
class Model:
    def __init__(self, path: str, log_level: str = "INFO"):
        self.model = load_model(path)
        self.logger = setup_logger(log_level)
    
    def __cache_key__(self) -> str:
        # Only cache based on path, ignore log_level
        return f"Model({self.path})"  # Custom logic
```

### 3. Handle Resource Cleanup

Use `__del__()` for cleanup if needed:

```python
@stateful
class DatabasePool:
    def __init__(self, url: str):
        self.pool = create_pool(url)
    
    def __del__(self):
        if hasattr(self, 'pool'):
            self.pool.close()  # Clean up on deletion
```

### 4. Be Careful with Mutable State

Stateful objects are reused, so mutations affect all operations:

```python
@stateful
class Counter:
    def __init__(self):
        self.count = 0
    
    def increment(self, x: int) -> int:
        self.count += 1  # ⚠️ Mutates shared state
        return x + self.count

# This can lead to unexpected behavior in .map()
# Item 1: count=1, Item 2: count=2, etc.
```

**Solution:** Make operations pure or use locks if mutations are necessary.

### 5. Test Both `.run()` and `.map()`

Ensure your stateful objects work in both contexts:

```python
def test_model_single():
    model = Model("path.pkl")
    pipeline = Pipeline(nodes=[predict])
    result = pipeline.run(inputs={"x": 1, "model": model})
    assert result == {"prediction": 2}

def test_model_batch():
    model = Model("path.pkl")
    pipeline = Pipeline(nodes=[predict])
    results = pipeline.map(
        inputs={"x": [1, 2, 3], "model": model},
        map_over="x"
    )
    assert len(results) == 3
```

## Debugging

### Enable Warnings

To see warnings about missing `__cache_key__()`:

```python
import warnings
warnings.simplefilter("always")  # Show all warnings
```

### Check if Object is Stateful

```python
@stateful
class MyClass:
    pass

obj = MyClass()

# Check for marker
assert hasattr(obj.__class__, "__hypernode_stateful__")
assert obj.__class__.__hypernode_stateful__ is True
```

### Verify Caching Behavior

```python
from hypernodes import SequentialEngine, DiskCache

@stateful
class Model:
    def __init__(self, path: str):
        self.path = path
        self.load_count = 0
    
    def __cache_key__(self) -> str:
        return f"Model({self.path})"
    
    def load(self):
        self.load_count += 1

@node(output_name="result")
def process(x: int, model: Model) -> int:
    model.load()
    return x * 2

model = Model("test.pkl")
engine = SequentialEngine(cache=DiskCache(path=".cache"))
pipeline = Pipeline(nodes=[process], engine=engine)

# First run
result1 = pipeline.run(inputs={"x": 5, "model": model})
print(f"Load count: {model.load_count}")  # Should be 1

# Second run (cached)
result2 = pipeline.run(inputs={"x": 5, "model": model})
print(f"Load count: {model.load_count}")  # Still 1 (cached)
```

## Migration from `__daft_stateful__`

If you have existing code using `__daft_stateful__ = True`, migrate to `@stateful`:

**Before:**
```python
class Model:
    __daft_stateful__ = True
    
    def __init__(self, path: str):
        self.model = load_model(path)
```

**After:**
```python
@stateful  # That's it! Auto-generates cache keys from init args
class Model:
    def __init__(self, path: str):
        self.model = load_model(path)
```

The `@stateful` decorator provides:
- ✅ Lazy initialization (only on first use)
- ✅ Automatic cache keys (from init args)
- ✅ Engine-agnostic optimization
- ✅ Efficient serialization (args only, not state)

## See Also

- [Caching Guide](../in-depth/caching.md) - How caching works in HyperNodes
- [DaftEngine](../engines/daft-engine.md) - Distributed execution with Daft
- [DualNode](./dual-node.md) - Combining stateful objects with batch operations

