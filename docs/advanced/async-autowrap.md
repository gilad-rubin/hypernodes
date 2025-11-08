# AsyncExecutor Auto-Wrapping Feature

## Overview

The `AsyncExecutor` now automatically detects whether your functions are synchronous or asynchronous and handles them appropriately:

- **`async def` functions**: Run directly in the event loop (best for true async I/O)
- **`def` functions**: Automatically wrapped with `loop.run_in_executor()` to run in thread pool

This means you can use `map_executor="async"` or `map_executor=AsyncExecutor()` with **regular blocking functions** and get concurrent execution via threading - no need to rewrite as `async def`!

## How It Works

The AsyncExecutor creates a **background event loop in a separate thread**. When you submit a function:

1. It checks if the function is async using `asyncio.iscoroutinefunction()`
2. **Async functions**: Scheduled directly in the event loop with `await func()`
3. **Sync functions**: Wrapped with `loop.run_in_executor(None, lambda: func())` to run in the default thread pool

This design provides:
- ✅ Concurrent execution for both sync and async functions
- ✅ No blocking of the main event loop
- ✅ Jupyter notebook compatibility (no "event loop already running" errors)
- ✅ Seamless mixing of sync and async nodes in pipelines

## Usage Examples

### Basic: Regular Sync Function

```python
from hypernodes import Pipeline, node, HypernodesEngine
import time

@node(output_name="result")
def blocking_fn(x: int) -> int:
    time.sleep(0.1)  # Blocking I/O
    return x ** 2

# AsyncExecutor auto-wraps blocking_fn to run in thread pool
pipeline = Pipeline(
    nodes=[blocking_fn],
    engine=HypernodesEngine(map_executor="async")
)

# 10 items complete in ~0.1s (concurrent), not 1.0s (sequential)
results = pipeline.map(inputs={"x": list(range(10))}, map_over="x")
```

### Advanced: Mixed Sync and Async

```python
@node(output_name="data")
def fetch_from_disk(path: str) -> dict:
    # Sync blocking I/O - auto-wrapped
    with open(path) as f:
        return json.load(f)

@node(output_name="enriched")
async def enrich_with_api(data: dict) -> dict:
    # Async I/O - runs directly in event loop
    async with aiohttp.ClientSession() as session:
        response = await session.get(f"https://api.example.com/{data['id']}")
        data['details'] = await response.json()
    return data

@node(output_name="result")
def save_to_disk(enriched: dict) -> str:
    # Sync blocking I/O - auto-wrapped
    with open('output.json', 'w') as f:
        json.dump(enriched, f)
    return "saved"

# All three node types work together seamlessly!
pipeline = Pipeline(
    nodes=[fetch_from_disk, enrich_with_api, save_to_disk],
    engine=HypernodesEngine(node_executor=AsyncExecutor())
)
```

### High Concurrency

```python
# Process many items concurrently with custom worker count
pipeline = Pipeline(
    nodes=[expensive_io_node],
    engine=HypernodesEngine(
        map_executor=AsyncExecutor(max_workers=100)  # High concurrency
    )
)

results = pipeline.map(inputs={"url": list_of_urls}, map_over="url")
```

## Performance Characteristics

| Function Type | Execution Method | Best For |
|---------------|------------------|----------|
| `async def` with AsyncExecutor | Direct event loop execution | True async I/O (aiohttp, asyncpg, aiofiles) |
| `def` with AsyncExecutor | Thread pool via `run_in_executor()` | Blocking I/O (time.sleep, requests, file I/O) |
| `def` with ThreadPoolExecutor | Thread pool (explicit) | Blocking I/O, when you want explicit control |
| `def` with ProcessPoolExecutor | Process pool | CPU-bound work (avoids GIL) |

## Jupyter Notebook Compatibility

The AsyncExecutor detects when it's running in a Jupyter environment and creates its own background event loop. This avoids the common "event loop already running" error.

```python
# Works perfectly in Jupyter notebooks!
executor = AsyncExecutor()
print(f"Running in Jupyter: {executor._in_jupyter}")  # True in notebooks

pipeline = Pipeline(
    nodes=[my_node],
    engine=HypernodesEngine(map_executor=executor)
)
# No errors! 🎉
```

## Testing

Comprehensive test coverage in:
- `tests/test_executor_adapters.py` - Basic adapter interface tests
- `tests/test_async_autowrap.py` - Auto-wrapping specific tests
- `notebooks/engines.ipynb` - Live demonstrations

Run tests:
```bash
uv run pytest tests/test_async_autowrap.py -v
uv run pytest tests/test_executor_adapters.py -v
```

## When to Use AsyncExecutor

✅ **Use AsyncExecutor when:**
- You have blocking I/O operations (file I/O, requests, time.sleep)
- You want concurrent execution without learning async/await
- You have mixed sync/async nodes in your pipeline
- You're working in Jupyter notebooks
- You want high concurrency (100+ workers) for I/O-bound work

❌ **Don't use AsyncExecutor when:**
- You have CPU-bound computations (use ProcessPoolExecutor/loky instead)
- Your functions release the GIL (use ThreadPoolExecutor)
- You need process isolation

## Migration Guide

### Before: Manual Async Rewrite Required

```python
# Had to rewrite as async to get concurrency
@node
async def process_item(x: int) -> int:
    await asyncio.sleep(0.1)  # Changed from time.sleep
    return x ** 2

pipeline = Pipeline(
    nodes=[process_item],
    engine=HypernodesEngine(map_executor="async")
)
```

### After: Just Use Regular Functions

```python
# Keep it sync - auto-wrapped automatically!
@node
def process_item(x: int) -> int:
    time.sleep(0.1)  # No change needed
    return x ** 2

pipeline = Pipeline(
    nodes=[process_item],
    engine=HypernodesEngine(map_executor="async")
)
# Same concurrent performance! 🚀
```

## Implementation Details

The auto-wrapping is implemented in `src/hypernodes/executors.py`:

```python
async def run_with_semaphore():
    async with self._semaphore:
        if asyncio.iscoroutinefunction(fn):
            # Async function - await directly in event loop
            return await fn(*args, **kwargs)
        else:
            # Sync function - run in thread pool
            return await self._loop.run_in_executor(
                None, 
                lambda: fn(*args, **kwargs)
            )
```

Key design decisions:
1. Uses `asyncio.iscoroutinefunction()` for detection (reliable and fast)
2. Thread pool is created by `run_in_executor(None, ...)` (uses default executor)
3. Semaphore controls concurrency for both sync and async functions
4. Background thread avoids interfering with Jupyter's event loop

## See Also

- [Execution Strategy Guide](notebooks/engines.ipynb) - Performance comparisons
- [Testing Guide](tests/test_async_autowrap.py) - Test examples
- [AnyIO Documentation](https://anyio.readthedocs.io/) - Inspiration for this feature
