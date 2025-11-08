"""
Test to verify that signature computation is skipped when caching is disabled.
"""

import asyncio
import time

# Patch compute_node_signature to track calls
from hypernodes import HypernodesEngine, Pipeline, node, node_execution
from hypernodes.executors import AsyncExecutor

original_compute = node_execution.compute_node_signature

call_count = 0

def tracked_compute(node, inputs, node_signatures):
    global call_count
    call_count += 1
    return original_compute(node, inputs, node_signatures)

node_execution.compute_node_signature = tracked_compute


# Native async function
@node(output_name="async_result", cache=False)  # Cache DISABLED
async def native_async_fn(delay: float) -> dict:
    await asyncio.sleep(delay)
    return {"delay": delay}


delays = [0.1] * 50

print("=" * 70)
print("🔬 TESTING SIGNATURE COMPUTATION WITH CACHE DISABLED")
print("=" * 70)

# Test: HyperNodes with AsyncExecutor, cache disabled
print("\n1️⃣  HyperNodes with cache=False")
call_count = 0
executor = AsyncExecutor(max_workers=50)
pipeline = Pipeline(
    nodes=[native_async_fn],
    backend=HypernodesEngine(map_executor=executor),
    cache=None,  # No cache
)

start = time.time()
results = pipeline.map(inputs={"delay": delays}, map_over="delay")
elapsed = time.time() - start

print(f"   Time: {elapsed:.3f}s")
print(f"   compute_node_signature() calls: {call_count}")

if call_count == 0:
    print("   ✅ SUCCESS: Signature computation skipped!")
else:
    print(f"   ❌ FAIL: Signature still computed {call_count} times")

executor.shutdown()


# Test 2: With cache enabled
print("\n2️⃣  HyperNodes with cache=True")

@node(output_name="cached_result", cache=True)  # Cache ENABLED
async def cached_async_fn(delay: float) -> dict:
    await asyncio.sleep(delay)
    return {"delay": delay}

import os
import tempfile

from hypernodes.cache import DiskCache

cache_dir = os.path.join(tempfile.gettempdir(), "test_cache")
cache = DiskCache(cache_dir)

call_count = 0
executor2 = AsyncExecutor(max_workers=50)
pipeline2 = Pipeline(
    nodes=[cached_async_fn],
    backend=HypernodesEngine(map_executor=executor2),
    cache=cache,  # Cache enabled
)

start = time.time()
results2 = pipeline2.map(inputs={"delay": delays}, map_over="delay")
elapsed2 = time.time() - start

print(f"   Time: {elapsed2:.3f}s")
print(f"   compute_node_signature() calls: {call_count}")

if call_count == len(delays):
    print(f"   ✅ SUCCESS: Signature computed {call_count} times (as expected)")
else:
    print(f"   ⚠️  Unexpected: Expected {len(delays)} calls, got {call_count}")

executor2.shutdown()

# Cleanup
import shutil

if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
