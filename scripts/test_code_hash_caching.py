"""
Test to verify that code hash is cached and not recomputed.
"""

import asyncio
import time

from hypernodes import HypernodesEngine, Pipeline, node

# Patch hash_code to track calls
from hypernodes import cache as cache_module
from hypernodes.executors import AsyncExecutor

original_hash_code = cache_module.hash_code

hash_code_calls = 0

def tracked_hash_code(func):
    global hash_code_calls
    hash_code_calls += 1
    print(f"   [hash_code called #{hash_code_calls}]")
    return original_hash_code(func)

cache_module.hash_code = tracked_hash_code


# Native async function
@node(output_name="async_result")
async def native_async_fn(delay: float) -> dict:
    await asyncio.sleep(delay)
    return {"delay": delay}


delays = [0.1] * 50

print("=" * 70)
print("🔬 TESTING CODE HASH CACHING")
print("=" * 70)

# Test: HyperNodes with AsyncExecutor
print("\n1️⃣  Creating pipeline with 1 node...")
hash_code_calls = 0
executor = AsyncExecutor(max_workers=50)
pipeline = Pipeline(
    nodes=[native_async_fn],
    backend=HypernodesEngine(map_executor=executor),
)

print(f"   hash_code() calls after pipeline creation: {hash_code_calls}")

print(f"\n2️⃣  Executing pipeline.map() with {len(delays)} items...")
hash_code_calls = 0

start = time.time()
results = pipeline.map(inputs={"delay": delays}, map_over="delay")
elapsed = time.time() - start

print(f"   Time: {elapsed:.3f}s")
print(f"   hash_code() calls during execution: {hash_code_calls}")

if hash_code_calls == 0:
    print("   ✅ SUCCESS: hash_code() not called during execution (using cached value)!")
elif hash_code_calls == 1:
    print("   ✅ GOOD: hash_code() called only once (cached for subsequent calls)")
elif hash_code_calls <= len(delays):
    print(f"   ⚠️  PARTIAL: hash_code() called {hash_code_calls} times (should be 0 or 1)")
else:
    print(f"   ❌ FAIL: hash_code() called {hash_code_calls} times (more than items!)")

executor.shutdown()

print("\n" + "=" * 70)
print("💡 EXPECTED BEHAVIOR:")
print("=" * 70)
print("   • hash_code() should be called at most ONCE (when node.code_hash is first accessed)")
print("   • Subsequent calls should use the cached value")
print("   • This avoids expensive inspect.getsource() on every execution")
