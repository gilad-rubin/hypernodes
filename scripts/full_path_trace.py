"""
Test the FULL HyperNodes execution path step by step.
"""

import asyncio
import time

from hypernodes import HypernodesEngine, Pipeline, node
from hypernodes.executors import AsyncExecutor

# Instrument the engine to track timing
original_map = HypernodesEngine.map


def timed_map(self, pipeline, items, inputs, output_name=None, _ctx=None):
    print(f"\n🔍 Engine.map() called with {len(items)} items")
    start = time.perf_counter()
    result = original_map(self, pipeline, items, inputs, output_name, _ctx)
    elapsed = time.perf_counter() - start
    print(f"   Engine.map() took: {elapsed:.3f}s")
    return result


HypernodesEngine.map = timed_map


# Native async function
@node(output_name="async_result")
async def native_async_fn(delay: float) -> dict:
    task_start = time.perf_counter()
    await asyncio.sleep(delay)
    task_time = time.perf_counter() - task_start
    return {"delay": delay, "task_time": task_time}


# Test parameters
delays = [0.1] * 50

print("=" * 70)
print("🔬 FULL HYPERNODES EXECUTION PATH ANALYSIS")
print("=" * 70)

# Test: HyperNodes with AsyncExecutor
print("\n1️⃣  HyperNodes Full Execution")
pipeline_start = time.perf_counter()

executor = AsyncExecutor(max_workers=50)
pipeline = Pipeline(
    nodes=[native_async_fn],
    backend=HypernodesEngine(map_executor=executor),
)

map_call_start = time.perf_counter()
results = pipeline.map(inputs={"delay": delays}, map_over="delay")
map_call_time = time.perf_counter() - map_call_start

total_time = time.perf_counter() - pipeline_start

executor.shutdown()

print("\n📊 Timing Breakdown:")
print(f"   Pipeline.map() call:   {map_call_time:.3f}s")
print(f"   Total time:            {total_time:.3f}s")


# Baseline: Direct executor usage (no Pipeline)
print("\n\n2️⃣  Direct AsyncExecutor (no Pipeline)")
executor2 = AsyncExecutor(max_workers=50)


async def direct_async_fn(delay):
    await asyncio.sleep(delay)
    return {"delay": delay}


direct_start = time.perf_counter()

futures = []
for d in delays:
    future = executor2.submit(direct_async_fn, d)
    futures.append(future)

results2 = [f.result() for f in futures]
direct_time = time.perf_counter() - direct_start

executor2.shutdown()

print(f"   Direct execution:      {direct_time:.3f}s")


print("\n" + "=" * 70)
print("📊 COMPARISON")
print("=" * 70)
print(f"HyperNodes Pipeline:    {total_time:.3f}s")
print(f"Direct Executor:        {direct_time:.3f}s")
print(f"Overhead:               {(total_time - direct_time) * 1000:.1f}ms ({(total_time - direct_time) / total_time * 100:.1f}%)")

if total_time > direct_time * 2:
    print("\n⚠️  MAJOR OVERHEAD DETECTED")
    print("   The overhead is coming from Pipeline/Engine logic:")
    print("   • Node wrapping/unwrapping")
    print("   • Cache checking")
    print("   • Dependency resolution")
    print("   • Callback invocation")
