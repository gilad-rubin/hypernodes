"""
Comprehensive trace of HyperNodes execution to find slowdown.
"""

import asyncio
import time

from hypernodes import HypernodesEngine, Pipeline, node
from hypernodes.executors import AsyncExecutor

# Patch AsyncExecutor to add timing
original_submit = AsyncExecutor.submit


def timed_submit(self, fn, *args, **kwargs):
    """Wrapper to time submit calls."""
    start = time.perf_counter()
    result = original_submit(self, fn, *args, **kwargs)
    elapsed = time.perf_counter() - start
    if not hasattr(self, '_submit_times'):
        self._submit_times = []
    self._submit_times.append(elapsed)
    return result


AsyncExecutor.submit = timed_submit


# Native async function
@node(output_name="async_result")
async def native_async_fn(delay: float) -> dict:
    start = time.perf_counter()
    await asyncio.sleep(delay)
    elapsed = time.perf_counter() - start
    return {"delay": delay, "actual": elapsed}


# Test parameters
delays = [0.1] * 50

print("=" * 70)
print("🔬 DETAILED HYPERNODES EXECUTION TRACE")
print("=" * 70)

# Test: HyperNodes with AsyncExecutor
print("\n1️⃣  HyperNodes AsyncExecutor Execution")
executor = AsyncExecutor(max_workers=50)
pipeline = Pipeline(
    nodes=[native_async_fn],
    backend=HypernodesEngine(map_executor=executor),
)

overall_start = time.perf_counter()
results = pipeline.map(inputs={"delay": delays}, map_over="delay")
overall_time = time.perf_counter() - overall_start

print("\n📊 Timing Breakdown:")
print(f"   Overall time:          {overall_time:.3f}s")
print(f"   Number of items:       {len(delays)}")
print(f"   Number of submit():    {len(executor._submit_times)}")
print(f"   Total submit() time:   {sum(executor._submit_times):.3f}s")
print(f"   Avg submit() time:     {sum(executor._submit_times) / len(executor._submit_times) * 1000:.2f}ms")
print(f"   Max submit() time:     {max(executor._submit_times) * 1000:.2f}ms")
print(f"   Min submit() time:     {min(executor._submit_times) * 1000:.2f}ms")

# Calculate actual task execution time
actual_delays = [r["async_result"]["actual"] for r in results]
print("\n   Actual task times:")
print(f"   Avg task time:         {sum(actual_delays) / len(actual_delays) * 1000:.1f}ms")
print(f"   Expected concurrent:   ~{delays[0]:.3f}s")
print(f"   Actual concurrent:     {overall_time:.3f}s")

# Analyze overhead
submit_overhead = sum(executor._submit_times)
task_time = delays[0]  # Expected concurrent time
other_overhead = overall_time - submit_overhead - task_time

print("\n💡 Overhead Analysis:")
print(f"   Expected task time:    {task_time:.3f}s")
print(f"   Submit overhead:       {submit_overhead:.3f}s ({submit_overhead/overall_time*100:.1f}%)")
print(f"   Other overhead:        {other_overhead:.3f}s ({other_overhead/overall_time*100:.1f}%)")

if other_overhead > 0.05:
    print(f"\n⚠️  MAJOR OVERHEAD FOUND: {other_overhead * 1000:.1f}ms unaccounted for!")
    print("   Likely causes:")
    print("   • Pipeline setup/teardown")
    print("   • Result collection (future.result() calls)")
    print("   • Cache checking")
    print("   • Callback overhead")

executor.shutdown()
