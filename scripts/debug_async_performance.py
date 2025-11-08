"""
Debug script to understand async vs threaded performance for I/O operations.
We'll test with progressively more HyperNodes code to isolate the bottleneck.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor


# Simple async I/O operation
async def async_io_operation(delay: float) -> dict:
    """Simulates async I/O like API calls"""
    start = time.time()
    await asyncio.sleep(delay)  # Non-blocking async sleep
    return {"delay": delay, "duration": time.time() - start}


# Sync I/O operation
def sync_io_operation(delay: float) -> dict:
    """Simulates sync I/O"""
    start = time.time()
    time.sleep(delay)  # Blocking sleep
    return {"delay": delay, "duration": time.time() - start}


print("=" * 70)
print("🧪 BASELINE: Pure asyncio vs ThreadPoolExecutor (no HyperNodes)")
print("=" * 70)

# Test parameters
delays = [0.1] * 50
num_items = len(delays)

# Test 1: Pure asyncio.gather
print("\n1️⃣  Pure asyncio.gather (baseline)")


async def run_pure_async():
    return await asyncio.gather(*[async_io_operation(d) for d in delays])


start = time.time()
results_pure_async = asyncio.run(run_pure_async())
time_pure_async = time.time() - start
print(f"   Time: {time_pure_async:.3f}s ({num_items * 0.1 / time_pure_async:.2f}x speedup)")

# Test 2: ThreadPoolExecutor with sync function
print("\n2️⃣  ThreadPoolExecutor with sync function")
start = time.time()
with ThreadPoolExecutor(max_workers=50) as executor:
    results_thread = list(executor.map(sync_io_operation, delays))
time_thread = time.time() - start
print(f"   Time: {time_thread:.3f}s ({num_items * 0.1 / time_thread:.2f}x speedup)")

# Test 3: asyncio with Semaphore (limiting concurrency)
print("\n3️⃣  asyncio with Semaphore (max_workers=50)")


async def async_with_semaphore(delays, max_workers):
    semaphore = asyncio.Semaphore(max_workers)
    
    async def limited_async_io(delay):
        async with semaphore:
            return await async_io_operation(delay)
    
    return await asyncio.gather(*[limited_async_io(d) for d in delays])


start = time.time()
results_semaphore = asyncio.run(async_with_semaphore(delays, 50))
time_semaphore = time.time() - start
print(f"   Time: {time_semaphore:.3f}s ({num_items * 0.1 / time_semaphore:.2f}x speedup)")

# Test 4: asyncio running sync functions via run_in_executor
print("\n4️⃣  asyncio.run_in_executor (sync function in async)")


async def async_executor_sync():
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, sync_io_operation, d) for d in delays]
    return await asyncio.gather(*tasks)


start = time.time()
results_executor = asyncio.run(async_executor_sync())
time_executor = time.time() - start
print(f"   Time: {time_executor:.3f}s ({num_items * 0.1 / time_executor:.2f}x speedup)")

# Test 5: asyncio with explicit ThreadPoolExecutor
print("\n5️⃣  asyncio.run_in_executor with explicit ThreadPoolExecutor(50)")


async def async_executor_explicit():
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=50) as executor:
        tasks = [loop.run_in_executor(executor, sync_io_operation, d) for d in delays]
        return await asyncio.gather(*tasks)


start = time.time()
results_explicit = asyncio.run(async_executor_explicit())
time_explicit = time.time() - start
print(f"   Time: {time_explicit:.3f}s ({num_items * 0.1 / time_explicit:.2f}x speedup)")

print("\n" + "=" * 70)
print("📊 SUMMARY")
print("=" * 70)
print(f"1. Pure asyncio.gather:              {time_pure_async:.3f}s (baseline)")
print(f"2. ThreadPoolExecutor (sync):        {time_thread:.3f}s")
print(f"3. asyncio + Semaphore:              {time_semaphore:.3f}s")
print(f"4. asyncio + run_in_executor(None):  {time_executor:.3f}s")
print(f"5. asyncio + explicit ThreadPool:    {time_explicit:.3f}s")

print("\n💡 Key Insights:")
print(f"   • Pure async is {time_thread / time_pure_async:.2f}x faster than threads")
print(f"   • Semaphore adds {(time_semaphore - time_pure_async) * 1000:.1f}ms overhead")
print(f"   • run_in_executor(None) is {time_executor / time_pure_async:.2f}x slower (limited default pool)")
print(f"   • Explicit ThreadPool is {time_explicit / time_pure_async:.2f}x slower than pure async")
