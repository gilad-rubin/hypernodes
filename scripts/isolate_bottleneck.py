"""
Isolate the exact bottleneck in AsyncExecutor by testing each component.
"""

import asyncio
import threading
import time


# Test function
async def async_io_operation(delay: float) -> dict:
    await asyncio.sleep(delay)
    return {"delay": delay, "completed": time.time()}


delays = [0.1] * 50

print("=" * 70)
print("🔬 ISOLATING ASYNCEXECUTOR BOTTLENECK")
print("=" * 70)

# Test 1: Pure asyncio.gather (baseline)
print("\n1️⃣  Pure asyncio.gather (baseline)")


async def run_pure():
    return await asyncio.gather(*[async_io_operation(d) for d in delays])


start = time.time()
results_1 = asyncio.run(run_pure())
time_1 = time.time() - start
print(f"   Time: {time_1:.3f}s")

# Test 2: With Semaphore (like AsyncExecutor uses)
print("\n2️⃣  With Semaphore(50)")


async def run_with_semaphore():
    semaphore = asyncio.Semaphore(50)
    
    async def limited(d):
        async with semaphore:
            return await async_io_operation(d)
    
    return await asyncio.gather(*[limited(d) for d in delays])


start = time.time()
results_2 = asyncio.run(run_with_semaphore())
time_2 = time.time() - start
print(f"   Time: {time_2:.3f}s (+{(time_2 - time_1) * 1000:.1f}ms)")

# Test 3: With Semaphore + run_coroutine_threadsafe (like AsyncExecutor)
print("\n3️⃣  With Semaphore + run_coroutine_threadsafe")


def run_in_background_loop():
    """Mimics AsyncExecutor's background event loop approach."""
    loop = asyncio.new_event_loop()
    
    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()
    
    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    
    # Initialize semaphore
    async def init_semaphore():
        return asyncio.Semaphore(50)
    
    future = asyncio.run_coroutine_threadsafe(init_semaphore(), loop)
    semaphore = future.result()
    
    # Submit tasks
    async def run_one(d):
        async with semaphore:
            return await async_io_operation(d)
    
    futures = []
    for d in delays:
        future = asyncio.run_coroutine_threadsafe(run_one(d), loop)
        futures.append(future)
    
    # Wait for all results
    results = [f.result() for f in futures]
    
    # Cleanup
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=1)
    
    return results


start = time.time()
results_3 = run_in_background_loop()
time_3 = time.time() - start
print(f"   Time: {time_3:.3f}s (+{(time_3 - time_1) * 1000:.1f}ms)")

# Test 4: run_coroutine_threadsafe + gather (better approach)
print("\n4️⃣  With run_coroutine_threadsafe + gather (optimized)")


def run_in_background_loop_optimized():
    """Use gather instead of individual run_coroutine_threadsafe calls."""
    loop = asyncio.new_event_loop()
    
    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()
    
    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    
    # Submit all tasks as a single gather operation
    async def run_all():
        semaphore = asyncio.Semaphore(50)
        
        async def run_one(d):
            async with semaphore:
                return await async_io_operation(d)
        
        return await asyncio.gather(*[run_one(d) for d in delays])
    
    future = asyncio.run_coroutine_threadsafe(run_all(), loop)
    results = future.result()
    
    # Cleanup
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=1)
    
    return results


start = time.time()
results_4 = run_in_background_loop_optimized()
time_4 = time.time() - start
print(f"   Time: {time_4:.3f}s (+{(time_4 - time_1) * 1000:.1f}ms)")

print("\n" + "=" * 70)
print("📊 ANALYSIS")
print("=" * 70)
print(f"1. Pure asyncio.gather:                  {time_1:.3f}s (baseline)")
print(f"2. + Semaphore:                          {time_2:.3f}s ({time_2/time_1:.2f}x)")
print(f"3. + run_coroutine_threadsafe (current): {time_3:.3f}s ({time_3/time_1:.2f}x) ❌")
print(f"4. + run_coroutine_threadsafe + gather:  {time_4:.3f}s ({time_4/time_1:.2f}x)")

print("\n💡 ROOT CAUSE:")
if time_3 > time_4 * 1.5:
    print("   ❌ Calling run_coroutine_threadsafe for EACH task is the bottleneck!")
    print("   ✅ Solution: Batch tasks with gather INSIDE the background loop")
    print(f"   ✅ Potential speedup: {time_3/time_4:.2f}x faster")
else:
    print("   The overhead is distributed across all components")
