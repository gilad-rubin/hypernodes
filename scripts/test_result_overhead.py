"""
Test to isolate where the 0.3s overhead comes from.
"""

import asyncio
import threading
import time
from concurrent.futures import Future


# Test function
async def async_io_operation(delay: float) -> dict:
    await asyncio.sleep(delay)
    return {"delay": delay}


delays = [0.1] * 50

print("=" * 70)
print("🔬 ISOLATING THE 0.3s OVERHEAD")
print("=" * 70)


# Simplified AsyncExecutor
class TestAsyncExecutor:
    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        self._loop = asyncio.new_event_loop()
        self._semaphore = None
        
        def run_loop():
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()
        
        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        
        # Initialize semaphore
        future = asyncio.run_coroutine_threadsafe(self._init_semaphore(), self._loop)
        future.result()
    
    async def _init_semaphore(self):
        self._semaphore = asyncio.Semaphore(self.max_workers)
    
    def submit(self, fn, *args, **kwargs) -> Future:
        async def run_with_semaphore():
            async with self._semaphore:
                return await fn(*args, **kwargs)
        
        return asyncio.run_coroutine_threadsafe(run_with_semaphore(), self._loop)
    
    def shutdown(self):
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=1)


# Test 1: Submit + immediately collect results (like HyperNodes does)
print("\n1️⃣  Submit all → Collect all (HyperNodes pattern)")
executor = TestAsyncExecutor(max_workers=50)

start = time.perf_counter()

# Submit phase
submit_start = time.perf_counter()
futures = []
for d in delays:
    future = executor.submit(async_io_operation, d)
    futures.append(future)
submit_time = time.perf_counter() - submit_start

# Collect phase (THIS might be the bottleneck!)
collect_start = time.perf_counter()
results = [f.result() for f in futures]
collect_time = time.perf_counter() - collect_start

total_time = time.perf_counter() - start

executor.shutdown()

print(f"   Submit phase:   {submit_time * 1000:.1f}ms")
print(f"   Collect phase:  {collect_time * 1000:.1f}ms")
print(f"   Total time:     {total_time:.3f}s")


# Test 2: Submit + wait a bit before collecting
print("\n2️⃣  Submit all → Wait 0.1s → Collect all")
executor2 = TestAsyncExecutor(max_workers=50)

start = time.perf_counter()

# Submit phase
submit_start = time.perf_counter()
futures2 = []
for d in delays:
    future = executor2.submit(async_io_operation, d)
    futures2.append(future)
submit_time2 = time.perf_counter() - submit_start

# Wait for tasks to complete
time.sleep(0.11)  # Just over the task time

# Collect phase
collect_start2 = time.perf_counter()
results2 = [f.result() for f in futures2]
collect_time2 = time.perf_counter() - collect_start2

total_time2 = time.perf_counter() - start

executor2.shutdown()

print(f"   Submit phase:   {submit_time2 * 1000:.1f}ms")
print("   Wait time:      110ms")
print(f"   Collect phase:  {collect_time2 * 1000:.1f}ms")
print(f"   Total time:     {total_time2:.3f}s")


# Test 3: Submit one, collect immediately (serial pattern)
print("\n3️⃣  Submit one → Collect immediately (serial)")
executor3 = TestAsyncExecutor(max_workers=50)

start = time.perf_counter()

results3 = []
for d in delays:
    future = executor3.submit(async_io_operation, d)
    result = future.result()  # Wait immediately
    results3.append(result)

total_time3 = time.perf_counter() - start

executor3.shutdown()

print(f"   Total time:     {total_time3:.3f}s")


print("\n" + "=" * 70)
print("📊 ANALYSIS")
print("=" * 70)
print(f"1. Submit all → Collect all:        {total_time:.3f}s")
print(f"   └─ Collect phase overhead:       {collect_time * 1000:.1f}ms")
print(f"2. Submit all → Wait → Collect:     {total_time2:.3f}s")
print(f"   └─ Collect phase overhead:       {collect_time2 * 1000:.1f}ms")
print(f"3. Submit → Collect (serial):       {total_time3:.3f}s")

print("\n💡 ROOT CAUSE:")
if collect_time > 0.1:
    print(f"   ❌ Collecting results (future.result()) takes {collect_time * 1000:.1f}ms!")
    print("   ❌ This is the bottleneck - calling result() on 50 futures sequentially")
    print("   ℹ️  Each result() call waits even though tasks are concurrent")
elif total_time > 0.15:
    print("   ❌ The overhead is somewhere else (not in result() calls)")
