"""
Test the exact pattern HyperNodes uses: submit() for each item individually.
"""

import asyncio
import threading
import time
from concurrent.futures import Future


# Test function
async def async_io_operation(delay: float) -> dict:
    await asyncio.sleep(delay)
    return {"delay": delay, "completed": time.time()}


delays = [0.1] * 50

print("=" * 70)
print("🔬 TESTING HYPERNODES PATTERN: submit() for each item")
print("=" * 70)

# Test 1: Pure asyncio.gather (baseline)
print("\n1️⃣  Pure asyncio.gather (baseline)")


async def run_pure():
    return await asyncio.gather(*[async_io_operation(d) for d in delays])


start = time.time()
results_1 = asyncio.run(run_pure())
time_1 = time.time() - start
print(f"   Time: {time_1:.3f}s")


# Test 2: Mimicking HyperNodes AsyncExecutor's submit() pattern
print("\n2️⃣  AsyncExecutor with submit() per item (HyperNodes pattern)")


class TestAsyncExecutor:
    """Simplified version of AsyncExecutor."""
    
    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        self._loop = None
        self._thread = None
        self._semaphore = None
        self._start_loop()
    
    def _start_loop(self):
        """Start event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        
        def run_loop():
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()
        
        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        
        # Initialize semaphore
        future = asyncio.run_coroutine_threadsafe(self._init_semaphore(), self._loop)
        future.result()
    
    async def _init_semaphore(self):
        """Initialize semaphore in the event loop."""
        self._semaphore = asyncio.Semaphore(self.max_workers)
    
    def submit(self, fn, *args, **kwargs) -> Future:
        """Execute function concurrently using asyncio."""
        async def run_with_semaphore():
            async with self._semaphore:
                if asyncio.iscoroutinefunction(fn):
                    return await fn(*args, **kwargs)
                else:
                    return await self._loop.run_in_executor(None, lambda: fn(*args, **kwargs))
        
        # This is called for EACH item!
        future = asyncio.run_coroutine_threadsafe(run_with_semaphore(), self._loop)
        return future
    
    def shutdown(self):
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=1)


executor = TestAsyncExecutor(max_workers=50)

start = time.time()
futures = []
for d in delays:
    future = executor.submit(async_io_operation, d)
    futures.append(future)

# Wait for all results (this is what HyperNodes does)
results_2 = [f.result() for f in futures]
time_2 = time.time() - start

executor.shutdown()

print(f"   Time: {time_2:.3f}s")


# Test 3: Batched approach (what it SHOULD do)
print("\n3️⃣  AsyncExecutor with batched submit (optimized)")


class OptimizedAsyncExecutor:
    """Optimized version that batches tasks."""
    
    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        self._loop = None
        self._thread = None
        self._semaphore = None
        self._start_loop()
    
    def _start_loop(self):
        """Start event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        
        def run_loop():
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()
        
        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        
        # Initialize semaphore
        future = asyncio.run_coroutine_threadsafe(self._init_semaphore(), self._loop)
        future.result()
    
    async def _init_semaphore(self):
        """Initialize semaphore in the event loop."""
        self._semaphore = asyncio.Semaphore(self.max_workers)
    
    def submit_batch(self, fn, args_list) -> Future:
        """Execute multiple function calls in a batch."""
        async def run_batch():
            async def run_one(args):
                async with self._semaphore:
                    if asyncio.iscoroutinefunction(fn):
                        return await fn(*args)
                    else:
                        return await self._loop.run_in_executor(None, lambda: fn(*args))
            
            return await asyncio.gather(*[run_one(args) for args in args_list])
        
        # Single run_coroutine_threadsafe call for all items!
        future = asyncio.run_coroutine_threadsafe(run_batch(), self._loop)
        return future
    
    def shutdown(self):
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=1)


executor_opt = OptimizedAsyncExecutor(max_workers=50)

start = time.time()
args_list = [(d,) for d in delays]
future = executor_opt.submit_batch(async_io_operation, args_list)
results_3 = future.result()
time_3 = time.time() - start

executor_opt.shutdown()

print(f"   Time: {time_3:.3f}s")


print("\n" + "=" * 70)
print("📊 ANALYSIS")
print("=" * 70)
print(f"1. Pure asyncio.gather:                     {time_1:.3f}s (baseline)")
print(f"2. AsyncExecutor submit() per item:         {time_2:.3f}s ({time_2/time_1:.2f}x slower)")
print(f"3. AsyncExecutor with batched submit:       {time_3:.3f}s ({time_3/time_1:.2f}x slower)")

print("\n💡 ROOT CAUSE IDENTIFIED:")
print(f"   ❌ Calling submit() (run_coroutine_threadsafe) {len(delays)} times")
print(f"   ❌ Each call has ~{(time_2 - time_1) * 1000 / len(delays):.2f}ms overhead")
print(f"   ✅ Batching with a single call saves {(time_2 - time_3) * 1000:.1f}ms")
print(f"   ✅ Potential speedup: {time_2/time_3:.2f}x faster")
