import asyncio
import time
import warnings

import pytest

from hypernodes import Pipeline, node, HypernodesEngine
from hypernodes.executors import AsyncExecutor
from concurrent.futures import ThreadPoolExecutor


@node(output_name="async_out")
async def _async_sleep(d: float) -> float:
    await asyncio.sleep(d)
    return d


@node(output_name="sync_out")
def _sync_sleep(d: float) -> float:
    time.sleep(d)
    return d


def test_async_native_faster_than_per_call():
    # Amplify difference with larger items/delay but still under ~0.6s
    delays = [0.05] * 50

    # per_call strategy
    eng_per = HypernodesEngine(map_executor=AsyncExecutor(max_workers=60), async_strategy="per_call")
    p_per = Pipeline(nodes=[_async_sleep], backend=eng_per)
    t0 = time.time(); p_per.map(inputs={"d": delays}, map_over="d"); per_time = time.time() - t0
    eng_per.shutdown()

    # async_native strategy
    eng_nat = HypernodesEngine(map_executor=AsyncExecutor(max_workers=60), async_strategy="async_native")
    p_nat = Pipeline(nodes=[_async_sleep], backend=eng_nat)
    t0 = time.time(); p_nat.map(inputs={"d": delays}, map_over="d"); nat_time = time.time() - t0
    eng_nat.shutdown()

    # per_call should not be faster than native; allow small variance
    assert per_time >= nat_time * 1.0


def test_async_auto_matches_pure_async_within_50_percent():
    delays = [0.02] * 30

    # Hypernodes auto async-native
    eng = HypernodesEngine(map_executor=AsyncExecutor(max_workers=60))
    p = Pipeline(nodes=[_async_sleep], backend=eng)
    t0 = time.time(); p.map(inputs={"d": delays}, map_over="d"); hn_time = time.time() - t0
    eng.shutdown()

    # Pure asyncio baseline
    async def run_pure():
        return await asyncio.gather(*[asyncio.sleep(d) for d in delays])

    t1 = time.time(); asyncio.run(run_pure()); pure_time = time.time() - t1

    # Allow some buffer for framework overhead
    assert hn_time <= pure_time * 1.5


def test_async_executor_sync_respects_max_workers():
    delays = [0.02] * 40

    eng_5 = HypernodesEngine(map_executor=AsyncExecutor(max_workers=5))
    p5 = Pipeline(nodes=[_sync_sleep], backend=eng_5)
    t0 = time.time(); p5.map(inputs={"d": delays}, map_over="d"); t5 = time.time() - t0
    eng_5.shutdown()

    eng_20 = HypernodesEngine(map_executor=AsyncExecutor(max_workers=20))
    p20 = Pipeline(nodes=[_sync_sleep], backend=eng_20)
    t0 = time.time(); p20.map(inputs={"d": delays}, map_over="d"); t20 = time.time() - t0
    eng_20.shutdown()

    assert t5 > t20 * 2.0


def test_async_vs_threaded_equivalence_for_sync_blocking():
    delays = [0.02] * 40

    eng_async = HypernodesEngine(map_executor=AsyncExecutor(max_workers=20))
    p_async = Pipeline(nodes=[_sync_sleep], backend=eng_async)
    t0 = time.time(); p_async.map(inputs={"d": delays}, map_over="d"); t_async = time.time() - t0
    eng_async.shutdown()

    eng_thread = HypernodesEngine(map_executor=ThreadPoolExecutor(max_workers=20))
    p_thread = Pipeline(nodes=[_sync_sleep], backend=eng_thread)
    t0 = time.time(); p_thread.map(inputs={"d": delays}, map_over="d"); t_thread = time.time() - t0
    eng_thread.shutdown()

    ratio = t_async / t_thread if t_thread > 0 else 1.0
    assert 0.7 <= ratio <= 1.3


def test_pipeline_run_inside_running_event_loop_not_error():
    # Call sync pipeline.run from inside an event loop; should offload safely
    async def main():
        eng = HypernodesEngine(map_executor=AsyncExecutor(max_workers=10))
        p = Pipeline(nodes=[_async_sleep], backend=eng)
        try:
            res = p.run(inputs={"d": 0.001})
            assert res["async_out"] == 0.001
        finally:
            eng.shutdown()

    asyncio.run(main())


def test_parallel_executor_reuse_and_no_worker_stop_warning():
    delays = [0.02] * 24
    p = Pipeline(nodes=[_sync_sleep], backend=HypernodesEngine(map_executor="parallel", max_workers=2,))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t0 = time.time(); p.map(inputs={"d": delays}, map_over="d"); first = time.time() - t0
        t0 = time.time(); p.map(inputs={"d": delays}, map_over="d"); second = time.time() - t0

    worker_warnings = [str(wi.message) for wi in w if "worker stopped" in str(wi.message).lower()]
    assert not worker_warnings
    assert second <= first


def test_code_hash_cached_not_recomputed(monkeypatch):
    # Count calls to hash_code across many map items
    import hypernodes.cache as cache_mod

    call_count = {"n": 0}
    _orig = cache_mod.hash_code

    def _wrapped(func):
        call_count["n"] += 1
        return _orig(func)

    monkeypatch.setattr(cache_mod, "hash_code", _wrapped)

    @node(output_name="out")
    def simple(x: int) -> int:
        return x + 1

    p = Pipeline(nodes=[simple])
    # Map over many items; hash_code should have been called only once at Node creation
    items = list(range(50))
    p.map(inputs={"x": items}, map_over="x")

    assert call_count["n"] == 1
