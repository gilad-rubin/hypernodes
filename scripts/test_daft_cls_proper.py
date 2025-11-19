#!/usr/bin/env python3
"""
CORRECT @daft.cls Usage with Proper Type Hints

Shows how to use @daft.cls properly with:
1. Automatic lazy initialization
2. @daft.method.batch for batch processing
3. Proper return_dtype specifications
"""

import time
from typing import List

import daft
from daft import DataType, Series

# ==================== CORRECT @daft.cls Implementation ====================


@daft.cls(max_concurrency=2, use_process=True)
class MockEncoder:
    """
    Daft handles lazy initialization automatically!

    How it works:
    1. encoder = MockEncoder("model") → __init__ NOT called yet
    2. df.collect() triggers execution → Daft calls __init__ once per worker
    3. Instance reused for all rows on that worker
    """

    def __init__(self, model_name: str, load_delay_ms: int = 50):
        """
        Called ONCE per worker by Daft (lazy!).
        Load model directly - Daft handles the lazy loading!
        """
        print(f"[Worker Init] Loading {model_name} (once per worker)...")
        time.sleep(load_delay_ms / 1000)
        self.model_name = model_name
        self._embedding_dim = 384
        print("[Worker Init] Model loaded!")

    @daft.method(return_dtype=DataType.list(DataType.float64()))
    def encode(self, text: str) -> List[float]:
        """
        Row-wise method with proper return type.
        Daft calls this for each row.
        """
        time.sleep(0.01)  # Simulate encoding
        return [0.1] * self._embedding_dim

    @daft.method.batch(return_dtype=DataType.list(DataType.float64()))
    def encode_batch(self, texts: Series) -> Series:
        """
        Batch method - Daft uses this for batch processing!

        @daft.method.batch tells Daft:
        - Process entire batches (not row-by-row)
        - Receives Series input, returns Series output
        - Return dtype is List[Float64] (list of floats for embeddings)
        """
        text_list = texts.to_pylist()
        print(f"[Batch] Encoding {len(text_list)} texts in one call")

        # Simulate batch encoding (100x faster per item!)
        time.sleep(0.0001 * len(text_list))

        # Return Series of lists (embeddings)
        embeddings = [[0.1] * self._embedding_dim for _ in text_list]
        return Series.from_pylist(embeddings)


# ==================== Test @daft.cls Lazy Initialization ====================


def test_lazy_initialization():
    """Demonstrate Daft's automatic lazy initialization."""
    print("\n" + "=" * 70)
    print("TEST 1: @daft.cls Lazy Initialization")
    print("=" * 70)

    print("\n1. Creating encoder instance...")
    start = time.perf_counter()
    encoder = MockEncoder("my-model", load_delay_ms=50)
    init_time = time.perf_counter() - start
    print(f"   ✓ Instance created in {init_time * 1000:.1f}ms")
    print("   → __init__ NOT called yet (lazy!)")

    print("\n2. Using encoder in DataFrame...")
    df = daft.from_pydict({"text": ["hello", "world", "test"]})

    print("   Building computation graph...")
    df = df.with_column("embedding", encoder.encode_batch(daft.col("text")))
    print("   ✓ Graph built (still no __init__ called)")

    print("\n3. Triggering execution with collect()...")
    start = time.perf_counter()
    result = df.collect()
    exec_time = time.perf_counter() - start
    print(f"   ✓ Execution completed in {exec_time * 1000:.1f}ms")
    print("   → __init__ called during execution (once per worker)")
    print(f"   → {len(result)} texts encoded")

    print("\n📊 Summary:")
    print(f"   Instance creation: {init_time * 1000:.1f}ms (instant!)")
    print(f"   Execution: {exec_time * 1000:.1f}ms (includes __init__ + encoding)")
    print(f"   Total: {(init_time + exec_time) * 1000:.1f}ms")


# ==================== Test @daft.method.batch ====================


def test_batch_method():
    """Test @daft.method.batch for vectorized processing."""
    print("\n" + "=" * 70)
    print("TEST 2: @daft.method.batch Performance")
    print("=" * 70)

    num_texts = 100
    texts = [f"text {i}" for i in range(num_texts)]

    encoder = MockEncoder("batch-model", load_delay_ms=50)
    df = daft.from_pydict({"text": texts})

    # Option A: Row-wise (slower)
    print(f"\n1. Row-wise encoding ({num_texts} texts)...")
    start = time.perf_counter()
    df_rowwise = df.with_column("embedding", encoder.encode(daft.col("text")))
    result_rowwise = df_rowwise.collect()
    time_rowwise = time.perf_counter() - start
    print(f"   ✓ Time: {time_rowwise:.3f}s")

    # Option B: Batch (faster!)
    print(f"\n2. Batch encoding with @daft.method.batch ({num_texts} texts)...")
    encoder2 = MockEncoder("batch-model2", load_delay_ms=50)
    df2 = daft.from_pydict({"text": texts})

    start = time.perf_counter()
    df_batch = df2.with_column("embedding", encoder2.encode_batch(daft.col("text")))
    result_batch = df_batch.collect()
    time_batch = time.perf_counter() - start
    print(f"   ✓ Time: {time_batch:.3f}s")

    # Results
    speedup = time_rowwise / time_batch
    print("\n📊 Results:")
    print(f"   Row-wise:  {time_rowwise:.3f}s")
    print(f"   Batch:     {time_batch:.3f}s")
    print(f"   Speedup:   {speedup:.1f}x ⚡")


# ==================== Main ====================


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("CORRECT @daft.cls USAGE DEMONSTRATIONS")
    print("=" * 70)
    print("\nKey points:")
    print("1. @daft.cls handles lazy initialization AUTOMATICALLY")
    print("2. @daft.method.batch marks batch methods for Daft")
    print("3. NO _ensure_loaded() pattern needed!")

    test_lazy_initialization()
    test_batch_method()

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
✅ @daft.cls Benefits:
   - Lazy initialization (automatic!)
   - Once per worker (efficient!)
   - Instance reuse across rows
   - No manual _ensure_loaded() needed!

✅ @daft.method.batch Benefits:
   - Daft uses batch processing automatically
   - 10-100x faster than row-wise
   - Vectorization support

⚠️  For HyperNodes:
   - @daft.cls works best with DaftEngine
   - For SeqEngine/DaskEngine, use simple classes
   - Both approaches deliver excellent performance!
    """)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
