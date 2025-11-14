#!/usr/bin/env python3
"""
Test script to understand Daft's batch processing with Series.
"""

import time

import daft
from daft import DataType, Series


# Test 1: Row-wise UDF (default)
@daft.func
def add_one_rowwise(x: int) -> int:
    """Process one row at a time."""
    return x + 1


# Test 2: Batch UDF with Series
@daft.func.batch(return_dtype=DataType.int64())
def add_one_batch(x: Series) -> Series:
    """Process entire batch at once."""
    # Convert to PyArrow, do computation, return as Series
    import pyarrow.compute as pc

    x_arrow = x.to_arrow()
    result = pc.add(x_arrow, 1)
    return result


# Test 3: Batch UDF with Python list
@daft.func.batch(return_dtype=DataType.int64())
def add_one_batch_pylist(x: Series) -> Series:
    """Process batch using Python lists."""
    pylist = x.to_pylist()
    result_list = [val + 1 for val in pylist]
    return Series.from_pylist(result_list)


# Test 4: Class-based UDF with @daft.cls
@daft.cls
class Adder:
    def __init__(self, increment: int):
        print(f"Initializing Adder with increment={increment}")
        self.increment = increment

    @daft.method(return_dtype=DataType.int64())
    def add(self, x: int) -> int:
        """Row-wise method."""
        return x + self.increment

    @daft.method.batch(return_dtype=DataType.int64())
    def add_batch(self, x: Series) -> Series:
        """Batch method."""
        import pyarrow.compute as pc

        x_arrow = x.to_arrow()
        result = pc.add(x_arrow, self.increment)
        return result


def main():
    print("=" * 60)
    print("Testing Daft Batch UDF with Series")
    print("=" * 60)

    # Create test data
    n = 1000
    df = daft.from_pydict({"x": list(range(n))})

    print(f"\nTest data: {n} rows")
    print("\n" + "=" * 60)

    # Test 1: Row-wise
    print("\n1. Row-wise UDF (@daft.func)")
    start = time.time()
    result1 = df.select(add_one_rowwise(df["x"])).collect()
    elapsed1 = time.time() - start
    print(f"   Time: {elapsed1:.4f}s")

    # Test 2: Batch with PyArrow
    print("\n2. Batch UDF with PyArrow (@daft.func.batch)")
    start = time.time()
    result2 = df.select(add_one_batch(df["x"])).collect()
    elapsed2 = time.time() - start
    print(f"   Time: {elapsed2:.4f}s")
    print(f"   Speedup: {elapsed1 / elapsed2:.1f}x")

    # Test 3: Batch with Python list
    print("\n3. Batch UDF with Python list (@daft.func.batch)")
    start = time.time()
    result3 = df.select(add_one_batch_pylist(df["x"])).collect()
    elapsed3 = time.time() - start
    print(f"   Time: {elapsed3:.4f}s")
    print(f"   Speedup vs row-wise: {elapsed1 / elapsed3:.1f}x")

    # Test 4: Class-based row-wise
    print("\n4. Class-based row-wise (@daft.cls + @daft.method)")
    adder = Adder(increment=1)
    start = time.time()
    result4 = df.select(adder.add(df["x"])).collect()
    elapsed4 = time.time() - start
    print(f"   Time: {elapsed4:.4f}s")

    # Test 5: Class-based batch
    print("\n5. Class-based batch (@daft.cls + @daft.method.batch)")
    start = time.time()
    result5 = df.select(adder.add_batch(df["x"])).collect()
    elapsed5 = time.time() - start
    print(f"   Time: {elapsed5:.4f}s")
    print(f"   Speedup vs class row-wise: {elapsed4 / elapsed5:.1f}x")

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Row-wise:          {elapsed1:.4f}s (1.0x)")
    print(f"  Batch (PyArrow):   {elapsed2:.4f}s ({elapsed1 / elapsed2:.1f}x)")
    print(f"  Batch (Python):    {elapsed3:.4f}s ({elapsed1 / elapsed3:.1f}x)")
    print(f"  Class row-wise:    {elapsed4:.4f}s ({elapsed1 / elapsed4:.1f}x)")
    print(f"  Class batch:       {elapsed5:.4f}s ({elapsed1 / elapsed5:.1f}x)")
    print("=" * 60)

    # Verify results are the same
    print("\nVerifying correctness...")
    r1 = result1.to_pydict()["x"]
    r2 = result2.to_pydict()["x"]
    r3 = result3.to_pydict()["x"]
    r4 = result4.to_pydict()["x"]
    r5 = result5.to_pydict()["x"]

    assert r1 == r2 == r3 == r4 == r5, "Results don't match!"
    print("✅ All results match!")


if __name__ == "__main__":
    main()
