
I have implemented the **Strict Input / Relaxed Output PyArrow Contract** for DualNode batch execution.

### Summary of Changes

1.  **Strict Input Contract:** `SeqEngine` now automatically converts list inputs to `pyarrow.Array` before calling a `DualNode` batch function. This ensures your batch logic (e.g., `pc.multiply`) works identically in tests (`SeqEngine`) and production (`DaftEngine`).

2.  **Relaxed Output Contract:** `SeqEngine` accepts `pyarrow.Array`, `list`, or `numpy.ndarray` as return values from your batch function. It automatically converts them back to lists for downstream sequential nodes. This aligns with Daft's flexibility.

3.  **Optional Dependency:** `pyarrow` is now an optional dependency (`hypernodes[batch]`).
    *   If you use a `DualNode` with a batch function but don't have `pyarrow` installed, `SeqEngine` raises a helpful `ImportError` instructing you to run `uv add pyarrow`.

4.  **Verified:** Tests pass for:
    *   Success case: Batch function using `pyarrow.compute` works with `SeqEngine`.
    *   Relaxed case: Batch function returning a `list` works with `SeqEngine`.
    *   Missing dependency case: Proper error message raised.

### Updated Documentation
*   **README.md**: Updated "Dual Execution" example to show the PyArrow pattern and installation command.
*   **New Guide**: `guides/batch_functions_pyarrow.md` explains the contract, examples, and best practices.
*   **DualNode Docstrings**: Updated with PyArrow examples and contract details.

You can now rely on `DualNode` batch functions receiving `pa.Array` inputs without writing manual conversion boilerplate in your tests!

