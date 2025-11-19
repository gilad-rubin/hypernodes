
import daft
import pyarrow as pa
import numpy as np

try:
    s = daft.Series.from_pylist([1, 2, 3])
    print(f"Daft Series * 2: {s * 2}")
except Exception as e:
    print(f"Daft Series * 2 failed: {e}")

try:
    a = pa.array([1, 2, 3])
    print(f"PyArrow Array * 2: {a * 2}")
except Exception as e:
    print(f"PyArrow Array * 2 failed: {e}")

try:
    l = [1, 2, 3]
    print(f"List * 2: {l * 2}")
except Exception as e:
    print(f"List * 2 failed: {e}")

