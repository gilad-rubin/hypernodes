import daft
from daft import DataType, Series

# Test: Can we create a column of lists and process it?
@daft.func.batch(return_dtype=DataType.python())
def make_list(texts: Series) -> Series:
    result = [[1, 2, 3] for _ in texts.to_pylist()]
    return Series.from_pylist(result)

@daft.func.batch(return_dtype=DataType.python())
def process_list(lists: Series) -> Series:
    result = [sum(lst) for lst in lists.to_pylist()]
    return Series.from_pylist(result)

df = daft.from_pydict({"text": ["a", "b", "c"]})
df = df.with_column("mylist", make_list(df["text"]))
print("After first UDF:")
print(df.collect())

df = df.with_column("sum", process_list(df["mylist"]))
print("\nAfter second UDF:")
print(df.collect())
