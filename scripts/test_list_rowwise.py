import daft
from daft import DataType

# Test: Row-wise UDFs with lists
@daft.func
def make_list(text: str) -> list:
    return [1, 2, 3]

@daft.func
def process_list(lst: list) -> float:
    return sum(lst)

df = daft.from_pydict({"text": ["a", "b", "c"]})
df = df.with_column("mylist", make_list(df["text"]))
print("After first UDF:")
print(df.collect())
print("\nSchema:", df.schema())

df = df.with_column("sum", process_list(df["mylist"]))
print("\nAfter second UDF:")
print(df.collect())
