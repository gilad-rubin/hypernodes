import unittest

from hypernodes import Pipeline, node
from hypernodes.sequential_engine import SeqEngine


# Test nodes
@node(output_name="doubled")
def double_node(x: int) -> int:
    return x * 2


@node(output_name="added")
def add_node(doubled: int, y: int) -> int:
    return doubled + y


@node(output_name="result")
def result_node(added: int) -> int:
    return added


class TestSeqEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SeqEngine()
        self.pipeline = Pipeline(
            nodes=[double_node, add_node, result_node], engine=self.engine
        )

    def test_run(self):
        inputs = {"x": 5, "y": 3}
        # 5 * 2 = 10
        # 10 + 3 = 13
        result = self.pipeline.run(inputs)
        self.assertEqual(result["result"], 13)
        self.assertEqual(result["doubled"], 10)
        self.assertEqual(result["added"], 13)

    def test_map_zip(self):
        # map over x, fixed y
        inputs = {"x": [1, 2, 3], "y": 10}
        # 1*2+10=12, 2*2+10=14, 3*2+10=16
        results = self.pipeline.map(inputs, map_over="x", map_mode="zip")
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["result"], 12)
        self.assertEqual(results[1]["result"], 14)
        self.assertEqual(results[2]["result"], 16)

    def test_map_product(self):
        # map over x and y
        inputs = {"x": [1, 2], "y": [10, 20]}
        # x=1, y=10 -> 12
        # x=1, y=20 -> 22
        # x=2, y=10 -> 14
        # x=2, y=20 -> 24
        results = self.pipeline.map(inputs, map_over=["x", "y"], map_mode="product")
        self.assertEqual(len(results), 4)
        # Order depends on implementation, but let's check values
        values = sorted([r["result"] for r in results])
        self.assertEqual(values, [12, 14, 22, 24])


if __name__ == "__main__":
    unittest.main()
