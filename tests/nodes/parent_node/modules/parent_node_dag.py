from hypernodes import HyperNode
from hamilton.function_modifiers import config


def nested(nested_node: HyperNode) -> dict:
    nested_node.instantiate()
    return nested_node.execute()


def downstream(downstream_node: HyperNode, nested: dict, input: str) -> dict:
    downstream_node.instantiate()
    results = downstream_node.execute()
    return nested["query"] + " " + input
