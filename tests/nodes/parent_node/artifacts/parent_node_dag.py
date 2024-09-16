from hypernodes import HyperNode


def nested(nested_node : HyperNode) -> dict:
    nested_node.instantiate_inputs()
    return nested_node.execute()

def downstream(downstream_node : HyperNode, nested: dict, input: str) -> dict:
    downstream_node.instantiate_inputs()
    results = downstream_node.execute()
    return nested["query"] + " " + input
