from hypster import HP

def my_config(hp: HP):
    from hypernodes import NodeRegistry
    registry = NodeRegistry()
    nested_node = registry.load("basic_usage")
    downstream_node = registry.mock("downstream")

    input = hp.text_input("testing")