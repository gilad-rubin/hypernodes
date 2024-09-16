from hypster import HP

def my_config(hp: HP):
    from hypernodes import NodeRegistry
    node_registry_path = hp.text_input("conf/node_registry.yaml")
    registry = NodeRegistry(registry_path=node_registry_path)

    nested_node = registry.load("basic_usage")
    nested_node._instantiated_inputs = hp.propagate(nested_node.hypster_config, 
                                                    name="basic_usage")

    downstream_node = registry.mock("downstream")
    input = hp.text_input("testing")