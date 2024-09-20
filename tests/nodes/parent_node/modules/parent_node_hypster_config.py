from hypster import HP

def my_config(hp: HP):
    from hypernodes import create_registry

    node_registry_path = hp.text_input("conf/node_registry.yaml")
    registry = create_registry(registry_path=node_registry_path)

    nested_node = registry.load("basic_usage")
    basic_usage_inputs = hp.propagate(nested_node.hypster_config, name="basic_usage")
    nested_node.set_instantiated_config(basic_usage_inputs)

    downstream_node = registry.mock("downstream")
    input = hp.text_input("testing")