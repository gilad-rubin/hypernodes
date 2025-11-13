"""Test script for Pipeline factory pattern with __new__ and @overload."""

from typing import overload, List, Dict, Any, Union, cast


# Mock Engine classes
class Engine:
    """Base engine class."""

    def run(self, pipeline, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": "base_engine"}


class HypernodesEngine(Engine):
    """Standard HyperNodes engine."""

    def run(self, pipeline, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": "hypernodes_engine"}


class DaftEngine(Engine):
    """Daft-specific engine."""

    def run(self, pipeline, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": "daft_engine"}

    def map_to_dataframe(self, pipeline, inputs: Dict[str, Any]) -> str:
        """Daft-specific method that returns a 'DataFrame'."""
        return "DaftDataFrame(result=mapped)"


# Pipeline classes
class Pipeline:
    """Base Pipeline class that acts as a factory."""

    @overload
    def __new__(cls, nodes: List[str], engine: DaftEngine) -> "DaftPipeline": ...

    @overload
    def __new__(cls, nodes: List[str], engine: HypernodesEngine) -> "HypernodesPipeline": ...

    def __new__(cls, nodes: List[str], engine: Engine):  # type: ignore[misc]
        """Factory that returns the appropriate subclass based on engine type."""
        # Only intercept direct Pipeline() calls
        if cls is Pipeline:
            if isinstance(engine, DaftEngine):
                return object.__new__(DaftPipeline)
            elif isinstance(engine, HypernodesEngine):
                return object.__new__(HypernodesPipeline)
            else:
                return object.__new__(BasePipeline)
        # For subclass instantiation, use normal flow
        return object.__new__(cls)

    def __init__(self, nodes: List[str], engine: Engine):
        self.nodes = nodes
        self.engine = engine

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Standard run method."""
        return self.engine.run(self, inputs)

    def map(self, inputs: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Standard map method - returns dict of lists."""
        return {"result": [1, 2, 3]}


class BasePipeline(Pipeline):
    """Basic pipeline implementation."""

    pass


class HypernodesPipeline(Pipeline):
    """HyperNodes-specific pipeline."""

    engine: HypernodesEngine


class DaftPipeline(Pipeline):
    """Daft-specific pipeline with additional methods."""

    engine: DaftEngine

    def map_to_dataframe(self, inputs: Dict[str, Any]) -> str:
        """Daft-specific method."""
        return self.engine.map_to_dataframe(self, inputs)


# Test the factory
if __name__ == "__main__":
    print("Testing Pipeline factory with __new__ and @overload\n")

    # Test 1: DaftEngine (cast for type safety)
    print("1. Creating Pipeline with DaftEngine:")
    daft_pipeline = cast(DaftPipeline, Pipeline(nodes=["node1"], engine=DaftEngine()))
    print(f"   Type: {type(daft_pipeline).__name__}")
    print(f"   Run result: {daft_pipeline.run({})}")
    print(f"   Map result: {daft_pipeline.map({})}")
    print(f"   map_to_dataframe: {daft_pipeline.map_to_dataframe({})}")  # Now type-safe!
    print()

    # Test 2: HypernodesEngine
    print("2. Creating Pipeline with HypernodesEngine:")
    hyper_pipeline = Pipeline(nodes=["node1"], engine=HypernodesEngine())
    print(f"   Type: {type(hyper_pipeline).__name__}")
    print(f"   Run result: {hyper_pipeline.run({})}")
    print()

    # Test 3: Base Engine
    print("3. Creating Pipeline with base Engine:")
    base_pipeline = Pipeline(nodes=["node1"], engine=Engine())
    print(f"   Type: {type(base_pipeline).__name__}")
    print(f"   Run result: {base_pipeline.run({})}")
    print()

    # Test 4: Type checking demonstration
    print("4. Type checking (IDE would catch this):")
    print("   ✅ daft_pipeline.map_to_dataframe() - exists on DaftPipeline")
    print("   ❌ hyper_pipeline.map_to_dataframe() - IDE/mypy would error")
    
    # This would error at type-check time but runs at runtime
    try:
        result = hyper_pipeline.map_to_dataframe({})  # type: ignore
        print(f"   Runtime: {result}")
    except AttributeError as e:
        print(f"   Runtime error (expected): {e}")
    
    print("\n✅ Factory pattern works! Pipeline() returns correct subclass.")

