"""Test to reproduce the CWD-based serialization issue locally (without Modal).

This test simulates the issue where classes defined in scripts have
module names that aren't importable on workers.
"""

import pytest
import sys
import types

# Skip all tests if daft is not available
pytest.importorskip("daft")

from typing import List, Dict
from pydantic import BaseModel
from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine


def test_script_pattern_with_unimportable_module():
    """Test script classes with non-importable module names."""
    
    # Define classes that simulate script-defined types
    class Passage(BaseModel):
        uuid: str
        text: str
        model_config = {"frozen": True}
    
    class Prediction(BaseModel):
        query_uuid: str
        paragraph_uuid: str
        score: float
        model_config = {"frozen": True}
    
    class RecallEvaluator:
        """Intentionally no __daft_hint__."""
        
        def __init__(self, k_list: List[int]):
            self.k_list = k_list
        
        def compute(self, predictions: List[Prediction]) -> Dict[str, float]:
            metrics = {}
            for k in self.k_list:
                metrics[f"recall@{k}"] = min(len(predictions), k) / max(1, k)
            return metrics
    
    # Simulate classes being from a non-importable script module
    # This is what happens when Modal runs "scripts/test_modal_failure_repro.py"
    fake_module_name = "test_modal_failure_repro"
    
    # Set the module name to simulate the script
    Passage.__module__ = fake_module_name
    Prediction.__module__ = fake_module_name
    RecallEvaluator.__module__ = fake_module_name
    
    # Create fake module in sys.modules so the classes are "available" locally
    # (but they won't be on remote workers)
    fake_module = types.ModuleType(fake_module_name)
    fake_module.Passage = Passage
    fake_module.Prediction = Prediction
    fake_module.RecallEvaluator = RecallEvaluator
    sys.modules[fake_module_name] = fake_module
    
    try:
        # Define nodes that use these classes
        @node(output_name="passages")
        def create_passages() -> List[Passage]:
            return [
                Passage(uuid=f"p{i}", text=f"passage {i}")
                for i in range(3)
            ]
        
        @node(output_name="predictions")
        def create_predictions(passages: List[Passage]) -> List[Prediction]:
            return [
                Prediction(
                    query_uuid="q1",
                    paragraph_uuid=p.uuid,
                    score=float(len(p.text))
                )
                for p in passages
            ]
        
        @node(output_name="metrics")
        def compute_metrics(
            predictions: List[Prediction],
            evaluator: RecallEvaluator
        ) -> Dict[str, float]:
            return evaluator.compute(predictions)
        
        # Create evaluator
        evaluator = RecallEvaluator(k_list=[1, 2, 3])
        
        # Run with DaftEngine in debug mode
        pipeline = Pipeline(
            nodes=[create_passages, create_predictions, compute_metrics],
            engine=DaftEngine(debug=True)
        )
        
        print("\n" + "=" * 80)
        print("RUNNING PIPELINE WITH DEBUG MODE")
        print("=" * 80)
        
        result = pipeline.run(
            inputs={"evaluator": evaluator},
            output_name="metrics"
        )
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Result: {result}")
        
        # Verify result
        assert "metrics" in result
        assert "recall@1" in result["metrics"]
        assert "recall@2" in result["metrics"]
        assert "recall@3" in result["metrics"]
        
    finally:
        # Clean up sys.modules
        sys.modules.pop(fake_module_name, None)


def test_pydantic_output_from_script_module():
    """Test that Pydantic models returned from nodes get fixed."""
    
    class Document(BaseModel):
        id: str
        content: str
        model_config = {"frozen": True}
    
    # Simulate non-importable module
    Document.__module__ = "unimportable_script"
    
    # Add to sys.modules locally (but won't exist on workers)
    fake_module = types.ModuleType("unimportable_script")
    fake_module.Document = Document
    sys.modules["unimportable_script"] = fake_module
    
    try:
        @node(output_name="doc")
        def create_doc() -> Document:
            return Document(id="doc1", content="test content")
        
        @node(output_name="content")
        def extract_content(doc: Document) -> str:
            return doc.content
        
        pipeline = Pipeline(
            nodes=[create_doc, extract_content],
            engine=DaftEngine(debug=True)
        )
        
        print("\n" + "=" * 80)
        print("TEST: Pydantic Output Fixing")
        print("=" * 80)
        
        result = pipeline.run(inputs={})
        
        print("Result:", result)
        assert result["content"] == "test content"
        
    finally:
        sys.modules.pop("unimportable_script", None)


def test_nested_pydantic_models_from_script():
    """Test nested Pydantic models that reference each other."""
    
    class Author(BaseModel):
        name: str
        model_config = {"frozen": True}
    
    class Article(BaseModel):
        title: str
        author: Author
        model_config = {"frozen": True}
    
    # Both from same fake script
    script_name = "fake_models_script"
    Author.__module__ = script_name
    Article.__module__ = script_name
    
    fake_module = types.ModuleType(script_name)
    fake_module.Author = Author
    fake_module.Article = Article
    sys.modules[script_name] = fake_module
    
    try:
        @node(output_name="article")
        def create_article() -> Article:
            author = Author(name="John Doe")
            return Article(title="Test Article", author=author)
        
        @node(output_name="author_name")
        def get_author_name(article: Article) -> str:
            return article.author.name
        
        pipeline = Pipeline(
            nodes=[create_article, get_author_name],
            engine=DaftEngine(debug=True)
        )
        
        print("\n" + "=" * 80)
        print("TEST: Nested Pydantic Models")
        print("=" * 80)
        
        result = pipeline.run(inputs={})
        
        print("Result:", result)
        assert result["author_name"] == "John Doe"
        
    finally:
        sys.modules.pop(script_name, None)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
