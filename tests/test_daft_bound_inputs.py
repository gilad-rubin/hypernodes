"""Test DaftEngine with bound inputs in nested pipelines."""

import pytest

from hypernodes import Pipeline, node


# Mock objects that should be treated as stateful
class MockVectorStore:
    """Mock vector store with stateful marker."""
    
    __hypernode_stateful__ = True
    
    def search(self, query: str, top_k: int = 3):
        """Mock search."""
        return [f"doc_{i}" for i in range(top_k)]


class MockLLM:
    """Mock LLM with stateful marker."""
    
    __hypernode_stateful__ = True
    
    def generate(self, prompt: str):
        """Mock generation."""
        return f"Generated: {prompt[:50]}"


@node(output_name="retrieved_docs")
def retrieve(query: str, vector_store: MockVectorStore, top_k: int = 2):
    """Retrieve relevant documents."""
    docs = vector_store.search(query, top_k=top_k)
    return docs


@node(output_name="answer")
def generate(query: str, retrieved_docs, llm: MockLLM):
    """Generate an answer using the retrieved documents."""
    context = "\n".join(retrieved_docs)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    response = llm.generate(prompt)
    return response


@node(output_name="query")
def extract_query(text: str):
    """Extract query from text."""
    return text.upper()


@node(output_name="result")
def format_result(answer: str):
    """Format the result."""
    return f"Result: {answer}"


@pytest.fixture
def retrieval_pipeline():
    """Create retrieval pipeline with bound inputs."""
    vector_store = MockVectorStore()
    llm = MockLLM()
    top_k = 3
    
    pipeline = Pipeline(nodes=[retrieve, generate], name="retrieval")
    return pipeline.bind(vector_store=vector_store, llm=llm, top_k=top_k)


@pytest.fixture
def evaluation_pipeline(retrieval_pipeline):
    """Create evaluation pipeline with nested bound retrieval pipeline."""
    retrieval_node = retrieval_pipeline.as_node()
    pipeline = Pipeline(
        nodes=[extract_query, retrieval_node, format_result],
        name="evaluation"
    )
    return pipeline


class TestDaftEngineWithBoundInputs:
    """Test DaftEngine with bound inputs in nested pipelines."""
    
    def test_single_run_with_bound_inputs(self, retrieval_pipeline):
        """Test single run with bound inputs works with DaftEngine."""
        pytest.importorskip("daft")
        from hypernodes.engines import DaftEngine
        
        engine = DaftEngine()
        pipeline = retrieval_pipeline.with_engine(engine)
        
        result = pipeline.run(inputs={"query": "What is Hypernodes?"})
        
        assert "answer" in result
        assert "Generated:" in result["answer"]
    
    def test_nested_pipeline_with_bound_inputs(self, evaluation_pipeline):
        """Test nested pipeline with bound inputs works with DaftEngine."""
        pytest.importorskip("daft")
        from hypernodes.engines import DaftEngine
        
        engine = DaftEngine()
        pipeline = evaluation_pipeline.with_engine(engine)
        
        result = pipeline.run(inputs={"text": "What is Hypernodes?"})
        
        assert "result" in result
        assert "Result:" in result["result"]
        assert "Generated:" in result["result"]
    
    def test_map_with_nested_bound_pipeline(self, evaluation_pipeline):
        """Test map operation with nested pipeline that has bound inputs."""
        pytest.importorskip("daft")
        from hypernodes.engines import DaftEngine
        
        engine = DaftEngine()
        pipeline = evaluation_pipeline.with_engine(engine)
        
        results = pipeline.map(
            inputs={"text": ["query 1", "query 2", "query 3"]},
            map_over="text"
        )
        
        assert len(results) == 3
        for result in results:
            assert "result" in result
            assert "Result:" in result["result"]
            assert "Generated:" in result["result"]
    
    def test_double_nested_with_map_over(self, retrieval_pipeline):
        """Test double nested pipeline with map_over that has bound inputs."""
        pytest.importorskip("daft")
        from hypernodes.engines import DaftEngine
        
        # Create evaluation pipeline with map_over
        evaluation_node = retrieval_pipeline.as_node(
            name="batch_retrieval",
            map_over="queries",
            input_mapping={"queries": "query"},
            output_mapping={"answer": "answers"}
        )
        
        @node(output_name="summary")
        def summarize_answers(answers):
            """Summarize answers."""
            return f"Summary of {len(answers)} answers"
        
        metrics_pipeline = Pipeline(
            nodes=[evaluation_node, summarize_answers],
            name="metrics"
        )
        
        engine = DaftEngine()
        pipeline = metrics_pipeline.with_engine(engine)
        
        result = pipeline.run(inputs={"queries": ["q1", "q2", "q3"]})
        
        assert "summary" in result
        assert "Summary of 3 answers" in result["summary"]

