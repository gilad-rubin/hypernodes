#!/usr/bin/env python3
"""
Generate actual HTML visualizations to debug label issues.
This script calls .visualize() like the user does, not just UIHandler internals.
"""

from hypernodes import Pipeline
from hypernodes.node import node
from dataclasses import dataclass


# ============================================================================
# Mock Data Models
# ============================================================================

@dataclass
class EvaluationPair:
    query: str
    ground_truth: str


# ============================================================================
# Mock Retrieval Pipeline (nested)
# ============================================================================

@node(output_name="embeddings")
def encode_query(query: str, model: str = "mock-model") -> list:
    """Mock encoding step"""
    return [0.1, 0.2, 0.3]


@node(output_name="retrieved_docs")
def retrieve(embeddings: list, index: str = "mock-index") -> list:
    """Mock retrieval step"""
    return [
        {"doc": "Document 1", "score": 0.95},
        {"doc": "Document 2", "score": 0.87},
    ]


@node(output_name="reranked_docs")
def rerank(retrieved_docs: list, reranker: str = "mock-reranker") -> list:
    """Mock reranking step"""
    return retrieved_docs[:1]  # Return top 1


# Build retrieval pipeline
retrieval_pipeline = Pipeline(
    nodes=[encode_query, retrieve, rerank],
    name="retrieval"  # Give it a name!
).bind(model="text-embedding-3-small", index="faiss-index", reranker="cross-encoder")


# ============================================================================
# Mock Evaluation Pipeline (contains nested retrieval)
# ============================================================================

@node(output_name="eval_pair")
def extract_eval_pair(eval_pair: EvaluationPair) -> EvaluationPair:
    """Pass through evaluation pair"""
    return eval_pair


@node(output_name="query")
def extract_query(eval_pair: EvaluationPair) -> str:
    """Extract query from evaluation pair"""
    return eval_pair.query


@node(output_name="ground_truth")
def extract_ground_truth(eval_pair: EvaluationPair) -> str:
    """Extract ground truth from evaluation pair"""
    return eval_pair.ground_truth


# Create retrieval node with mapping
retrieval_node = retrieval_pipeline.as_node(
    name="retrieval_step",  # Give it a name!
    input_mapping={"query": "query"},
    output_mapping={"reranked_docs": "context"}
)


@node(output_name="generated_answer")
def generate_answer(context: list, query: str, llm: str = "gpt-4") -> str:
    """Mock LLM generation"""
    return f"Generated answer for: {query}"


@node(output_name="evaluation_result")
def evaluate(generated_answer: str, ground_truth: str, evaluator: str = "mock-eval") -> dict:
    """Mock evaluation"""
    return {
        "score": 0.85,
        "passed": True,
        "explanation": "Good match"
    }


# Build evaluation pipeline
evaluation_pipeline = Pipeline(
    nodes=[
        extract_eval_pair,
        extract_query,
        extract_ground_truth,
        retrieval_node,
        generate_answer,
        evaluate
    ],
    name="evaluation"  # Give it a name!
).bind(llm="gpt-4o", evaluator="llm-evaluator")


# ============================================================================
# Mock Metrics Pipeline (contains nested evaluation)
# ============================================================================

@node(output_name="eval_pairs")
def pass_through_pairs(eval_pairs: list) -> list:
    """Pass through eval pairs"""
    return eval_pairs


# Create evaluation node with map_over
evaluation_node = evaluation_pipeline.as_node(
    name="batch_evaluation",  # Give it a name!
    map_over="eval_pairs",
    input_mapping={"eval_pairs": "eval_pair"},
    output_mapping={"evaluation_result": "results"}
)


@node(output_name="metrics")
def aggregate_metrics(results: list) -> dict:
    """Aggregate evaluation results"""
    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))
    avg_score = sum(r.get("score", 0) for r in results) / total if total > 0 else 0
    
    return {
        "total": total,
        "passed": passed,
        "accuracy": passed / total if total > 0 else 0,
        "average_score": avg_score
    }


# Build metrics pipeline
metrics_pipeline = Pipeline(
    nodes=[
        pass_through_pairs,
        evaluation_node,
        aggregate_metrics
    ],
    name="metrics"  # Give it a name!
)


# ============================================================================
# Main: Generate visualizations
# ============================================================================

def main():
    import os
    os.makedirs("outputs", exist_ok=True)
    
    print("=" * 80)
    print("GENERATING ACTUAL VISUALIZATIONS (like .visualize() does)")
    print("=" * 80)
    print()
    
    # Generate static visualizations at different depths
    print("1. metrics_pipeline.visualize(depth=1)")
    try:
        result = metrics_pipeline.visualize(depth=1)
        # Properly extract HTML content
        if hasattr(result, 'data'):
            html_str = result.data
        elif hasattr(result, '_repr_html_'):
            html_str = result._repr_html_()
        else:
            html_str = str(result)
            
        with open("outputs/viz_test_depth1.html", "w") as f:
            f.write(html_str)
        print("   ✅ Saved to outputs/viz_test_depth1.html")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n2. metrics_pipeline.visualize(depth=2)")
    try:
        result = metrics_pipeline.visualize(depth=2)
        # Properly extract HTML content
        if hasattr(result, 'data'):
            html_str = result.data
        elif hasattr(result, '_repr_html_'):
            html_str = result._repr_html_()
        else:
            html_str = str(result)
            
        with open("outputs/viz_test_depth2.html", "w") as f:
            f.write(html_str)
        print("   ✅ Saved to outputs/viz_test_depth2.html")
        
        # Parse HTML to check for raw IDs
        print("   Checking for raw ID labels...")
        import re
        # Look for text elements containing only digits (potential raw IDs)
        # More aggressive pattern to catch IDs in various contexts
        id_patterns = [
            r'>\s*(\d{10,})\s*<',  # In text nodes
            r'<text[^>]*>(\d{10,})</text>',  # In SVG text elements
            r'<title>(\d{10,})</title>',  # In titles
        ]
        
        all_matches = []
        for pattern in id_patterns:
            matches = re.findall(pattern, html_str)
            all_matches.extend(matches)
        
        if all_matches:
            unique_ids = set(all_matches)
            print(f"   ⚠️  FOUND RAW IDS: {len(unique_ids)} unique IDs")
            for raw_id in sorted(unique_ids)[:5]:  # Show first 5
                print(f"      {raw_id}")
        else:
            print("   ✅ No raw IDs found")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n3. metrics_pipeline.visualize(depth=3)")
    try:
        result = metrics_pipeline.visualize(depth=3)
        # Properly extract HTML content
        if hasattr(result, 'data'):
            html_str = result.data
        elif hasattr(result, '_repr_html_'):
            html_str = result._repr_html_()
        else:
            html_str = str(result)
            
        with open("outputs/viz_test_depth3.html", "w") as f:
            f.write(html_str)
        print("   ✅ Saved to outputs/viz_test_depth3.html")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Also output UI handler structure for comparison
    print("\n" + "=" * 80)
    print("EXTRACTING UI HANDLER STRUCTURE")
    print("=" * 80)
    
    from hypernodes.viz.ui_handler import UIHandler
    import json
    
    for depth in [1, 2, 3]:
        print(f"\nDepth={depth}:")
        try:
            handler = UIHandler(metrics_pipeline, depth=depth)
            viz_data = handler.get_visualization_data()
            
            # Convert to dict
            def node_to_dict(n):
                base = {"id": n.id, "type": type(n).__name__}
                if hasattr(n, "label"):
                    base["label"] = n.label
                if hasattr(n, "name"):
                    base["name"] = n.name
                if hasattr(n, "function_name"):
                    base["function_name"] = n.function_name
                if hasattr(n, "is_expanded"):
                    base["is_expanded"] = n.is_expanded
                return base
            
            structure = {
                "depth": depth,
                "nodes": [node_to_dict(n) for n in viz_data.nodes],
                "edges": [{"source": e.source, "target": e.target, "label": e.label} 
                         for e in viz_data.edges],
            }
            
            output_file = f"outputs/ui_structure_depth{depth}.json"
            with open(output_file, "w") as f:
                json.dump(structure, f, indent=2)
            
            print(f"   ✅ Saved structure to {output_file}")
            print(f"   Total nodes: {len(structure['nodes'])}")
            print(f"   Total edges: {len(structure['edges'])}")
            
            # Show any PipelineNodes
            pipeline_nodes = [n for n in structure['nodes'] if n['type'] == 'PipelineNode']
            if pipeline_nodes:
                print(f"   PipelineNodes:")
                for pn in pipeline_nodes:
                    print(f"      - {pn.get('label', 'NO LABEL')} (expanded={pn.get('is_expanded', False)})")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 80)
    print("Open the HTML files in a browser to inspect visually!")
    print("=" * 80)


if __name__ == "__main__":
    main()

