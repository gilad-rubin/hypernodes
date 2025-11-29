"""Test edge coordinates after collapse using Playwright.

This test specifically verifies the Y coordinate discrepancy issue when
collapsing a pipeline.
"""

import tempfile

from playwright.sync_api import sync_playwright

from hypernodes import Pipeline, node
from hypernodes.viz import UIHandler
from hypernodes.viz.js.html_generator import generate_widget_html
from hypernodes.viz.js.renderer import JSRenderer


@node(output_name="query")
def extract_query(eval_pair: dict) -> str:
    return eval_pair.get("query", "")


@node(output_name="documents")
def retrieve(query: str, num_results: int = 5) -> list[dict[str, str]]:
    return [{"text": f"Doc {i}"} for i in range(num_results)]


@node(output_name="answer")
def generate_answer(query: str, documents: list[dict[str, str]], model_name: str) -> str:
    return f"Answer from {model_name}"


@node(output_name=("score", "details"))
def evaluate(answer: str, expected: str) -> tuple[float, dict[str, any]]:
    return 0.9, {"match": True}


def create_pipeline():
    """Create the RAG evaluation pipeline."""
    retrieval = Pipeline(nodes=[retrieve], name="retrieval")
    rag = Pipeline(
        nodes=[extract_query, retrieval.as_node(name="retrieval_step"), generate_answer],
        name="rag"
    )
    return Pipeline(
        nodes=[rag.as_node(name="rag_pipeline"), evaluate],
        name="evaluation"
    )


def generate_html(pipeline):
    """Generate HTML for the pipeline."""
    handler = UIHandler(pipeline, depth=99)
    graph_data = handler.get_visualization_data(traverse_collapsed=True)
    renderer = JSRenderer()
    rf_data = renderer.render(
        graph_data,
        theme="dark",
        separate_outputs=False,
        show_types=True,
    )
    return generate_widget_html(rf_data)


def run_test():
    """Run the coordinate test."""
    pipeline = create_pipeline()
    html = generate_html(pipeline)
    
    # Save HTML to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write(html)
        temp_path = f.name
    
    print(f"HTML saved to: {temp_path}")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            page.goto(f"file://{temp_path}")
            page.wait_for_selector('.react-flow__node', timeout=10000)
            page.wait_for_timeout(2000)  # Wait for layout
            
            # Get initial state
            print("\n=== INITIAL STATE (Expanded) ===")
            
            # Use the debug API to get accurate node positions
            debug_result = page.evaluate("() => HyperNodesVizState.debug.inspectLayout()")
            if debug_result:
                print(f"Nodes: {len(debug_result.get('nodes', []))}")
                
                rag = next((n for n in debug_result['nodes'] if n['id'] == 'rag_pipeline'), None)
                if rag:
                    print("\nrag_pipeline BEFORE collapse:")
                    print(f"  y={rag['y']}, h={rag['height']}, bottom={rag['bottom']}")
                
                # Also check generate_answer's position
                gen_ans = next((n for n in debug_result['nodes'] if n['id'] == 'rag_pipeline__generate_answer'), None)
                if gen_ans:
                    print("\ngenerate_answer BEFORE collapse:")
                    print(f"  y={gen_ans['y']}, h={gen_ans.get('height', 68)}, bottom={gen_ans['y'] + gen_ans.get('height', 68)}")
            
            # Click to collapse rag_pipeline
            print("\n=== COLLAPSING rag_pipeline ===")
            collapse_btn = page.locator('.react-flow__node[data-id="rag_pipeline"] button').first
            if collapse_btn:
                collapse_btn.click()
            else:
                page.click('.react-flow__node[data-id="rag_pipeline"]', position={"x": 50, "y": 10})
            
            page.wait_for_timeout(3000)  # Wait for collapse + layout
            
            # Use debug API to validate connections
            print("\n=== VALIDATING CONNECTIONS ===")
            validation = page.evaluate("() => HyperNodesVizState.debug.validateConnections()")
            
            if validation:
                print(f"Valid: {validation['valid']}")
                print(f"Summary: {validation['summary']}")
                
                if validation.get('issues'):
                    print("\nIssues found:")
                    for issue in validation['issues']:
                        print(f"  Edge: {issue['edge']}")
                        print(f"    Type: {issue['type']}")
                        print(f"    Issue: {issue['issue']}")
                        if 'expected' in issue:
                            print(f"    Expected: {issue['expected']}, Actual: {issue['actual']}")
                        print()
            
            # Also get detailed layout info
            print("\n=== COLLAPSED LAYOUT DETAILS ===")
            debug_after = page.evaluate("() => HyperNodesVizState.debug.inspectLayout()")
            if debug_after:
                print("Visible nodes:")
                for n in debug_after.get('nodes', []):
                    print(f"  {n['id']}: y={n['y']:.1f}, h={n['height']:.1f}, bottom={n['bottom']:.1f}")
                
                print("\nEdge paths from DOM:")
                for e in debug_after.get('edgePaths', []):
                    idx = e.get('index', 'N/A')
                    print(f"  #{idx}: ({e['startX']:.1f}, {e['startY']:.1f}) -> ({e['endX']:.1f}, {e['endY']:.1f})")
            
            # Check edge details including handles
            print("\n=== CHECKING REACT FLOW KEY ===")
            rf_key = page.evaluate("""() => {
                // The key is on the ReactFlow component, which we can't easily access
                // But we can check if edges were re-rendered by their data-testid containing expansion key
                const edges = document.querySelectorAll('.react-flow__edge');
                const edgeTestIds = Array.from(edges).map(e => e.getAttribute('data-testid'));
                return {
                    edgeCount: edges.length,
                    sampleEdgeTestId: edgeTestIds[0] || 'none'
                };
            }""")
            print(f"React Flow edges: {rf_key}")
            
            # Check expansion state
            exp_state = page.evaluate("() => HyperNodesVizState.debug.getExpansionState()")
            print(f"\nExpansion state: {exp_state}")
            
        finally:
            browser.close()


if __name__ == "__main__":
    run_test()
