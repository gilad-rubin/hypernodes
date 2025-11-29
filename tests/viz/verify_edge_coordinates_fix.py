#!/usr/bin/env python3
"""
Verify that edge coordinates are correct after collapsing pipelines.

This test specifically checks that when a pipeline collapses:
1. The collapsed node gets the correct height from ELK layout
2. The edge from the collapsed pipeline starts at the bottom of the node
3. The Y coordinate discrepancy is within acceptable tolerance (< 5px)
"""

import tempfile
import time

# Generate the test HTML
from hypernodes import Pipeline, node
from hypernodes.viz.js.html_generator import generate_widget_html
from hypernodes.viz.js.renderer import JSRenderer
from hypernodes.viz.ui_handler import UIHandler


@node(output_name="query")
def extract_query(eval_pair: dict) -> str:
    return eval_pair.get("query", "")


@node(output_name="documents")
def retrieve(query: str, num_results: int) -> list[dict[str, str]]:
    return [{"text": f"doc about {query}"}] * num_results


@node(output_name="answer")
def generate_answer(query: str, documents: list[dict[str, str]], model_name: str) -> str:
    return f"Answer using {model_name}"


@node(output_name=("score", "details"))
def evaluate(answer: str, expected: str) -> tuple[float, dict[str, any]]:
    return 1.0 if answer == expected else 0.0, {"answer": answer}


def test_edge_coordinates():
    # Build nested pipeline
    retrieval_step = Pipeline(nodes=[retrieve]).as_node(
        name="retrieval_step",
        input_mapping={"query": "query", "num_results": "num_results"},
        output_mapping={"documents": "documents"},
    )

    rag_pipeline = Pipeline(nodes=[extract_query, retrieval_step, generate_answer]).as_node(
        name="rag_pipeline",
        input_mapping={
            "eval_pair": "eval_pair",
            "num_results": "num_results",
            "model_name": "model_name",
        },
        output_mapping={"answer": "answer"},
    )

    pipeline = Pipeline(nodes=[rag_pipeline, evaluate])

    # Generate HTML
    handler = UIHandler(pipeline, depth=99)
    graph_data = handler.get_visualization_data(traverse_collapsed=True)
    renderer = JSRenderer()
    rf_data = renderer.render(graph_data, theme="dark", separate_outputs=False, show_types=True)
    html = generate_widget_html(rf_data)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        f.write(html.encode())
        html_path = f.name

    print(f"HTML saved to: {html_path}")

    # Use Playwright to test
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright not installed. Run: pip install playwright && playwright install chromium")
        return False

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"file://{html_path}")
        time.sleep(2)  # Wait for layout

        # Click to collapse rag_pipeline
        rag_btn = page.locator('[data-testid="rf__node-rag_pipeline"] button:has-text("Click to collapse")')
        rag_btn.click()
        time.sleep(1)  # Wait for collapse animation and re-layout

        # Get node and edge coordinates
        result = page.evaluate("""
            () => {
                // Get rag_pipeline node
                const ragNode = document.querySelector('[data-id="rag_pipeline"]');
                const ragY = parseFloat(ragNode.style.transform.match(/translate\\([^,]+px,\\s*([^)]+)px\\)/)[1]);
                const ragHeight = ragNode.getBoundingClientRect().height;
                const ragBottom = ragY + ragHeight;
                
                // Get edge from rag_pipeline to evaluate
                let edgeSourceY = null;
                document.querySelectorAll('.react-flow__edge').forEach(e => {
                    const testId = e.getAttribute('data-testid') || '';
                    if (testId.includes('rag_pipeline_evaluate')) {
                        const path = e.querySelector('path');
                        const d = path ? path.getAttribute('d') : '';
                        const match = d.match(/M([^,]+),([^\\s]+)/);
                        if (match) edgeSourceY = parseFloat(match[2]);
                    }
                });
                
                return { ragY, ragHeight, ragBottom, edgeSourceY };
            }
        """)

        browser.close()

        print("\n=== RESULTS ===")
        print(f"rag_pipeline node: y={result['ragY']}, height={result['ragHeight']}, bottom={result['ragBottom']}")
        print(f"Edge source Y: {result['edgeSourceY']}")

        discrepancy = abs(result['edgeSourceY'] - result['ragBottom'])
        print(f"\nDiscrepancy: {discrepancy:.1f}px")

        # Tolerance: handle size (8px) / 2 = 4px, plus some margin
        TOLERANCE = 5.0
        if discrepancy <= TOLERANCE:
            print(f"✅ PASS: Edge coordinates are correct (within {TOLERANCE}px tolerance)")
            return True
        else:
            print(f"❌ FAIL: Edge source Y ({result['edgeSourceY']}) doesn't match node bottom ({result['ragBottom']})")
            print(f"   Expected discrepancy <= {TOLERANCE}px, got {discrepancy:.1f}px")
            return False


if __name__ == "__main__":
    import sys
    success = test_edge_coordinates()
    sys.exit(0 if success else 1)
