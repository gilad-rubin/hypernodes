#!/usr/bin/env python3
"""
Drop this file next to your actual metrics_pipeline and run it to debug labels.
Usage: python debug_your_viz.py
"""


def debug_pipeline_labels(pipeline, name="your_pipeline"):
    """Inspect all nodes in a pipeline and their label generation."""

    print("=" * 80)
    print(f"DEBUGGING: {name}")
    print("=" * 80)
    print()

    # Check each node
    print("Pipeline nodes:")
    for i, node in enumerate(pipeline.nodes):
        print(f"\n[{i}] Node type: {type(node).__name__}")
        print(f"    ID: {id(node)}")
        print(f"    name attribute: {getattr(node, 'name', 'NO NAME ATTRIBUTE')}")

        if hasattr(node, "func"):
            func = node.func
            print(f"    func.__name__: {getattr(func, '__name__', 'NO __name__')}")

        if hasattr(node, "pipeline"):
            print("    [NESTED PIPELINE]")
            print(f"    pipeline.name: {getattr(node.pipeline, 'name', 'NO NAME')}")
            print(f"    inner nodes: {len(node.pipeline.nodes)}")
            if node.pipeline.nodes:
                first = node.pipeline.nodes[0]
                print(f"    first inner node: {type(first).__name__}")
                if hasattr(first, "func"):
                    print(
                        f"    first func.__name__: {getattr(first.func, '__name__', 'NO __name__')}"
                    )

    # Generate visualization at depth=2 and check for raw IDs
    print("\n" + "=" * 80)
    print("Generating visualization at depth=2...")
    print("=" * 80)

    try:
        result = pipeline.visualize(depth=2)
        html_str = str(result) if hasattr(result, "_repr_html_") else result

        # Save it
        with open(f"outputs/debug_{name}_depth2.html", "w") as f:
            f.write(html_str)
        print(f"✅ Saved to outputs/debug_{name}_depth2.html")

        # Check for raw IDs
        import re

        id_pattern = r">\s*\d{10,}\s*<"
        matches = re.findall(id_pattern, html_str)

        if matches:
            print(f"\n⚠️  FOUND {len(matches)} RAW IDS IN VISUALIZATION:")
            for match in matches:
                clean_id = match.strip(">< \n")
                print(f"   - {clean_id}")

                # Try to find which node this is
                try:
                    node_id_int = int(clean_id)
                    for node in pipeline.nodes:
                        if id(node) == node_id_int:
                            print(f"     → This is: {type(node).__name__}")
                            print(f"        name: {getattr(node, 'name', 'NO NAME')}")
                            break
                except:
                    pass
        else:
            print("\n✅ No raw IDs found!")

    except Exception as e:
        print(f"\n❌ Error generating visualization: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # INSTRUCTIONS:
    # 1. Import your actual metrics_pipeline here
    # 2. Call debug_pipeline_labels(metrics_pipeline, "metrics")

    print("=" * 80)
    print("HOW TO USE THIS SCRIPT:")
    print("=" * 80)
    print()
    print("1. Import your actual pipeline:")
    print("   from your_module import metrics_pipeline")
    print()
    print("2. Call the debug function:")
    print("   debug_pipeline_labels(metrics_pipeline, 'metrics')")
    print()
    print("3. Check the output for nodes without proper names")
    print()
    print("=" * 80)
