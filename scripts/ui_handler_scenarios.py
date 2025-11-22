"""Progressively more complex UIHandler examples for manual verification.

Run with:
    uv run python scripts/ui_handler_scenarios.py
"""

from pprint import pprint
from typing import Dict, List

from hypernodes import Pipeline, node
from hypernodes.viz.ui_handler import UIHandler


# --- Base nodes used across scenarios ---
@node(output_name="text_step")
def add_bang(text: str) -> str:
    return text + "!"


@node(output_name="upper_text")
def to_upper(text_step: str) -> str:
    return text_step.upper()


@node(output_name="raw_text")
def passthrough(raw: str) -> str:
    return raw


@node(output_name="clean")
def clean_text(raw_text: str, suffix: str = "", language: str = "en") -> str:
    return f"{language}:{raw_text.strip()}{suffix}"


@node(output_name="length")
def text_length(clean: str) -> int:
    return len(clean)


@node(output_name="preview")
def text_preview(clean: str) -> str:
    return clean[:8]


@node(output_name="score")
def score(inner_length: int, tag: str, bias: int = 0) -> str:
    return f"{tag}:{inner_length + bias}"


# --- Helpers ---
def summarize_view(title: str, handler: UIHandler) -> Dict:
    view = handler.get_view_data()
    print(f"\n== {title}")
    for n in view["nodes"]:
        print(
            f"- {n.get('label')} [{n.get('node_type')}] "
            f"root_args={n.get('root_args')} "
            f"unfulfilled={n.get('unfulfilled_args')} "
            f"bound={n.get('bound_inputs')} "
            f"outputs={n.get('output_name') or n.get('output_names')}"
        )
    print("  edges:", [(e["source"], e["target"]) for e in view["edges"]])
    return view


def find_node_id_by_label(view: Dict, label: str) -> str:
    for n in view["nodes"]:
        if n.get("label") == label:
            return n["id"]
    raise ValueError(f"Node with label {label!r} not found")


def describe_update(prefix: str, update) -> None:
    print(
        f"{prefix}: "
        f"+{len(update.added_nodes)} nodes / -{len(update.removed_nodes)} nodes, "
        f"+{len(update.added_edges)} edges / -{len(update.removed_edges)} edges"
    )


# --- Scenarios ---
def scenario_single_node() -> None:
    pipeline = Pipeline(nodes=[add_bang], name="SingleNode")
    handler = UIHandler(pipeline, depth=1)
    summarize_view("Single node (depth=1)", handler)


def scenario_two_nodes() -> None:
    pipeline = Pipeline(nodes=[add_bang, to_upper], name="TwoNodes")
    handler = UIHandler(pipeline, depth=1)
    summarize_view("Two nodes connected (depth=1)", handler)


def build_inner_pipeline() -> Pipeline:
    inner = Pipeline(
        nodes=[clean_text, text_length, text_preview],
        name="InnerPipeline",
    ).bind(suffix="*", language="en")
    return inner


def build_outer_pipeline() -> Pipeline:
    inner = build_inner_pipeline()

    inner_node = inner.as_node(
        name="InnerPipeline",
        input_mapping={"text_step": "raw_text"},  # input renaming
        output_mapping={
            "length": "inner_length",  # output renaming (used)
            "preview": "inner_preview",  # output renaming (unused)
        },
    )

    outer = Pipeline(
        nodes=[
            add_bang,  # produces text_step
            inner_node,  # nested pipeline, two outputs (length + preview)
            score,  # consumes ONLY inner_length + tag (outer-only input)
        ],
        name="OuterPipeline",
    ).bind(bias=2)  # bound outer input

    return outer


def scenario_nested_depth_events() -> None:
    outer = build_outer_pipeline()

    # Depth=1 (collapsed nested pipeline)
    handler = UIHandler(outer, depth=1)
    view = summarize_view("Nested pipeline collapsed (depth=1)", handler)
    inner_id = find_node_id_by_label(view, "InnerPipeline")

    update = handler.handle_event({"type": "expand", "node_id": inner_id})
    describe_update("After expand event", update)
    # Show expanded view
    summarize_view("Nested pipeline after expand", handler)

    # Depth=2 (expanded by default), then collapse
    handler2 = UIHandler(outer, depth=2)
    view2 = summarize_view("Nested pipeline expanded (depth=2)", handler2)
    inner_id2 = find_node_id_by_label(view2, "InnerPipeline")

    update2 = handler2.handle_event({"type": "collapse", "node_id": inner_id2})
    describe_update("After collapse event", update2)
    summarize_view("Nested pipeline after collapse", handler2)


def scenario_bound_inputs_and_mixed_io() -> None:
    outer = build_outer_pipeline()
    # Bind bias already set; we also bind nothing else so tag/text remain unbound (outer),
    # inner binds suffix & language (inner-only).
    handler = UIHandler(outer, depth=2)
    view = summarize_view("Mixed bound/unbound inputs (depth=2)", handler)

    # Show event index sample for the inner pipeline
    inner_id = find_node_id_by_label(view, "InnerPipeline")
    event_index = handler.get_full_graph_with_state(include_events=True)["event_index"]
    print("\nPrecomputed event patches for InnerPipeline:")
    pprint({k: list(v.keys()) for k, v in event_index.items() if k == inner_id})


def main() -> None:
    scenario_single_node()
    scenario_two_nodes()
    scenario_nested_depth_events()
    scenario_bound_inputs_and_mixed_io()


if __name__ == "__main__":
    main()
