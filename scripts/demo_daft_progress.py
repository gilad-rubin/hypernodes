import argparse
import os
import time

from hypernodes import Pipeline, node
from hypernodes.integrations.daft.engine import DaftEngine
from hypernodes.sequential_engine import SeqEngine
from hypernodes.telemetry import ProgressCallback


# Define a slow node to make progress visible
@node(output_name="y")
def slow_add_one(x: int) -> int:
    time.sleep(0.1)  # Sleep 100ms
    return x + 1


@node(output_name="z")
def slow_multiply_two(y: int) -> int:
    time.sleep(0.1)
    return y * 2


@node(output_name="w")
def slow_add_two(y: int) -> int:
    time.sleep(0.1)
    return y + 2


def _build_pipeline(callback: ProgressCallback | None = None) -> Pipeline:
    pipeline = Pipeline(nodes=[slow_add_one, slow_multiply_two, slow_add_two])
    if callback:
        pipeline.callbacks = [callback]
    return pipeline


def _build_inputs(items: int) -> dict:
    return {"x": list(range(items))}


def run_sequential(items: int) -> None:
    print("\n" + "=" * 50)
    print("DEMO: SeqEngine (Baseline)")
    print("=" * 50)

    seq_engine = SeqEngine()
    callback = ProgressCallback()
    pipeline = _build_pipeline(callback)
    seq_engine.map(
        pipeline,
        _build_inputs(items),
        map_over="x",
    )


def run_daft(items: int) -> None:
    print("\n" + "=" * 50)
    print("DEMO: DaftEngine (Should match baseline)")
    print("=" * 50)

    # Ensure Daft's internal bar is off so we only see HyperNodes bar
    os.environ.setdefault("DAFT_PROGRESS_BAR", "0")

    daft_engine = DaftEngine()
    callback = ProgressCallback()
    pipeline = _build_pipeline(callback)
    daft_engine.map(
        pipeline,
        _build_inputs(items),
        map_over="x",
        callbacks=[callback],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Sequential vs Daft progress bars."
    )
    parser.add_argument(
        "--engine",
        choices=["both", "sequential", "daft"],
        default="both",
        help="Which engine demo to run.",
    )
    parser.add_argument(
        "--items",
        type=int,
        default=10,
        help="Number of items to process in the demo map operation.",
    )
    args = parser.parse_args()

    if args.engine in ("both", "sequential"):
        run_sequential(args.items)
        if args.engine == "both":
            print("\nSequential demo complete.\n")

    if args.engine in ("both", "daft"):
        run_daft(args.items)


if __name__ == "__main__":
    main()
