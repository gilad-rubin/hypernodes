import importlib.util
import pathlib
from hypernodes import Pipeline
import hypernodes.pipeline as pipeline_mod

repo_root = pathlib.Path(__file__).resolve().parents[1]
script_path = repo_root / "scripts" / "test_modal_failure_repro.py"
spec = importlib.util.spec_from_file_location("test_modal_failure_repro", script_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # type: ignore

original_validate = pipeline_mod.Pipeline._validate

def main():
    pipeline_mod.Pipeline._validate = lambda self: None  # type: ignore
    try:
        pipeline = module.build_pipeline()
    finally:
        pipeline_mod.Pipeline._validate = original_validate
    print("nodes:", [getattr(n, "output_name", None) for n in pipeline.nodes])
    print("edges:")
    for edge in pipeline.graph.edges:
        print(edge)

if __name__ == "__main__":
    main()
