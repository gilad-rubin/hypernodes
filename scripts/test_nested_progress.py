"""Test script for nested progress bars fix."""
import time
from hypernodes import node, Pipeline
from hypernodes.telemetry import ProgressCallback, TelemetryCallback
import logfire

# Configure logfire (suppress console output to avoid interference with tqdm)
logfire.configure(send_to_logfire=False, console=False)

# Define inner pipeline nodes
@node(output_name="cleaned")
def clean_data(data: str) -> str:
    """Clean the raw data."""
    time.sleep(0.2)
    return data.strip()


@node(output_name="normalized")
def normalize(cleaned: str) -> str:
    """Normalize the data."""
    time.sleep(0.3)
    return cleaned.lower()


# Create inner pipeline
inner_pipeline = Pipeline(nodes=[clean_data, normalize], id="preprocessing")


# Define outer pipeline nodes
@node(output_name="data")
def load_data(source: str) -> str:
    """Load data from source."""
    time.sleep(0.1)
    return f"  DATA FROM {source}  "


@node(output_name="final_result")
def aggregate(normalized: str) -> dict:
    """Aggregate the results."""
    time.sleep(0.2)
    return {"processed": normalized, "length": len(normalized)}


# Create outer pipeline with both callbacks
print("Testing nested progress bars with ProgressCallback and TelemetryCallback...\n")

outer_pipeline = Pipeline(
    nodes=[load_data, inner_pipeline, aggregate],
    callbacks=[ProgressCallback(), TelemetryCallback()],
    id="main_pipeline",
)

result = outer_pipeline.run(inputs={"source": "database"})
print(f"\nResult: {result}")
print("\n✓ Test completed! Check if progress bars displayed correctly.")
