import os
from hypernodes import Pipeline, node
from hypernodes.viz.visualization_widget import PipelineWidget

# Define some dummy nodes to match the user's image structure
@node(output_name="dataframe")
def user_data_source():
    return "data"

@node(output_name=("cleaned_df", "metrics"))
def clean_and_normalize(dataframe):
    return "cleaned", "metrics"

@node(output_name=("train_set", "test_set"))
def split_train_test(cleaned_df):
    return "train", "test"

@node(output_name="model_artifact")
def training_pipeline(train_set):
    return "model"

@node(output_name="report")
def evaluation(test_set, model_artifact):
    return "report"

# Create pipeline
pipeline = Pipeline(nodes=[
    user_data_source,
    clean_and_normalize,
    split_train_test,
    training_pipeline,
    evaluation
])

# Generate HTML
widget = PipelineWidget(pipeline)
html_content = widget.value

# Extract the base64 content to get the actual HTML
import base64
import re

match = re.search(r'src="data:text/html;base64,([^"]+)"', html_content)
if match:
    b64_data = match.group(1)
    decoded_html = base64.b64decode(b64_data).decode("utf-8")
    
    output_path = "current_viz.html"
    with open(output_path, "w") as f:
        f.write(decoded_html)
    print(f"Generated {output_path}")
else:
    print("Could not extract HTML from widget value")
