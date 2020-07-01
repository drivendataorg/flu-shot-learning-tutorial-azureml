from azureml.core import Environment, Workspace
from azureml.core.runconfig import RunConfiguration
from azureml.data.data_reference import DataReference
from azureml.data.datapath import DataPath, DataPathComputeBinding
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep


workspace = Workspace.from_config()
blobstore = workspace.get_default_datastore()

environment = Environment.get(workspace, name="AzureML-Scikit-learn-0.20.3")
environment.docker.enabled = True

run_config = RunConfiguration()
run_config.environment = environment

compute_target = workspace.compute_targets["cpu"]
run_config.target = compute_target

train_features_datapath = DataPath(
    datastore=blobstore, path_on_datastore="training_set_features.csv"
)
train_features_path_parameter = PipelineParameter(
    name="train_features", default_value=train_features_datapath
)
train_features_path = (train_features_path_parameter, DataPathComputeBinding(mode="mount"))

train_labels_datapath = DataPath(
    datastore=blobstore, path_on_datastore="training_set_labels.csv"
)
train_labels_path_parameter = PipelineParameter(
    name="train_labels", default_value=train_labels_datapath
)
train_labels_path = (train_labels_path_parameter, DataPathComputeBinding(mode="mount"))

test_features_datapath = DataPath(
    datastore=blobstore, path_on_datastore="test_set_features.csv"
)
test_features_path_parameter = PipelineParameter(
    name="test_features", default_value=test_features_datapath
)
test_features_path = (test_features_path_parameter, DataPathComputeBinding(mode="mount"))

submission_format_path = DataReference(
    data_reference_name="submission_format",
    datastore=blobstore,
    path_on_datastore="submission_format.csv",
)

submission_path = PipelineData(name="submission", datastore=blobstore)

model_path = PipelineData(name="model", datastore=blobstore)

step = PythonScriptStep(
    script_name="script.py",
    source_directory="script",
    name="flu_shot_learning",
    arguments=[
        train_features_path,
        train_labels_path,
        test_features_path,
        submission_format_path,
        submission_path,
        model_path,
    ],
    inputs=[train_features_path, train_labels_path, test_features_path, submission_format_path],
    outputs=[submission_path, model_path],
    runconfig=run_config,
)

pipeline = Pipeline(workspace=workspace, steps=[step])
pipeline.validate()

published_pipeline = pipeline.publish(
    "flu_shot_learning", description="Train a model for DrivenData flu shot learning competition"
)

published_pipeline.submit(workspace, experiment_name="flu-shot-learning")
