from azureml.core import Workspace


workspace = Workspace.from_config()
blobstore = workspace.get_default_datastore()

blobstore.upload("data")
