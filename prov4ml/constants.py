
from collections import namedtuple
from mlflow.tracking.fluent import get_current_time_millis

from .constants import ArtifactInfo

MLFLOW_SUBDIR = "mlflow"
ARTIFACTS_SUBDIR = "artifacts"

def artifact_is_pytorch_model(artifact: ArtifactInfo) -> bool:
    """
    Checks if the given artifact is a PyTorch model file.

    Parameters:
        artifact (Any): The artifact to check. Should have a 'path' attribute.

    Returns:
        bool: True if the artifact is a PyTorch model file, False otherwise.
    """
    return artifact.path.endswith(".pt") or artifact.path.endswith(".pth") or artifact.path.endswith(".torch")


class Prov4MLLOD():
    LVL_1 = "1"
    LVL_2 = "2"

    lv_attr = namedtuple('lv_attr', ['level', 'value'])
    lv_epoch_value = namedtuple('lv_step_attr', ['level', 'epoch', 'value'])

    def get_lv1_attr(value):
        return str(Prov4MLLOD.lv_attr(Prov4MLLOD.LVL_1, value))

    def get_lv2_attr(value):
        return str(Prov4MLLOD.lv_attr(Prov4MLLOD.LVL_2, value))
    
    def get_lv1_epoch_value(epoch, value):
        return str(Prov4MLLOD.lv_epoch_value(Prov4MLLOD.LVL_1, epoch, value))
    
    def get_lv2_epoch_value(epoch, value):
        return str(Prov4MLLOD.lv_epoch_value(Prov4MLLOD.LVL_2, epoch, value))


class ParameterInfo():
    def __init__(self, name, value):
        self.name = name
        self.value = value


class MetricInfo():
    def __init__(self, name, context):
        self.name = name
        self.context = context
        self.epochDataList = {} 

    def add_metric(self, value, epoch):
        if epoch not in self.epochDataList:
            self.epochDataList[epoch] = []

        self.epochDataList[epoch].append(value)


class ArtifactInfo():
    def __init__(self, name, value=None, step=None, context=None, timestamp=None):
        self.path = name
        self.value = value
        self.step = step
        self.context = context
        self.creation_timestamp = timestamp
        self.last_modified_timestamp = timestamp

        self.is_model_version = artifact_is_pytorch_model({"path": name})

    def update(self, value=None, step=None, context=None):
        self.value = value if value is not None else self.value
        self.step = step if step is not None else self.step
        self.context = context if context is not None else self.context
        self.last_modified_timestamp = get_current_time_millis()


class Prov4MLData():
    def __init__(self):
        self.metrics = {} # MetricList
        self.parameters = {} # ParameterInfo
        self.artifacts = {} # ArtifactList

        self.experiment_name = "test_experiment"

    def add_metric(self, metric, value, step, context=None):
        if metric not in self.metrics:
            self.metrics[metric] = MetricInfo(metric, context)
        self.metrics[metric].add_metric(value, step)

    def add_parameter(self, parameter, value):
        self.parameters[parameter] = ParameterInfo(parameter, value)

    def add_artifact(self, artifact_name, value=None, step=None, context=None, timestamp=None):
        self.artifacts[artifact_name] = ArtifactInfo(artifact_name, value, step, context=context, timestamp=timestamp)

    def get_artifacts(self):
        return self.artifacts.values()
    
    def get_model_versions(self):
        return [artifact for artifact in self.artifacts.values() if artifact.is_model_version]
    
    def get_final_model(self):
        model_versions = self.get_model_versions()
        if model_versions:
            return model_versions[-1]
        return None

PROV4ML_DATA = Prov4MLData()
