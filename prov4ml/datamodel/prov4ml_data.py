
from typing import Any, Dict, List, Optional

from .artifact_data import ArtifactInfo
from .parameter_data import ParameterInfo
from .metric_data import MetricInfo

class Prov4MLData:
    """
    Holds the provenance data for metrics, parameters, and artifacts in a machine learning experiment.

    Attributes:
        metrics (Dict[str, MetricInfo]): A dictionary of metrics.
        parameters (Dict[str, ParameterInfo]): A dictionary of parameters.
        artifacts (Dict[str, ArtifactInfo]): A dictionary of artifacts.
        experiment_name (str): The name of the experiment.
    """
    def __init__(self) -> None:
        self.metrics: Dict[str, MetricInfo] = {}
        self.parameters: Dict[str, ParameterInfo] = {}
        self.artifacts: Dict[str, ArtifactInfo] = {}

        self.experiment_name = "test_experiment"

    def add_metric(self, metric: str, value: Any, step: int, context: Optional[Any] = None) -> None:
        """
        Adds a metric to the metrics dictionary.

        Parameters:
            metric (str): The name of the metric.
            value (Any): The value of the metric.
            step (int): The step number for the metric.
            context (Optional[Any]): The context of the metric. Defaults to None.
        """
        if metric not in self.metrics:
            self.metrics[metric] = MetricInfo(metric, context)
        self.metrics[metric].add_metric(value, step)

    def add_parameter(self, parameter: str, value: Any) -> None:
        """
        Adds a parameter to the parameters dictionary.

        Parameters:
            parameter (str): The name of the parameter.
            value (Any): The value of the parameter.
        """
        self.parameters[parameter] = ParameterInfo(parameter, value)

    def add_artifact(
        self, 
        artifact_name: str, 
        value: Any = None, 
        step: Optional[int] = None, 
        context: Optional[Any] = None, 
        timestamp: Optional[int] = None
    ) -> None:
        """
        Adds an artifact to the artifacts dictionary.

        Parameters:
            artifact_name (str): The name of the artifact.
            value (Any): The value of the artifact. Defaults to None.
            step (Optional[int]): The step number for the artifact. Defaults to None.
            context (Optional[Any]): The context of the artifact. Defaults to None.
            timestamp (Optional[int]): The timestamp of the artifact. Defaults to None.
        """
        self.artifacts[artifact_name] = ArtifactInfo(artifact_name, value, step, context=context, timestamp=timestamp)

    def get_artifacts(self) -> List[ArtifactInfo]:
        """
        Returns a list of all artifacts.

        Returns:
            List[ArtifactInfo]: A list of artifact information objects.
        """
        return list(self.artifacts.values())
    
    def get_model_versions(self) -> List[ArtifactInfo]:
        """
        Returns a list of all model version artifacts.

        Returns:
            List[ArtifactInfo]: A list of model version artifact information objects.
        """
        return [artifact for artifact in self.artifacts.values() if artifact.is_model_version]
    
    def get_final_model(self) -> Optional[ArtifactInfo]:
        """
        Returns the most recent model version artifact.

        Returns:
            Optional[ArtifactInfo]: The most recent model version artifact information object, or None if no model versions exist.
        """
        model_versions = self.get_model_versions()
        if model_versions:
            return model_versions[-1]
        return None
