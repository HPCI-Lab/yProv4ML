
from typing import Any, Dict, List

class MetricInfo:
    """
    Holds information about a metric, including its context and epoch data.

    Attributes:
        name (str): The name of the metric.
        context (Any): The context of the metric.
        epochDataList (Dict[int, List[Any]]): A dictionary mapping epochs to lists of metric values.
    """
    def __init__(self, name: str, context: Any) -> None:
        self.name = name
        self.context = context
        self.epochDataList: Dict[int, List[Any]] = {}

    def add_metric(self, value: Any, epoch: int) -> None:
        """
        Adds a metric value for a specific epoch.

        Parameters:
            value (Any): The metric value to add.
            epoch (int): The epoch number.
        """
        if epoch not in self.epochDataList:
            self.epochDataList[epoch] = []

        self.epochDataList[epoch].append(value)
