
import os
from typing import Any, Dict, List
from .attribute_type import LoggingItemKind

class MetricInfo:
    """
    Holds information about a metric, including its context and epoch data.

    Attributes:
        name (str): The name of the metric.
        context (Any): The context of the metric.
        epochDataList (Dict[int, List[Any]]): A dictionary mapping epochs to lists of metric values.
    """
    def __init__(self, name: str, context: Any, source=LoggingItemKind) -> None:
        self.name = name
        self.context = context
        self.source = source
        self.total_metric_values = 0
        self.epochDataList: Dict[int, List[Any]] = {}

    def add_metric(self, value: Any, epoch: int, timestamp : int) -> None:
        """
        Adds a metric value for a specific epoch.

        Parameters:
            value (Any): The metric value to add.
            epoch (int): The epoch number.
        """
        if epoch not in self.epochDataList:
            self.epochDataList[epoch] = []

        self.epochDataList[epoch].append((value, timestamp))
        self.total_metric_values += 1

    def save_to_file(self, path, process=None) -> None:
        if process:
            file = os.path.join(path, f"{self.name}_{self.context}_GR{process}.txt")
        else:
            file = os.path.join(path, f"{self.name}_{self.context}.txt")
        file_exists = os.path.exists(file)

        with open(file, "a") as f:
            if not file_exists:
                f.write(f"{self.name}, {self.context}, {self.source}\n")
            for epoch, values in self.epochDataList.items():
                for value, timestamp in values:
                    f.write(f"{epoch}, {value}, {timestamp}\n")


