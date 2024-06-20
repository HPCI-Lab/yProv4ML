
import os
from typing import Any, Dict, Optional, Union
from typing_extensions import override
from typing import List
from torch.utils.data import DataLoader

from ..logging import *
from ..provenance.context import Context
from .itwinai_logger import Logger
from ..prov4ml import *
from ..datamodel.attribute_type import LoggingItemKind

class ProvMLItwinAILogger(Logger):
    def __init__(
        self,
        prov_user_namespace="www.example.org",
        experiment_name="experiment_name", 
        provenance_save_dir="prov",
        collect_all_processes: Optional[bool] = False,
        save_after_n_logs: Optional[int] = 100,
        create_graph: Optional[bool] = True,
        create_svg: Optional[bool] = True,
    ) -> None:
        """
        Initializes a ProvMLLogger instance.

        Parameters:
            name (Optional[str]): The name of the experiment. Defaults to "lightning_logs".
            version (Optional[Union[int, str]]): The version of the experiment. Defaults to None.
            prefix (str): The prefix for the experiment. Defaults to an empty string.
            flush_logs_every_n_steps (int): The number of steps after which logs should be flushed. Defaults to 100.
        """
        super().__init__()
        self._name = experiment_name
        self._version = None
        self.prov_user_namespace = prov_user_namespace
        self.provenance_save_dir = provenance_save_dir
        self.collect_all_processes = collect_all_processes
        self.save_after_n_logs = save_after_n_logs
        self.create_graph = create_graph
        self.create_svg = create_svg

    @property
    @override
    def root_dir(self) -> str:
        """
        Parent directory for all checkpoint subdirectories.

        If the experiment name parameter is an empty string, no experiment subdirectory is used and the checkpoint will
        be saved in "save_dir/version".

        Returns:
            str: The root directory path.
        """
        return os.path.join(self.save_dir, self.name)

    @property
    @override
    def log_dir(self) -> str:
        """
        The log directory for this run.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.

        Returns:
            str: The log directory path.
        """
        # create a pseudo standard path
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        return os.path.join(self.root_dir, version)

    @property
    @override
    def name(self) -> str:
        """
        The name of the experiment.

        Returns:
            str: The name of the experiment.
        """
        return self._name
    
    @property
    @override
    def version(self) -> Optional[Union[int, str]]:
        """
        The version of the experiment.

        Returns:
            Optional[Union[int, str]]: The version of the experiment.
        """
        return self._version
    
    @override
    def create_logger_context(self):
        """
        Initializes the logger context.
        """
        start_run(
            prov_user_namespace=self.prov_user_namespace,
            experiment_name=self.name,
            provenance_save_dir=self.provenance_save_dir,
            save_after_n_logs=self.save_after_n_logs,
            collect_all_processes=self.collect_all_processes,
        )

    @override
    def destroy_logger_context(self):
        """
        Destroys the logger context.
        """
        end_run(create_graph=self.create_graph, create_svg=self.create_svg)

    @override
    def save_hyperparameters(self, params: Dict[str, Any]) -> None:
        """
        Saves the hyperparameters to the MLflow tracking context.

        Parameters:
            params (Dict[str, Any]): A dictionary containing the hyperparameters.
        """
        for key, value in params.items():
            self.log(value, key, kind=LoggingItemKind.PARAMETER)

    @override
    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: Union[str, LoggingItemKind] = 'metric',
        step: Optional[int] = None,
        context: Optional[Context] = None,
        **kwargs
        ) -> None:
        """
        Logs the provided metrics to the MLflow tracking context.

        Parameters:
            metrics (Dict[str, Union[Tensor, float]]): A dictionary containing the metrics and their associated values.
            step (Optional[int]): The step number for the metrics. Defaults to None.
        """

        if kind == LoggingItemKind.METRIC:
            log_metric(identifier, item, context, step=step)
        elif kind == LoggingItemKind.FLOPS_PER_BATCH:
            model, batch = item
            log_flops_per_batch(identifier, model=model, batch=batch, context=context, step=step)
        elif kind == LoggingItemKind.FLOPS_PER_EPOCH:
            model, dataset = item
            log_flops_per_epoch(identifier, model=model, dataset=dataset, context=context, step=step)
        elif kind == LoggingItemKind.SYSTEM_METRIC:
            log_system_metrics(context=context, step=step)
        elif kind == LoggingItemKind.CARBON_METRIC:
            log_carbon_metrics(context=context, step=step)
        elif kind == LoggingItemKind.EXECUTION_TIME:
            log_current_execution_time(identifier, context, step=step)
        elif kind == LoggingItemKind.MODEL_VERSION:
            save_model_version(item, identifier, context, step=step)
        elif kind == LoggingItemKind.FINAL_MODEL_VERSION:
            log_model(item, identifier, log_model_info=True, log_as_artifact=True)
        elif kind == LoggingItemKind.PARAMETER:
            
            if isinstance(item, DataLoader):
                log_dataset(item, identifier)
            else:
                log_param(identifier, item)


