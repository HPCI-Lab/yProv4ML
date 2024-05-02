
import torch
import warnings
import mlflow
from mlflow.entities import Metric, RunTag
from mlflow.tracking.fluent import log_metric, log_param
from mlflow.tracking.fluent import get_current_time_millis
from mlflow.utils.async_logging.run_operations import RunOperations
from typing import Any, Dict, Optional, Tuple, Union

from .utils import energy_utils, flops_utils, system_utils, time_utils
from .provenance.context import Context

def log_metrics(
        metrics:Dict[str,Tuple[float,Context]],
        step:Optional[int]=None,
        synchronous:bool=True
        ) -> Optional[RunOperations]:
    """
    Logs the given metrics and their associated contexts to the active MLflow run.

    Parameters:
        metrics (Dict[str, Tuple[float, Context]]): A dictionary containing the metrics and their associated contexts.
        step (Optional[int]): The step number for the metrics. Defaults to None.
        synchronous (bool): Whether to log the metrics synchronously or asynchronously. Defaults to True.

    Returns:
        Optional[RunOperations]: The run operations object if logging is successful, None otherwise.
    """
    #create two separate lists, one for metrics and one for tags, and log them together using native log.batch
    client= mlflow.MlflowClient()

    timestamp=get_current_time_millis()
    metrics_arr=[Metric(key,value,timestamp,step or 0) for key,(value,context) in metrics.items()]
    tag_arr=[RunTag(f'metric.context.{key}',context.name) for key,(value,context) in metrics.items()]

    return client.log_batch(mlflow.active_run().info.run_id,metrics=metrics_arr,tags=tag_arr,synchronous=synchronous)

def log_metric(key: str, value: float, context:Context, step: Optional[int] = None, synchronous: bool = True, timestamp: Optional[int] = None) -> Optional[RunOperations]:
    """
    Logs a metric with the specified key, value, and context.

    Args:
        key (str): The key of the metric.
        value (float): The value of the metric.
        context (Context): The context of the metric.
        step (Optional[int], optional): The step of the metric. Defaults to None.
        synchronous (bool, optional): Whether to log the metric synchronously. Defaults to True.
        timestamp (Optional[int], optional): The timestamp of the metric. Defaults to None.

    Returns:
        Optional[RunOperations]: The run operations object.

    """
    client = mlflow.MlflowClient()
    client.set_tag(mlflow.active_run().info.run_id,f'metric.context.{key}',context.name)
    return client.log_metric(mlflow.active_run().info.run_id,key,value,step=step or 0,synchronous=synchronous,timestamp=timestamp or get_current_time_millis())

def log_execution_start_time() -> None:
    """Logs the start time of the current execution to the MLflow tracking context."""
    return log_param("execution_start_time", time_utils.get_time())

def log_execution_end_time() -> None:
    """Logs the end time of the current execution to the MLflow tracking context."""
    return log_param("execution_end_time", time_utils.get_time())

def log_current_execution_time(label: str, context: Context, step: Optional[int] = None) -> None:
    """Logs the current execution time under the given label in the MLflow tracking context.
    
    Args:
        label (str): The label to associate with the logged execution time.
        context (mlflow.tracking.Context): The MLflow tracking context.
        step (Optional[int], optional): The step number for the logged execution time. Defaults to None.
    """
    return log_metric(label, time_utils.get_time(), context, step=step)

def log_param(key: str, value: Any) -> None:
    """Logs a single parameter key-value pair to the MLflow tracking context.
    
    Args:
        key (str): The key of the parameter.
        value (Any): The value of the parameter.
    """
    return mlflow.log_param(key, value)

def log_params(params: Dict[str, Any]) -> None:
    """Logs multiple parameter key-value pairs to the MLflow tracking context.
    
    Args:
        params (Dict[str, Any]): A dictionary containing parameter key-value pairs.
    """
    return mlflow.log_params(params)

def log_model_memory_footprint(model: Union[torch.nn.Module, Any], model_name: str = "default") -> None:
    """Logs memory footprint of the provided model to the MLflow tracking context.
    
    Args:
        model (Union[torch.nn.Module, Any]): The model whose memory footprint is to be logged.
        model_name (str, optional): Name of the model. Defaults to "default".
    """
    log_param("model_name", model_name)

    total_params = sum(p.numel() for p in model.parameters())
    try: 
        if hasattr(model, "trainer"): 
            precision_to_bits = {"64": 64, "32": 32, "16": 16, "bf16": 16}
            precision = precision_to_bits.get(model.trainer.precision, 32)  
        else: 
            precision = 32
    except RuntimeError: 
        warnings.warn("Could not determine precision, defaulting to 32 bits. Please make sure to provide a model with a trainer attached, this is often due to calling this before the trainer.fit() method")
        precision = 32
    
    precision_megabytes = precision / 8 / 1e6

    memory_per_model = total_params * precision_megabytes
    memory_per_grad = total_params * 4 * 1e-6
    memory_per_optim = total_params * 4 * 1e-6
    
    log_param("total_params", total_params)
    log_param("memory_of_model", memory_per_model)
    log_param("total_memory_load_of_model", memory_per_model + memory_per_grad + memory_per_optim)

def log_model(model: Union[torch.nn.Module, Any], model_name: str = "default", log_model_info: bool = True) -> None:
    """Logs the provided model to the MLflow tracking context.
    
    Args:
        model (Union[torch.nn.Module, Any]): The model to be logged.
        model_name (str, optional): Name of the model. Defaults to "default".
        log_model_info (bool, optional): Whether to log model memory footprint. Defaults to True.
    """
    if log_model_info:
        log_model_memory_footprint(model, model_name)

    return mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path=mlflow.active_run().info.run_name.split("/")[-1],
        registered_model_name=model_name
    )

def log_optimizer(optimizer: torch.optim.Optimizer) -> None:
    """Logs the provided optimizer to the MLflow tracking context.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer to be logged.
    """
    opt_name = optimizer.__class__.__name__
    log_param("optimizer_name", opt_name)
    log_param("optimizer_state_dict", optimizer.state_dict())
    # lr
    for param_group in optimizer.param_groups:
        log_param("lr", param_group["lr"])
        break

def log_flops_per_epoch(label: str, model: Any, dataset: Any, context: Context, step: Optional[int] = None) -> None:
    """Logs the number of FLOPs (floating point operations) per epoch for the given model and dataset.
    
    Args:
        label (str): The label to associate with the logged FLOPs per epoch.
        model (Any): The model for which FLOPs per epoch are to be logged.
        dataset (Any): The dataset used for training the model.
        context (mlflow.tracking.Context): The MLflow tracking context.
        step (Optional[int], optional): The step number for the logged FLOPs per epoch. Defaults to None.
    """
    return log_metric(label, flops_utils.get_flops_per_epoch(model, dataset), context, step=step)

def log_flops_per_batch(label: str, model: Any, batch: Any, context: Context, step: Optional[int] = None) -> None:
    """Logs the number of FLOPs (floating point operations) per batch for the given model and batch of data.
    
    Args:
        label (str): The label to associate with the logged FLOPs per batch.
        model (Any): The model for which FLOPs per batch are to be logged.
        batch (Any): A batch of data used for inference with the model.
        context (mlflow.tracking.Context): The MLflow tracking context.
        step (Optional[int], optional): The step number for the logged FLOPs per batch. Defaults to None.
    """
    return log_metric(label, flops_utils.get_flops_per_batch(model, batch), context, step=step)

def log_system_metrics(
    context: Context,
    step: Optional[int] = None,
    synchronous: bool = True,
    timestamp: Optional[int] = None
    ) -> None:
    """Logs system metrics to the MLflow tracking context.
    
    Args:
        context (mlflow.tracking.Context): The MLflow tracking context.
        step (Optional[int], optional): The step number for the logged metrics. Defaults to None.
        synchronous (bool, optional): If True, performs synchronous logging. Defaults to True.
        timestamp (Optional[int], optional): The timestamp for the logged metrics. Defaults to None.
    """
    client = mlflow.MlflowClient()
    client.set_tag(mlflow.active_run().info.run_id,'metric.context.cpu_usage',context.name)
    client.log_metric(mlflow.active_run().info.run_id, "cpu_usage", system_utils.get_cpu_usage(), step=step, synchronous=synchronous,timestamp=timestamp or get_current_time_millis())
    client.set_tag(mlflow.active_run().info.run_id,'metric.context.memory_usage',context.name)
    client.log_metric(mlflow.active_run().info.run_id, "memory_usage", system_utils.get_memory_usage(), step=step, synchronous=synchronous,timestamp=timestamp or get_current_time_millis())
    client.set_tag(mlflow.active_run().info.run_id,'metric.context.disk_usage',context.name)
    client.log_metric(mlflow.active_run().info.run_id, "disk_usage", system_utils.get_disk_usage(), step=step, synchronous=synchronous,timestamp=timestamp or get_current_time_millis())
    client.set_tag(mlflow.active_run().info.run_id,'metric.context.gpu_memory_usage',context.name)
    client.log_metric(mlflow.active_run().info.run_id, "gpu_memory_usage", system_utils.get_gpu_memory_usage(), step=step, synchronous=synchronous,timestamp=timestamp or get_current_time_millis())
    client.set_tag(mlflow.active_run().info.run_id,'metric.context.gpu_usage',context.name)
    client.log_metric(mlflow.active_run().info.run_id, "gpu_usage", system_utils.get_gpu_usage(), step=step, synchronous=synchronous,timestamp=timestamp or get_current_time_millis())

def log_carbon_metrics(
    context: Context,
    step: Optional[int] = None,
    synchronous: bool = True,
    timestamp: Optional[int] = None
    ) -> Tuple[float, float]:
    """Logs carbon emissions metrics to the MLflow tracking context.
    
    Args:
        context (mlflow.tracking.Context): The MLflow tracking context.
        step (Optional[int], optional): The step number for the logged metrics. Defaults to None.
        synchronous (bool, optional): If True, performs synchronous logging. Defaults to True.
        timestamp (Optional[int], optional): The timestamp for the logged metrics. Defaults to None.
    
    Returns:
        Tuple[float, float]: A tuple containing energy consumed and emissions rate.
    """    
    emissions = energy_utils.stop_carbon_tracked_block()
   
    client = mlflow.MlflowClient()
    client.set_tag(mlflow.active_run().info.run_id,'metric.context.emissions',context.name)
    client.log_metric(mlflow.active_run().info.run_id, "emissions", emissions.energy_consumed, step=step, synchronous=synchronous,timestamp=timestamp or get_current_time_millis())
    client.set_tag(mlflow.active_run().info.run_id,'metric.context.emissions_rate',context.name)
    client.log_metric(mlflow.active_run().info.run_id, "emissions_rate", emissions.emissions_rate, step=step, synchronous=synchronous,timestamp=timestamp or get_current_time_millis())
    client.set_tag(mlflow.active_run().info.run_id,'metric.context.cpu_power',context.name)
    client.log_metric(mlflow.active_run().info.run_id, "cpu_power", emissions.cpu_power, step=step, synchronous=synchronous,timestamp=timestamp or get_current_time_millis())
    client.set_tag(mlflow.active_run().info.run_id,'metric.context.gpu_power',context.name)
    client.log_metric(mlflow.active_run().info.run_id, "gpu_power", emissions.gpu_power, step=step, synchronous=synchronous,timestamp=timestamp or get_current_time_millis())
    client.set_tag(mlflow.active_run().info.run_id,'metric.context.ram_power',context.name)
    client.log_metric(mlflow.active_run().info.run_id, "ram_power", emissions.ram_power, step=step, synchronous=synchronous,timestamp=timestamp or get_current_time_millis())
    client.set_tag(mlflow.active_run().info.run_id,'metric.context.cpu_energy',context.name)
    client.log_metric(mlflow.active_run().info.run_id, "cpu_energy", emissions.cpu_energy, step=step, synchronous=synchronous,timestamp=timestamp or get_current_time_millis())
    client.set_tag(mlflow.active_run().info.run_id,'metric.context.gpu_energy',context.name)
    client.log_metric(mlflow.active_run().info.run_id, "gpu_energy", emissions.gpu_energy, step=step, synchronous=synchronous,timestamp=timestamp or get_current_time_millis())
    client.set_tag(mlflow.active_run().info.run_id,'metric.context.ram_energy',context.name)
    client.log_metric(mlflow.active_run().info.run_id, "ram_energy", emissions.ram_energy, step=step, synchronous=synchronous,timestamp=timestamp or get_current_time_millis())
    client.set_tag(mlflow.active_run().info.run_id,'metric.context.energy_consumed',context.name)
    client.log_metric(mlflow.active_run().info.run_id, "energy_consumed", emissions.energy_consumed, step=step, synchronous=synchronous,timestamp=timestamp or get_current_time_millis())
