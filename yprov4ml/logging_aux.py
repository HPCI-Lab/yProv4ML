import os
import torch
import json
import sys
import warnings
import prov.model as prov

from torch.utils.data import DataLoader, Subset, Dataset, RandomSampler
from typing import Any, Optional, Union

from yprov4ml.utils import energy_utils, flops_utils, system_utils, time_utils, funcs
from yprov4ml.constants import PROV4ML_DATA, VERBOSE

def log_metric(key: str, value: float, context: Optional[str] = None, step: int = 0, source: Optional[str] = None, timestamp : int = 0) -> None:
    PROV4ML_DATA.add_metric(key, value, step, context=context, source=source, timestamp=timestamp)

def _log_execution_start_time() -> None:
    PROV4ML_DATA.add_parameter("startedAtTime", time_utils.get_time(), source='std.time', prefix="prov:", is_input=False)

def _log_execution_end_time() -> None:
    PROV4ML_DATA.add_parameter("endedAtTime", time_utils.get_time(), source='std.time', prefix="prov:", is_input=False)

def log_current_execution_time(label: str, context : Optional[str] = None, step: Optional[int] = None) -> None:
    return log_metric(label, time_utils.get_time(), context=context, step=step, source='std.time', is_input=False)

def log_param(key: str, value: Any, context : Optional[str] = None, source : Optional[str] = None, is_input=False) -> None:
    PROV4ML_DATA.add_parameter(key,value, context, source, is_input=is_input)

def _get_model_memory_footprint(model_name: str, model: Union[torch.nn.Module, Any]) -> dict:

    total_params = sum(p.numel() for p in model.parameters())
    try: 
        if hasattr(model, "trainer"): 
            precision_to_bits = {"64": 64, "32": 32, "16": 16, "bf16": 16}
            if hasattr(model.trainer, "precision"):
                precision = precision_to_bits.get(model.trainer.precision, 32)
            else: 
                precision = 32
        else: 
            precision = 32
    except RuntimeError: 
        if VERBOSE: 
            warnings.warn("Could not determine precision, defaulting to 32 bits. Please make sure to provide a model with a trainer attached, this is often due to calling this before the trainer.fit() method")
        precision = 32
    
    precision_megabytes = precision / 8 / 1e6

    memory_per_model = total_params * precision_megabytes
    memory_per_grad = total_params * 4 * 1e-6
    memory_per_optim = total_params * 4 * 1e-6
    
    ret = {f"{PROV4ML_DATA.yProv_PREFIX}:model_name": model_name}
    ret[f"{PROV4ML_DATA.yProv_PREFIX}:total_params"] = total_params
    ret[f"{PROV4ML_DATA.yProv_PREFIX}:memory_of_model"] = memory_per_model
    ret[f"{PROV4ML_DATA.yProv_PREFIX}:total_memory_load_of_model"] = memory_per_model + memory_per_grad + memory_per_optim

    return ret

def _get_nested_model_desc(m: torch.nn.Module):
    children = dict(m.named_children())
    output = {}
    if children == {}:
        node = {}
        funcs._safe_get_model_attr(node, m, "type", attr_label="layer_type")
        funcs._safe_get_model_attr(node, m, "in_features")
        funcs._safe_get_model_attr(node, m, "out_features")
        funcs._safe_get_model_attr(node, m, "in_channels")
        funcs._safe_get_model_attr(node, m, "out_channels")
        funcs._safe_get_model_attr(node, m, "kernel_size")
        funcs._safe_get_model_attr(node, m, "stride")
        funcs._safe_get_model_attr(node, m, "padding")
        funcs._safe_get_model_attr(node, m, "weight.dtype", attr_label="dtype")
        # safe_get_attr(node, m, "bias", attr_label="layer_bias")
        return node
    else:
        for name, child in children.items():
            output[name] = _get_nested_model_desc(child)

    return output

def _get_model_layers_description(model_name : str, model: Union[torch.nn.Module, Any]) -> prov.ProvEntity: 
    mo = _get_nested_model_desc(model)
    
    path = os.path.join(PROV4ML_DATA.ARTIFACTS_DIR, model_name)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    with open(f"{path}/{model_name}_layers_description.json", "w") as fp:
        json.dump(mo , fp) 

    return {f"{PROV4ML_DATA.yProv_PREFIX}:layers_description_path": f"{path}/{model_name}_layers_description.json"}

def log_model(
        model_name: str, 
        model: Union[torch.nn.Module, Any], 
        context : Optional[str] = None, 
        source : Optional[str] = None, 
        log_model_info: bool = True, 
        log_model_layers : bool = False,
        is_input: bool = False,
    ) -> None:      
    e = save_model_version(model_name, model, context=context, source=source, incremental=False, is_input=is_input)

    if log_model_info:
        d = _get_model_memory_footprint(model_name, model)
        e.add_attributes(d)

    if log_model_layers: 
        d = _get_model_layers_description(model_name, model)
        e.add_attributes(d)
    
def log_flops_per_epoch(label: str, model: Any, dataset: Any, context: Optional[str] = None, step: Optional[int] = None) -> None:
    return log_metric(label, flops_utils.get_flops_per_epoch(model, dataset), context, step=step, source='fvcore.nn.FlopCountAnalysis')

def log_flops_per_batch(label: str, model: Any, batch: Any, context: Optional[str] = None, step: Optional[int] = None) -> None:
    return log_metric(label, flops_utils.get_flops_per_batch(model, batch), context, step=step, source='fvcore.nn.FlopCountAnalysis')

def log_system_metrics(context: Optional[str] = None, step: int = 0) -> None:
    data, src = system_utils.get_bulk_stats()
    timestamp = funcs.get_current_time_millis()
    log_metric("cpu_usage", data["cpu_usage"], context, step=step, source=src, timestamp=timestamp)
    log_metric("memory_usage", data["memory_usage"], context, step=step, source=src,timestamp=timestamp)
    log_metric("disk_usage", data["disk_usage"], context, step=step, source=src, timestamp=timestamp)
    log_metric("gpu_memory_power", data["gpu_memory_power"], context, step=step, source=src, timestamp=timestamp)
    log_metric("gpu_memory_usage", data["gpu_memory_usage"], context, step=step, source=src, timestamp=timestamp)
    log_metric("gpu_usage", data["gpu_usage"], context, step=step, source=src, timestamp=timestamp)
    log_metric("gpu_power_usage", data["gpu_power_usage"], context, step=step, source=src, timestamp=timestamp)
    log_metric("gpu_temperature", data["gpu_temperature"], context, step=step, source=src, timestamp=timestamp)

def log_carbon_metrics(context: Optional[str] = None, step: int = 0): 
    if PROV4ML_DATA.codecarbon_is_disabled: 
        raise Exception(">log_carbon_metrics(): The log_carbon_metrics function cannot be called if disable_codecarbon=True")

    emissions = energy_utils.stop_carbon_tracked_block()
    timestamp = funcs.get_current_time_millis()
   
    log_metric("emissions", emissions.energy_consumed, context, step=step, source='codecarbon', timestamp=timestamp)
    log_metric("emissions_rate", emissions.emissions_rate, context, step=step, source='codecarbon', timestamp=timestamp)
    log_metric("cpu_power", emissions.cpu_power, context, step=step, source='codecarbon', timestamp=timestamp)
    log_metric("gpu_power", emissions.gpu_power, context, step=step, source='codecarbon', timestamp=timestamp)
    log_metric("ram_power", emissions.ram_power, context, step=step, source='codecarbon', timestamp=timestamp)
    log_metric("cpu_energy", emissions.cpu_energy, context, step=step, source='codecarbon', timestamp=timestamp)
    log_metric("gpu_energy", emissions.gpu_energy, context, step=step, source='codecarbon', timestamp=timestamp)
    log_metric("ram_energy", emissions.ram_energy, context, step=step, source='codecarbon', timestamp=timestamp)
    log_metric("energy_consumed", emissions.energy_consumed, context, step=step, source='codecarbon', timestamp=timestamp)

def log_artifact(
        artifact_name : str, 
        artifact_path : str, 
        context: Optional[str] = None,
        source : Optional[str] = None, 
        step: int = 0, 
        log_copy_in_prov_directory : bool = True, 
        log_copy_subdirectory : Optional[str] = None, 
        is_model : bool = False, 
        is_input : bool = False, 
    ) -> prov.ProvEntity:
    return PROV4ML_DATA.add_artifact(
        artifact_name=artifact_name, 
        artifact_path=artifact_path, 
        step=step, 
        context=context, 
        source=source, 
        log_copy_in_prov_directory=log_copy_in_prov_directory, 
        log_copy_subdirectory=log_copy_subdirectory,
        is_model=is_model, 
        is_input=is_input, 
    )

def save_model_version(
        model_name: str, 
        model: Union[torch.nn.Module, Any], 
        context: Optional[str] = None, 
        source: Optional[str] = None, 
        step: Optional[int] = None, 
        incremental : bool = True, 
        is_input : bool =False, 
    ) -> prov.ProvEntity:

    if incremental: 
        path = os.path.join(PROV4ML_DATA.ARTIFACTS_DIR, model_name)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        num_files = len([file for file in os.listdir(path) if str(file).startswith(model_name)])
        file_name = f"{model_name}_{num_files}.pt"
        
        torch.save(model.state_dict(), file_name)
        return log_artifact(
            f"{model_name}_{num_files}", file_name, 
            context=context, source=source, step=step, 
            log_copy_in_prov_directory=True, 
            log_copy_subdirectory=model_name, 
            is_model=True, is_input=is_input, 
        )
    else: 
        file_name = f"{model_name}.pt"
        torch.save(model.state_dict(), file_name)
        return log_artifact(
            model_name, file_name, 
            context=context, source=source, step=step, 
            log_copy_in_prov_directory=True, 
            log_copy_subdirectory=model_name, 
            is_model=True, is_input=is_input
        )

def log_dataset(
        dataset_label : str, 
        dataset : Union[DataLoader, Subset, Dataset], 
        context : Optional[str] = None, 
        source : Optional[str] = None, 
        log_dataset_info : bool = True, 
        ): 

    e = log_artifact(dataset_label, "", context=context, log_copy_in_prov_directory=False, is_model=False, is_input=True, source=source)
    
    if not log_dataset_info: return

    e.add_attributes({f"{PROV4ML_DATA.yProv_PREFIX}:{dataset_label}_stat_total_samples": len(dataset)})
    if isinstance(dataset, DataLoader):
        dl = dataset
        dataset = dl.dataset
        attrs = {
            f"{PROV4ML_DATA.yProv_PREFIX}:{dataset_label}_stat_batch_size": dl.batch_size, 
            f"{PROV4ML_DATA.yProv_PREFIX}:{dataset_label}_stat_num_workers": dl.num_workers, 
            f"{PROV4ML_DATA.yProv_PREFIX}:{dataset_label}_stat_shuffle": isinstance(dl.sampler, RandomSampler), 
            f"{PROV4ML_DATA.yProv_PREFIX}:{dataset_label}_stat_total_steps": len(dl), 
        }
        e.add_attributes(attrs)

    elif isinstance(dataset, Subset):
        dl = dataset
        dataset = dl.dataset
        e.add_attributes({f"{PROV4ML_DATA.yProv_PREFIX}:{dataset_label}_stat_total_steps": len(dl)})

def log_execution_command(cmd: str, path : str) -> None:
    path = os.path.join("/workspace", f"{PROV4ML_DATA.CLEAN_EXPERIMENT_NAME}_{PROV4ML_DATA.RUN_ID}", "artifacts", path)
    log_param("execution_command", cmd + " " + path)

def log_source_code() -> None:
    PROV4ML_DATA.request_source_code()

def log_context(context : str, is_subcontext_of : Optional[str] = None, source_of_data : Optional[str] = None): 
    PROV4ML_DATA._add_ctx(is_subcontext_of, context, source=source_of_data)