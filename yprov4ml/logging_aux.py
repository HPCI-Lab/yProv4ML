import os, hashlib, io, sys
import torch
import json
import warnings
import prov.model as prov
import shlex
from pathlib import Path

from torch.utils.data import DataLoader, Subset, Dataset, RandomSampler
from typing import Any, Optional, Union

from yprov4ml.utils import flops_utils, time_utils, funcs
from yprov4ml.constants import PROV4ML_DATA

def log_metric(key: str, value: float, is_input_to: Optional[str] = None, step: int = 0, is_output_to: Optional[str] = None, timestamp : Optional[int] = None) -> None:
    PROV4ML_DATA.add_metric(key, value, step, is_input_to=is_input_to, is_output_to=is_output_to, timestamp=timestamp)

def log_current_execution_time(label: str, is_input_to: Optional[str] = None, is_output_to: Optional[str] = 'std.time', step: Optional[int] = None) -> None:
    return log_metric(label, time_utils.get_time(), is_input_to=is_input_to, is_output_to=is_output_to, step=step, is_input=False)

def log_param(key: str, value: Any, is_input_to : Optional[str] = None, is_output_to : Optional[str] = None) -> None:
    PROV4ML_DATA.add_parameter(key, value, is_input_to, is_output_to)

def _get_model_memory_footprint(model: Union[torch.nn.Module, Any]) -> dict:

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
        warnings.warn("Could not determine precision, defaulting to 32 bits. Please make sure to provide a model with a trainer attached, this is often due to calling this before the trainer.fit() method")
        precision = 32
    
    precision_megabytes = precision / 8 / 1e6

    memory_per_model = total_params * precision_megabytes
    memory_per_grad = total_params * 4 * 1e-6
    memory_per_optim = total_params * 4 * 1e-6
    
    ret = {}
    ret[f"{PROV4ML_DATA.yProv_PREFIX}:TotalParameters"] = total_params
    ret[f"{PROV4ML_DATA.yProv_PREFIX}:WeightsMemoryLoad"] = memory_per_model
    ret[f"{PROV4ML_DATA.yProv_PREFIX}:ModelTotalMemoryLoad"] = memory_per_model + memory_per_grad + memory_per_optim

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

    return {f"{PROV4ML_DATA.yProv_PREFIX}:layersDescriptionFilePath": f"{path}/{model_name}_layers_description.json"}

def log_model(
        model_name: str, 
        model: Union[torch.nn.Module, Any], 
        is_input_to : Optional[str] = None, 
        is_output_to : Optional[str] = None, 
        log_model_info: bool = True, 
        log_model_layers : bool = False,
        other_attrs : Optional[bool] = None
    ) -> None:      
    e = save_model_version(model_name, model, is_input_to=is_input_to, is_output_to=is_output_to, other_attrs=other_attrs)

    if log_model_info:
        d = _get_model_memory_footprint(model)
        e.add_attributes(d)

    if log_model_layers: 
        d = _get_model_layers_description(model_name, model)
        e.add_attributes(d)
    
def log_flops_per_epoch(label: str, model: Any, dataset: Any, is_input_to : Optional[str] = None, is_output_to: Optional[str] = 'fvcore.nn.FlopCountAnalysis', step: int = 0) -> None:
    return log_metric(label, flops_utils.get_flops_per_epoch(model, dataset), is_input_to=is_input_to, is_output_to=is_output_to, step=step)

def log_flops_per_batch(label: str, model: Any, batch: Any, is_input_to : Optional[str] = None, is_output_to: Optional[str] = 'fvcore.nn.FlopCountAnalysis', step: int = 0) -> None:
    return log_metric(label, flops_utils.get_flops_per_batch(model, batch), is_input_to=is_input_to, is_output_to=is_output_to, step=step)

def log_system_metrics(is_input_to : Optional[str] = None, is_output_to: Optional[str] = None, step: int = 0) -> None:
    data = PROV4ML_DATA.tracker.checkpoint()
    is_output_to = is_output_to or data["hardware_access_mode"]
    timestamp = funcs.get_current_time_millis()
    log_metric("cpu_usage_pct", data["cpu_utilization_pct"], is_input_to=is_input_to, is_output_to=is_output_to, step=step, timestamp=timestamp)
    log_metric("cpu_power", data["cpu_power_watts"], is_input_to=is_input_to, is_output_to=is_output_to, step=step, timestamp=timestamp)
    log_metric("cpu_energy", data["cpu_energy_delta_joules"], is_input_to=is_input_to, is_output_to=is_output_to, step=step, timestamp=timestamp)
    log_metric("ram_usage_pct", data["ram_used_pct"], is_input_to=is_input_to, is_output_to=is_output_to, step=step, timestamp=timestamp)
    log_metric("ram_usage_gb", data["ram_used_gb"], is_input_to=is_input_to, is_output_to=is_output_to, step=step, timestamp=timestamp)
    log_metric("ram_power", data["ram_power_watts"], is_input_to=is_input_to, is_output_to=is_output_to, step=step, timestamp=timestamp)
    log_metric("ram_energy", data["ram_energy_delta_joules"], is_input_to=is_input_to, is_output_to=is_output_to, step=step, timestamp=timestamp)
    log_metric("disk_usage_gb", data["disk_used_gb"], is_input_to=is_input_to, is_output_to=is_output_to, step=step, timestamp=timestamp)
    log_metric("disk_usage_pct", data["disk_used_pct"], is_input_to=is_input_to, is_output_to=is_output_to, step=step, timestamp=timestamp)
    log_metric("gpu_usage_pct", data["gpu_utilization_pct"], is_input_to=is_input_to, is_output_to=is_output_to, step=step, timestamp=timestamp)
    log_metric("gpu_power", data["gpu_power_watts"], is_input_to=is_input_to, is_output_to=is_output_to, step=step, timestamp=timestamp)
    log_metric("gpu_energy", data["gpu_energy_delta_joules"], is_input_to=is_input_to, is_output_to=is_output_to, step=step, timestamp=timestamp)
    log_metric("gpu_memory_usage_gb", data["gpu_memory_used_gb"], is_input_to=is_input_to, is_output_to=is_output_to, step=step, timestamp=timestamp)
    log_metric("gpu_memory_usage_pct", data["gpu_memory_used_pct"], is_input_to=is_input_to, is_output_to=is_output_to, step=step, timestamp=timestamp)
    log_metric("gpu_memory_power", data["gpu_memory_power_watts"], is_input_to=is_input_to, is_output_to=is_output_to, step=step, timestamp=timestamp)
    log_metric("gpu_temperature_c", data["gpu_temperature_c"], is_input_to=is_input_to, is_output_to=is_output_to, step=step, timestamp=timestamp)

def log_artifact(
        artifact_name : str, 
        artifact_path : str, 
        is_input_to : Optional[str] = None, 
        is_output_to : Optional[str] = None, 
        step: int = 0, 
        is_model : bool = False, 
        other_attrs : Optional[dict] = None, 
    ) -> prov.ProvEntity:
    return PROV4ML_DATA.add_artifact(
        artifact_name=artifact_name, 
        artifact_path=artifact_path, 
        step=step, 
        is_input_to=is_input_to, 
        is_output_to=is_output_to, 
        is_model=is_model, 
        attributes=other_attrs, 
    )

def save_model_version(
    model_name: Union[str, Path], 
    model: Union[torch.nn.Module, Any], 
    is_input_to : Optional[str] = None, 
    is_output_to : Optional[str] = None, 
    step: Optional[int] = None, 
    other_attrs : Optional[dict] = None, 
) -> prov.ProvEntity:

    model_path = Path(model_name)
    if model_path.parent != Path('.'):
        model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.suffix != '.pt':
        file_name = str(model_path.with_suffix('.pt'))
    else:
        file_name = str(model_path)
    state_dict = model.state_dict() if isinstance(model, torch.nn.Module) else model

    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    model_bytes = buffer.getvalue()
    model_hash = hashlib.sha256(model_bytes).hexdigest()

    if other_attrs is None: 
        other_attrs = {}
    other_attrs[f"{PROV4ML_DATA.yProv_PREFIX}:checksum"] = model_hash

    torch.save(state_dict, file_name)
    return log_artifact(
        model_path.name, file_name, 
        is_input_to=is_input_to, is_output_to=is_output_to, step=step, 
        is_model=True
    )
    
def compute_dataset_checksum(dataset: Union[DataLoader, Subset, Dataset]) -> str:
    hasher = hashlib.sha256()
    if isinstance(dataset, DataLoader):
        hasher.update(f"batch_size={dataset.batch_size}".encode())
        hasher.update(f"num_workers={dataset.num_workers}".encode())
        dataset = dataset.dataset
    if isinstance(dataset, Subset):
        indices_str = ",".join(map(str, sorted(dataset.indices)))
        hasher.update(indices_str.encode())
        dataset = dataset.dataset
    hasher.update(type(dataset).__name__.encode())
    hasher.update(str(len(dataset)).encode())
    if hasattr(dataset, 'tensors'):
        for tensor in dataset.tensors:
            hasher.update(tensor.detach().cpu().numpy().tobytes())
    elif hasattr(dataset, 'samples') or hasattr(dataset, 'imgs'):
        file_paths = [s[0] for s in getattr(dataset, 'samples', getattr(dataset, 'imgs', []))]
        for path in sorted(file_paths):
            hasher.update(str(path).encode())
    return hasher.hexdigest()

def log_dataset(
        dataset_label : str, 
        dataset : Union[DataLoader, Subset, Dataset], 
        is_input_to : Optional[str] = None, 
        is_output_to : Optional[str] = None, 
        log_dataset_info : bool = True, 
        other_attrs: Optional[dict] = None
        ): 

    other_attrs[f"{PROV4ML_DATA.yProv_PREFIX}:checksum"] = compute_dataset_checksum(dataset)
    e = log_artifact(dataset_label, "", is_input_to=is_input_to, is_model=False, is_output_to=is_output_to, other_attrs=other_attrs)
    
    if not log_dataset_info: return

    if isinstance(dataset, DataLoader):
        dl = dataset
        dataset = dl.dataset
        attrs = {
            f"prov:type": "provml:Dataset", 
            f"{PROV4ML_DATA.yProv_PREFIX}:datasetStatBatchSize": dl.batch_size, 
            f"{PROV4ML_DATA.yProv_PREFIX}:datasetStatNumWorkers": dl.num_workers, 
            f"{PROV4ML_DATA.yProv_PREFIX}:datasetStatShuffle": isinstance(dl.sampler, RandomSampler), 
            f"{PROV4ML_DATA.yProv_PREFIX}:datasetStatTotalSteps": len(dl), 
            f"{PROV4ML_DATA.yProv_PREFIX}:datasetStatTotalSamples": len(dataset) * dl.batch_size,
            f"{PROV4ML_DATA.yProv_PREFIX}:datasetStatTotalBatches": len(dataset), 
        }
        e.add_attributes(attrs)

    elif isinstance(dataset, Subset):
        dl = dataset
        dataset = dl.dataset
        e.add_attributes({f"{PROV4ML_DATA.yProv_PREFIX}:datasetStatTotalSteps": len(dl)})
        e.add_attributes({f"{PROV4ML_DATA.yProv_PREFIX}:datasetStatTotalSamples": len(dataset)})
        
    elif isinstance(dataset, Dataset): 
        e.add_attributes({f"{PROV4ML_DATA.yProv_PREFIX}:datasetStatTotalSamples": len(dataset)})

def log_execution_command(full_command_str: Optional[str] = None, path : Optional[str] = None) -> None:
    if not full_command_str:
        full_command_str = " ".join([sys.executable] + sys.argv)
    if not path: 
        path = sys.argv[0]

    tokens = shlex.split(full_command_str)
    raw_args = tokens[1:] if len(tokens) > 1 else []
    
    full_path = sys.executable + " " + os.path.join(
        "/workspace", 
        f"{PROV4ML_DATA.CLEAN_EXPERIMENT_NAME}_{PROV4ML_DATA.RUN_ID}", 
        "artifacts", 
        path
    )
    
    log_param("execution_executable", sys.executable)
    log_param("reproducibility_command", full_path)
    log_param("execution_arguments", " ".join(raw_args))
    log_param("execution_command", full_command_str)

def log_source_code(use_git : bool = False) -> None:
    PROV4ML_DATA.request_source_code(use_git=use_git)

# def log_context(context : str, is_subcontext_of : Optional[str] = None, source_of_data : Optional[str] = None): 
#     PROV4ML_DATA._add_ctx(is_subcontext_of, context, source=source_of_data)
