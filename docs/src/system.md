
# System Metrics

The yprov4ml.log_system_metrics function logs critical system performance metrics during machine learning experiments.
The information logged is related to the time between the last call to the function and the current call.

```python
yprov4ml.log_system_metrics(
    is_input_to: Optional[str] = None,
    is_output_to: Optional[str] = None,
    step: int = 0,
)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `is_input_to` | `Optional[str]` | **Optional**. Activity that consumes the metrics |
| `is_output_to` | `Optional[str]` | **Optional**. Producer of the metrics; defaults to the detected hardware access mode |
| `step` | `int` | **Optional**. Step of the metric (default 0) |

This function logs the following system metrics:

| Parameter | Description | Unit |
| :-------- | :---------- | :--- |
| `cpu_usage_pct` | CPU utilization percentage | % |
| `cpu_power` | Power consumed by the CPU | Watts (W) |
| `cpu_energy` | Energy consumed by the CPU over the logging interval | Joules (J) |
| `ram_usage_pct` | Percentage of total RAM used | % |
| `ram_usage_gb` | Amount of RAM used in gigabytes | Gigabytes (GB) |
| `ram_power` | Power consumed by the system RAM | Watts (W) |
| `ram_energy` | Energy consumed by RAM over the logging interval | Joules (J) |
| `disk_usage_gb` | Amount of disk space used in gigabytes | Gigabytes (GB) |
| `disk_usage_pct` | Percentage of total disk space used | % |
| `gpu_usage_pct` | GPU compute utilization percentage | % |
| `gpu_power` | Power consumed by the GPU | Watts (W) |
| `gpu_energy` | Energy consumed by the GPU over the logging interval | Joules (J) |
| `gpu_memory_usage_gb` | Amount of GPU VRAM used in gigabytes | Gigabytes (GB) |
| `gpu_memory_usage_pct` | Percentage of GPU VRAM used | % |
| `gpu_memory_power` | Power consumed by the GPU VRAM module | Watts (W) |
| `gpu_temperature_c` | Temperature of the GPU core | Degrees Celsius (°C) |


# FLOPs per Epoch

The log_flops_per_epoch function logs the number of floating-point operations (FLOPs) performed per epoch for a given model and dataset. 

```python
yprov4ml.log_flops_per_epoch(
    label: str, 
    model: Union[torch.nn.Module, Any],
    dataset: Union[torch.utils.data.Dataset, torch.utils.data.DataLoader, torch.utils.data.Subset], 
    is_input_to: Optional[str] = None, 
    is_output_to: Optional[str] = 'fvcore.nn.FlopCountAnalysis', 
    step: int = 0
):
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `label` | `string` | **Required**. Label of the FLOPs |
| `model` | `Union[torch.nn.Module, Any]` | **Required**. Model used for the FLOPs calculation |
| `dataset` | `Dataset \| DataLoader \| Subset` | **Required**. Dataset used for the FLOPs calculation |
| `is_input_to` | `Optional[str]` | **Optional**. Activity that consumes the metric |
| `is_output_to` | `Optional[str]` | **Optional**. Producer of the metric (default `fvcore.nn.FlopCountAnalysis`) |
| `step` | `int` | **Optional**. Step of the metric (default 0) |

# FLOPs per Batch

The log_flops_per_batch function logs the number of floating-point operations (FLOPs) performed per batch for a given model and batch of data. 

```python
yprov4ml.log_flops_per_batch(
    label: str, 
    model: Union[torch.nn.Module, Any],
    batch: Any, 
    is_input_to: Optional[str] = None, 
    is_output_to: Optional[str] = 'fvcore.nn.FlopCountAnalysis', 
    step: int = 0, 
):
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `label` | `string` | **Required**. Label of the FLOPs |
| `model` | `Union[torch.nn.Module, Any]` | **Required**. Model used for the FLOPs calculation |
| `batch` | `Any` | **Required**. Batch of data used for the FLOPs calculation |
| `is_input_to` | `Optional[str]` | **Optional**. Activity that consumes the metric |
| `is_output_to` | `Optional[str]` | **Optional**. Producer of the metric (default `fvcore.nn.FlopCountAnalysis`) |
| `step` | `int` | **Optional**. Step of the metric (default 0) |


<div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px;">
    <a href="custom_metrics.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">← Prev</a>
    <a href="." style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">🏠 Home</a>
    <a href="time.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">Next →</a>
</div>