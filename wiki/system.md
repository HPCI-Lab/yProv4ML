
# System Metrics

The prov4ml.log_system_metrics function logs critical system performance metrics during machine learning experiments.
The information logged is related to the time between the last call to the function and the current call.

```python
prov4ml.log_system_metrics(prov4ml.Context.TRAINING,step=current_epoch)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `context` | `prov4ml.Context` | **Required**. Context of the metric |
| `step` | `int` | **Optional**. Step of the metric |
| `synchronous` | `bool` | **Optional**. Whether to log the metric synchronously |
| `timestamp` | `int` | **Optional**. Timestamp of the metric |

This function logs the following system metrics:

| Parameter | Description                | Unit |
| :-------- | :-------------------------: | :---: |
| `Memory usage` | Memory usage of the system | % |
| `Disk usage` | Disk usage of the system | % |
| `Gpu memory usage` | Memory usage of the GPU | % |
| `Gpu usage` | Usage of the GPU | % |


# FLOPs per Epoch

The log_flops_per_epoch function logs the number of floating-point operations (FLOPs) performed per epoch for a given model and dataset. 

```python
prov4ml.log_flops_per_epoch(label, dataset, context, step):
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `label` | `string` | **Required**. Label of the FLOPs |
| `dataset` | `string` | **Required**. Dataset used for the FLOPs calculation |
| `context` | `prov4ml.Context` | **Required**. Context of the metric |
| `step` | `int` | **Optional**. Step of the metric |

# FLOPs per Batch

The log_flops_per_batch function logs the number of floating-point operations (FLOPs) performed per batch for a given model and batch of data. 

```python
prov4ml.log_flops_per_batch(label, context, step):
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `label` | `string` | **Required**. Label of the FLOPs |
| `context` | `prov4ml.Context` | **Required**. Context of the metric |
| `step` | `int` | **Optional**. Step of the metric |