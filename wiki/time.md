

# Execution Time

```python
prov4ml.log_current_execution_time("code_portion_label", prov4ml.Context.TRAINING, step=current_epoch)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `label` | `string` | **Required**. Label of the code portion |
| `context` | `prov4ml.Context` | **Required**. Context of the metric |
| `step` | `int` | **Optional**. Step of the metric |
| `synchronous` | `bool` | **Optional**. Whether to log the metric synchronously |
| `timestamp` | `int` | **Optional**. Timestamp of the metric |

The *log_current_execution_time* function logs the current execution time of the code portion specified by the label.

```python
prov4ml.log_execution_start_time()
```

The *log_execution_start_time* function logs the start time of the current execution. 
It is automatically called at the beginning of the experiment.

```python
prov4ml.log_execution_end_time()
```

The *log_execution_end_time* function logs the end time of the current execution.
It is automatically called at the end of the experiment.
