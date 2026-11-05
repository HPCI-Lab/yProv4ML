# Custom Metrics

## Log Metrics

To specify metrics, which can be tracked during the execution of the experiment, the user can call the following function.

```python
yprov4ml.log_metric(
    key: str, 
    value: float, 
    is_input_to: Optional[str] = None, 
    step: int = 0, 
    is_output_to: Optional[str] = None, 
    timestamp : Optional[int] = None
)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `key` | `string` | **Required**. Name of the metric |
| `value` | `float` | **Required**. Value of the metric |
| `is_input_to` | `Optional[str]` | **Optional**. Name of the activity (e.g. `"Training_0"`) that consumes the metric |
| `step` | `int` | **Optional**. Step of the metric (default 0) |
| `is_output_to` | `Optional[str]` | **Optional**. Name of the activity or tool that produced the metric |
| `timestamp` | `Optional[int]` | **Optional**. Time of logging in milliseconds; defaults to the current time |

The *step* parameter is optional and can be used to specify the current time step of the experiment, for example the current epoch, it defaults to 0.

> **Note (4.0):** the `Context` enum and the `context`/`source` arguments were replaced by `is_input_to` and `is_output_to`, which link the metric to activities in the provenance graph. They are plain strings (e.g. `"Training_0"`, `"Validation_0"`).
