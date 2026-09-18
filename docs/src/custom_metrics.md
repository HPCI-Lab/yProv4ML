# Custom Metrics

## Log Metrics

To specify metrics, which can be tracked during the execution of the experiment, the user can call the following function.

```python
prov4ml.log_metric(
    key: str, 
    value: float, 
    context: Optional[str] = None, 
    step: int = 0, 
    source: Optional[str] = None, 
    timestamp : int = 0
)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `key` | `string` | **Required**. Name of the metric |
| `value` | `float` | **Required**. Value of the metric |
| `context` | `Optional[prov4ml.str]` | **Required**. str of the metric |
| `step` | `Optional[int]` | **Optional**. Step of the metric |
| `source` | `Optional[str]` | **Optional**. Source of the metric |
| `timestamp` | `Optional[str]` | **Optional**. The time of logging of the current item in the metric |

The *step* parameter is optional and can be used to specify the current time step of the experiment, for example the current epoch, it defaults to 0.
In a similar manner, the *context* parameter can also be omitted, and it will default to the main experiment context. 
The *source* parameter is optional and can be used to specify the source of the metric, so for example which library the data comes from. If omitted, yProv4ML will try to automatically determine the origin. 
