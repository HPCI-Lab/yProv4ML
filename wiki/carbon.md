
# Carbon Metrics

The prov4ml.log_carbon_metrics function logs carbon-related system metrics during machine learning experiments. 
The information logged is related to the time between the last call to the function and the current call.

```python
prov4ml.log_carbon_metrics(prov4ml.Context.TRAINING,step=current_epoch)
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
| `Emissions` | Emissions of the system | gCO2eq |
| `Emissions rate` | Emissions rate of the system | gCO2eq/s |
| `CPU power` | Power usage of the CPU | W |
| `GPU power` | Power usage of the GPU | W |
| `RAM power` | Power usage of the RAM | W |
| `CPU energy` | Energy usage of the CPU | J |
| `GPU energy` | Energy usage of the GPU | J |
| `RAM energy` | Energy usage of the RAM | J |
| `Energy consumed` | Energy consumed by the system | J |
