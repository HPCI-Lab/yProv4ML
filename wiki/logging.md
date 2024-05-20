
# General Logging

When logging parameters and metrics, the user must specify the context of the information. 
The available contexts are: 
 - `TRAINING`: adds the information to the training context  
 - `VALIDATION`: adds the information to the validation context
 - `TESTING`: adds the information to the testing context

## Log Parameters

To specify arbitrary training parameters used during the execution of the experiment, the user can call the following function. 
    
```python
prov4ml.log_param("param_name", "param_value")
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `key` | `string` | **Required**. Name of the parameter |
| `value` | `string` | **Required**. Value of the parameter |

The `log_params` logs multiple parameter key-value pairs to the MLflow tracking context.

```python
prov4ml.log_params(params)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `params` | `dict` | **Required**. Dictionary of parameters to log |


## Log Metrics

To specify metrics, which can be tracked during the execution of the experiment, the user can call the following function.

```python
prov4ml.log_metric("metric_name",metric_value,prov4ml.Context.TRAINING, step=current_epoch)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `key` | `string` | **Required**. Name of the metric |
| `value` | `float` | **Required**. Value of the metric |
| `context` | `prov4ml.Context` | **Required**. Context of the metric |
| `step` | `int` | **Optional**. Step of the metric |
| `synchronous` | `bool` | **Optional**. Whether to log the metric synchronously |
| `timestamp` | `int` | **Optional**. Timestamp of the metric |

The *step* parameter is optional and can be used to specify the current time step of the experiment, for example the current epoch.

The `log_metrics` function logs the given metrics and their associated contexts to the active MLflow run.

```python
prov4ml.log_metrics(metrics, step, synchronous=False)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `metrics` | `dict` | **Required**. Dictionary of metrics to log |
| `step` | `int` | **Required**. Step of the metric |
| `synchronous` | `bool` | **Optional**. Whether to log the metric synchronously |

## Log Artifacts

To log artifacts, the user can call the following function.

```python
prov4ml.log_artifact("artifact_path")
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `artifact_path` | `string` | **Required**. Path to the artifact |

The function logs the artifact in the current experiment. The artifact can be a file or a directory. 
It uses the `mlflow.log_artifact` method, associating the artifact with the active run ID. 

## Log Models

```python
prov4ml.log_model(model, "model_name")
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `model` | `torch.nn.Module` | **Required**. Model to log |
| `model_name` | `string` | **Required**. Name of the model |
| `log_model_info` | `bool` | **Optional**. Whether to log model statistics |
| `log_as_artifact` | `bool` | **Optional**. Whether to log the model as an artifact |

It sets the model for the current experiment. It can be called anywhere before the end of the experiment. 
The same call also logs some model information, such as the number of parameters and the model architecture memory footprint. 
The saving of these information can be toggled with the ```log_model_info = False``` parameter.

```python
prov4ml.save_model_version(model, "model_name", context, step)
```

The save_model_version function saves the state of a PyTorch model and logs it as an artifact, enabling version control and tracking within machine learning experiments.

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `model`	| `torch.nn.Module` |	**Required**. The PyTorch model to be saved. |
| `model_name`	| `str`|	**Required**. The name under which to save the model. | 
| `context`	| `Context` |	**Required**. The context in which the model is saved. |
| `step`	| `Optional[int]` |	**Optional**. The step or epoch number associated with the saved model. |

This function saves the model's state dictionary to a specified directory and logs the saved model file as an artifact for provenance tracking. It ensures that the directory for saving the model exists, creates it if necessary, and uses the `torch.save` method to save the model. It then logs the saved model file using `log_artifact`, associating it with the given context and optional step number.


## Log Optimizer

The log_optimizer function logs details about a provided PyTorch optimizer to the MLFlow tracking context. 

```python
prov4ml.log_optimizer(optimizer)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `optimizer`	| `torch.optim.Optimizer` |	**Required**. The PyTorch optimizer to be logged. |

This function logs the name and learning rate of the provided optimizer to the MLFlow tracking context. It captures the class name of the optimizer and logs it as a parameter named `optimizer_name`. 
