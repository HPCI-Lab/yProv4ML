# Prov4ML

[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)

This library is a wrapper around MLFlow to provide a unified interface for logging and tracking provenance information in machine learning experiments. 

It allows users to create provenance graphs from the logged information.

## Installation

To install the library, run the following command:

```bash
pip install prov4ml
```

## Example

![Example](./assets/example.svg)

The image shown above has been generated from the [example](./examples/mlflow_lightning.py) program provided in the ```example``` directory.

## Experiments and Runs

An experiment is a collection of runs. Each run is a single execution of a machine learning model. 
By changing the ```experiment_name``` parameter in the ```start_run``` function, the user can create a new experiment. 
All artifacts and metrics logged during the execution of the experiment will be saved in the directory specified by the experiment ID. 

Several runs can be executed in the same experiment. All runs will be saved in the same directory (according to the specific experiment name and ID).

## Available Commands

### Setup

Before using the library, the user must set up the MLFlow execution, as well as library specific configurations: 

```python
prov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="experiment_name", 
    provenance_save_dir="prov_dir", 
    mlflow_save_dir="mlflow_dir", 
)
```

The parameters are as follows:

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `prov_user_namespace` | `string` | **Required**. User namespace for the provenance graph |
| `experiment_name` | `string` | **Required**. Name of the experiment |
| `provenance_save_dir` | `string` | **Required**. Directory to save the provenance graph |
| `mlflow_save_dir` | `string` | **Required**. Directory to save the mlflow logs |
| `nested` | `bool` | **Optional**. Whether to create a nested directory for the experiment |
| `tags` | `dict` | **Optional**. Tags to add to the experiment |
| `description` | `string` | **Optional**. Description of the experiment |
| `log_system_metrics` | `bool` | **Optional**. Whether to log system metrics |

At the end of the experiment, the user must end the run:

```python
prov4ml.end_run()
```

This call allows the library to save the provenance graph in the specified directory. 

The final necessary call to save the graph is the following:

```python
prov4ml.log_model(model, "model_name")
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `model` | `torch.nn.Module` | **Required**. Model to log |
| `model_name` | `string` | **Required**. Name of the model |
| `log_model_info` | `bool` | **Optional**. Whether to log model statistics |

It sets the model for the current experiment. It can be called anywhere before the end of the experiment. 
The same call also logs some model information, such as the number of parameters and the model architecture memory footprint. 
The saving of these information can be toggled with the ```log_model_info = False``` parameter.

### General Logging

When logging parameters and metrics, the user must specify the context of the information. 
The available contexts are: 
 - `TRAINING`: adds the information to the training context  
 - `VALIDATION`: adds the information to the validation context
 - `TESTING`: adds the information to the testing context

##### Log Parameters

To specify arbitrary training parameters used during the execution of the experiment, the user can call the following function. 
    
```python
prov4ml.log_param("param_name", "param_value")
```

##### Log Metrics

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

### Utility Logging

**Prov4ml** also provides utility functions to log system metrics, carbon metrics, and execution time.

##### System Metrics

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


##### Carbon Metrics

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

##### Execution Time

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

## MLFlow and Provenance Graph Creation

To view prov information in mlflow:

```bash
mlflow server
```

Or: 

```bash
mlflow ui --backend-store-uri ./path_to_mlflow_logs --port free_port
```

Where `--backend-store-uri` has to point to the subdirectory containing the `models` folder.

To generate the graph svg image: 

```bash
dot -Tsvg -O prov_graph.dot
```
