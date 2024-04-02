# Prov4ML

This library is a wrapper around MLFlow to provide a unified interface for logging and tracking provenance information in machine learning experiments. 

It allows users to create provenance graphs from the logged information.

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

At the end of the experiment, the user must end the run:

```python
prov4ml.end_run()
```

This call allows the library to save the provenance graph in the specified directory. 

The final necessary call to save the graph is the following:

```python
prov4ml.log_model(model, "model_name")
```

It sets the model for the current experiment. It can be called anywhere before the end of the experiment.

### General Logging

When logging parameters and metrics, the user must specify the context of the information. 
The available contexts are: 
 - TRAINING
 - VALIDATION
 - TESTING

To specify arbitrary training parameters used during the execution of the experiment, the user can call the following function. 
    
```python
prov4ml.log_param("param_name", "param_value")
```

To specify metrics, which can be tracked during the execution of the experiment, the user can call the following function.

```python
prov4ml.log_metric("metric_name",metric_value,prov4ml.Context.TRAINING, step=current_time_step)
```

The *step* parameter is optional and can be used to specify the current time step of the experiment, for example the current epoch.

### Utility Logging

**Prov4ml** also provides utility functions to log system metrics, carbon metrics, and execution time.

```python
prov4ml.log_system_metrics(prov4ml.Context.TRAINING,step=current_time_step)
```

System metrics include: 
 - Memory usage
 - Disk usage
 - Gpu memory usage
 - Gpu usage


```python
prov4ml.log_carbon_metrics(prov4ml.Context.TRAINING,step=current_time_step)
```

Carbon metrics include:
 - emissions
 - emissions_rate
 - cpu_power
 - gpu_power
 - ram_power
 - cpu_energy
 - gpu_energy
 - ram_energy
 - energy_consumed


```python
prov4ml.log_current_execution_time("code_portion_label", prov4ml.Context.TRAINING, step=current_time_step)
```

The *log_current_execution_time* function logs the current execution time of the code portion specified by the label.

## MLFlow and Provenance Graph Creation

To view prov information in mlflow:

```bash
mlflow server
```

To generate the graph svg image: 

```bash
dot -Tsvg -O prov_graph.dot
```

## Example

![Example](./assets/example.svg)