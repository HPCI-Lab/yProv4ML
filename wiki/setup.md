
# Setup

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


## MLFlow and Provenance Graph Creation

To view prov information in mlflow:

```bash
mlflow server
```

Or, to specify the backend store uri and port number: 

```bash
mlflow ui --backend-store-uri ./path_to_mlflow_logs --port free_port
```

Where `--backend-store-uri` has to point to the subdirectory containing the `models` folder.

To generate the graph svg image: 

```bash
dot -Tsvg -O prov_graph.dot
```
