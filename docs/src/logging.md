
# General Logging

In 4.0 the context of the information is expressed through the `is_input_to` and `is_output_to` arguments: strings naming the activities that consume or produce the item (for instance `"Training_0"`, `"Validation_0"`, `"Testing_0"`).

## Log Parameters

To specify arbitrary training parameters used during the execution of the experiment, the user can call the following function. 
    
```python
yprov4ml.log_param(
    key: str, 
    value: Any, 
    is_input_to : Optional[str] = None, 
    is_output_to : Optional[str] = None
)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `key` | `string` | **Required**. Name of the parameter |
| `value` | `string` | **Required**. Value of the parameter |
| `is_input_to` | `Optional[str]` | **Optional**. Activity that uses the parameter as input |
| `is_output_to` | `Optional[str]` | **Optional**. Activity or tool that produced the parameter |

## Log Artifacts

To log artifacts, the user can call the following function.

```python
yprov4ml.log_artifact(
    artifact_name : str, 
    artifact_path : str, 
    is_input_to : Optional[str] = None, 
    is_output_to : Optional[str] = None, 
    step: int = 0, 
    is_model : bool = False, 
    other_attrs : Optional[dict] = None, 
)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `artifact_name` | `string` | **Required**. Label to give to the artifact |
| `artifact_path` | `string` | **Required**. Path to the artifact |
| `is_input_to` | `Optional[str]` | **Optional**. Activity that uses the artifact as input |
| `is_output_to` | `Optional[str]` | **Optional**. Activity that generates the artifact |
| `step` | `int` | **Optional**. Step of the artifact (default 0) |
| `is_model` | `bool` | **Optional**. Whether the artifact is a model |
| `other_attrs` | `Optional[dict]` | **Optional**. Extra PROV attributes to attach to the artifact entity |

The function logs the artifact in the current experiment. The artifact can be a file or a directory. 
All logged artifacts are saved in the artifacts directory of the current experiment, while the related information is saved in the PROV-JSON file, along with a reference to the file. 
Use `is_input_to` for artifacts consumed by an activity (e.g. pretrained model weights) and `is_output_to` for artifacts it produces. 

## Log Models

```python
yprov4ml.log_model(
    model_name: str, 
    model: Union[torch.nn.Module, Any], 
    is_input_to : Optional[str] = None, 
    is_output_to : Optional[str] = None, 
    log_model_info: bool = True, 
    log_model_layers : bool = False,
    other_attrs : Optional[dict] = None,
)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `model_name` | `string` | **Required**. Name of the model |
| `model` | `Union[torch.nn.Module, Any]` | **Required**. The model to be logged |
| `is_input_to` | `Optional[str]` | **Optional**. Activity that uses the model as input |
| `is_output_to` | `Optional[str]` | **Optional**. Activity that produces the model |
| `log_model_info` | `bool` | **Optional**. Whether to log model information |
| `log_model_layers` | `bool` | **Optional**. Whether to log model layers |
| `other_attrs` | `Optional[dict]` | **Optional**. Extra PROV attributes to attach to the model entity |

It sets the model for the current experiment. It can be called anywhere before the end of the experiment. 
The same call also logs some model information, such as the number of parameters and the model architecture memory footprint. 
The saving of these information can be toggled with the ```log_model_info = False``` parameter. 
The model layers details can be logged in an external .json file, which will be linked to the provenance file as an artefact. 
The parameters saved for each layer depend on the type of the latter, but generally include input and output size, as well as dtype. 

```python
yprov4ml.save_model_version(
    model_name: str, 
    model: Union[torch.nn.Module, Any], 
    is_input_to : Optional[str] = None, 
    is_output_to : Optional[str] = None, 
    step: Optional[int] = None, 
    other_attrs : Optional[dict] = None, 
)
```

The save_model_version function saves the state of a PyTorch model and logs it as an artifact, enabling version control and tracking within machine learning experiments.

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `model_name`	| `Union[str, Path]`|	**Required**. The name (or path) under which to save the model; a `.pt` suffix is added if missing. | 
| `model`	| `torch.nn.Module` |	**Required**. The PyTorch model to be saved. |
| `is_input_to`	| `Optional[str]` |	**Optional**. Activity that uses the model as input. |
| `is_output_to`	| `Optional[str]` |	**Optional**. Activity that produces the model. |
| `step`	| `Optional[int]` |	**Optional**. The step or epoch number associated with the saved model. |
| `other_attrs`	| `Optional[dict]` |	**Optional**. Extra PROV attributes to attach to the entity. |

This function saves the model's state dictionary to a specified directory and logs the saved model file as an artifact for provenance tracking. It then logs the saved model file using `log_artifact`.

## Log Datasets

yProv4ML offers helper functions to log information and stats on specific datasets.  

```python
yprov4ml.log_dataset(
    dataset_label : str, 
    dataset : Union[DataLoader, Subset, Dataset], 
    is_input_to : Optional[str] = None, 
    is_output_to : Optional[str] = None, 
    log_dataset_info : bool = True, 
    other_attrs : dict = {}
)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `dataset_label` | `string` | **Required**. The label of the dataset |
| `dataset` | `Union[DataLoader, Subset, Dataset]` | **Required**. The dataset to be logged |
| `is_input_to` | `Optional[str]` | **Optional**. Activity that uses the dataset as input |
| `is_output_to` | `Optional[str]` | **Optional**. Activity that produces the dataset |
| `log_dataset_info` | `bool` | **Optional**. Whether to log the dataset information |
| `other_attrs` | `dict` | **Optional**. Extra PROV attributes to attach to the dataset entity |

The function logs the dataset in the current experiment. The dataset can be a DataLoader, a Subset, or a Dataset class from pytorch.
A SHA-256 checksum of the dataset is always recorded. Parameters which are logged include batch size, number of workers, whether the dataset is shuffled, the number of batches and the number of total samples. 

# Logging through Wrappers

yProv4ML offers helper wrappers to log information automatically. 

<div style="display: flex; align-items: center; background-color: #cc3300; color: #333; border: 5px solid #cc3300; font-weight: bold; border-radius: 5px; position: relative;">
    <span style="position: absolute; left: 10px; font-size: 20px;">‼</span>
    <span style="margin-left: 35px; padding: 5px; background-color: white; border-radius: 5px; width: 100%">These are only supported for PyTorch objects.  </span>
</div>


## ProvenanceTrackedFunction

```python
class ProvenanceTrackedFunction:
    def __init__(self, 
        func : Any, 
        is_input_to : Optional[str] = None, 
        is_output_to : Optional[str] = None
    )
```


| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| func	  | Any	          |  Required	| The target callable function or object to wrap (e.g., PyTorch loss function or metric evaluator). |
| is_input_to |	Optional[str]|	None	    | Activity the call is an input to; defaults to the wrapped object's type name. |
| is_output_to |	Optional[str]|	None	    | Activity or tool that produces the output. |


## WeightDistributionTrackedModel

```python
class WeightDistributionTrackedModel(nn.Module):
    def __init__(self, 
        model_label : str, 
        model : Any, 
        is_input_to : Optional[str] = None, 
        is_output_to : Optional[str] = None, 
        chunk_size : int = 64
    )
```

Tracks per-layer weight statistics per epoch. Arguments follow the same convention as the other 4.0 logging calls.

## ProvenanceTrackedModel

```python
class ProvenanceTrackedModel(nn.Module):
    def __init__(self, 
        model_label : str, 
        model : Any, 
        context : Optional[str] = None, 
        chunk_size : int = 64
    )
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_label` | `str` | *Required* | Label identifier for the model. Used as the output Zarr dataset name (`<model_label>.zarr`). |
| `model` | `Any` (`nn.Module`) | *Required* | The underlying PyTorch model instance to track. |
| `context` | `Optional[str]` | `None` | Optional execution context (e.g. stage or step name) associated with provenance tracking. |
| `chunk_size` | `int` | `64` | Zarr chunk dimension along the batch axis for layer datasets. |



<div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px;">
    <a href="prov_graph.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">← Prev</a>
    <a href="." style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">🏠 Home</a>
    <a href="prov_collection.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">Next →</a>
</div>