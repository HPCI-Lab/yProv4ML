# Querying from Provenance Files

yProv4ml offers a set of directives to easily extract the information logged from the provenance.json file. 

<div style="display: flex; align-items: center; background-color: #ffcc00; color: #333; border: 5px solid #ffcc00; font-weight: bold; border-radius: 5px; position: relative;">
    <span style="position: absolute; left: 10px; font-size: 20px;">⚠</span>
    <span style="margin-left: 55px; padding: 5px; background-color: white; border-radius: 5px; width:100%">
    All these functions expect the data to be passed to be a dictionary (json file opened in python). When using a provenance json file coming from yProv4ML, this can be easily obtained following the example below. 
    </span>
</div>

<div style="display: flex; align-items: center; margin: 20px 0;">
    <hr style="flex-grow: 0.05; border: 2px solid #009B77; margin: 0;">
    <span style="background: white; padding: 0 10px; font-weight: bold; color: #009B77;">Example:</span>
    <hr style="flex-grow: 1; border: 2px solid #009B77; margin: 0;">
</div>

```python
from yprov4ml import (
    list_activities, 
    list_entities, 
    get_parameter, 
    list_parameters,
    list_metrics, 
    list_metric_paths, 
    get_metric
)

import json
data = json.load(open(path_to_prov_json))     
```

<hr style="border: 2px solid #009B77; margin: 20px 0;">
### Utility Functions

#### Listing Functions

```python 
def list_activities(source : dict | str) -> list[str]
def list_entities(source : dict | str, entity_type: str | None = None) -> list[str]
```

- **`list_activities`**: Retrieves a list of all activity names stored in the provenance document.
- **`list_entities`**: Retrieves a list of entity names from the provenance document. Can be filtered by passing an explicit `entity_type` (e.g., `"provml:Metric"`).

---

#### Parameter Retrieval

```python 
def get_parameter(source : dict | str, name: str, param: str, unwrap: bool = True) -> Any
```

Retrieves a single parameter value associated with a specified activity or entity key name.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source` | `dict \| str` | *Required* | Loaded PROV dictionary or file path. |
| `name` | `str` | *Required* | Name of the activity or entity. |
| `param` | `str` | *Required* | Specific attribute or parameter key to extract. |
| `unwrap` | `bool` | `True` | Automatically unwraps PROV typed-literals into native Python values. |

```python 
def list_parameters(data : dict | str, name: str | None = None, unwrap: bool = True) -> dict[str, Any]
```

Retrieves a dictionary of key-value parameters. If `name` is omitted, it aggregates parameters across all entities and activities in the file.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `dict \| str` | *Required* | Loaded PROV dictionary or file path. |
| `name` | `str \| None` | `None` | Target activity/entity name. If `None`, extracts all available parameters. |
| `unwrap` | `bool` | `True` | Automatically unwraps PROV typed-literals into native Python values. |

---

#### Metric Management

```python 
def list_metrics(data : dict | str, context: str | None = None, source: str | None = None) -> pd.DataFrame
```

Summarizes all metric entities recorded in the provenance JSON into a single Pandas DataFrame with metadata columns (`label`, `context`, `source`, `csv_path`).

```python 
def list_metric_paths(data : dict | str, context: str | None = None, source : str | None = None, file_type : str | None = None) -> dict[str, str]
```

Returns a dictionary mapping metric entity identifiers to their underlying dataset file paths.

```python 
def get_metric(data : dict | str, name: str | None = None, context: str | None = None, source : str | None = None)
```

Fetches and automatically loads the data object for a metric (named in the format `{name}_{context}_{source}`) using the appropriate reader.

---

#### Project Helpers

```python 
def list_runs_in_proj(path: str | Path) -> list[Path]
def list_provjson_in_proj(path : str | Path) -> list[Path]
```

- **`list_runs_in_proj`**: Returns paths to all run subdirectories found within a given project folder.
- **`list_provjson_in_proj`**: Searches across run directories to locate all available `.json` provenance files.



<div style="display: flex; align-items: center; background-color: #ffcc00; color: #333; border: 5px solid #ffcc00; font-weight: bold; border-radius: 5px; position: relative;">
    <span style="position: absolute; left: 10px; font-size: 20px;">⚠</span>
    <span style="margin-left: 55px; padding: 5px; background-color: white; border-radius: 5px; width:100%">
    Viewing metrics data depends on the way it is saved in the experiment. 
    - If CSV format is used, we suggest opening it with [pandas](https://pandas.pydata.org/)
    - If ZARR or NETCDF are used, then either [xarray](https://docs.xarray.dev/en/stable/index.html) or an ad-hoc library ([zarr-python](https://zarr.readthedocs.io/en/stable/) and [netcdf4](https://pypi.org/project/netCDF4/)) can be used. 
    </span>
</div>


<div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px;">
    <a href="prov_viewer.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">← Prev</a>
    <a href="." style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">🏠 Home</a>
    <a href="examples.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">Next →</a>
</div>