from __future__ import annotations
import os
import ast
import json
from typing import Any
import pandas as pd
import xarray as xr
from pathlib import Path

def _unwrap(value: Any) -> Any:
    """
    Unwrap a PROV typed-literal to its plain Python value.

    Handles three shapes:
      • {"$": v, "type": "xsd:*"}   – already a dict
      • "{'$': v, 'type': '...'}"   – stringified dict (from JSON serialisation)
      • anything else               – returned as-is
    """
    if isinstance(value, dict) and "$" in value:
        return value["$"]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") and "'$'" in stripped:
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, dict) and "$" in parsed:
                    return parsed["$"]
            except (ValueError, SyntaxError):
                pass
    return value

def _is_metric(entity: dict) -> bool:
    return entity.get("prov:type") == "provml:Metric"

def _get_source(source : dict | str): 
    if isinstance(source, dict):
        _doc = source
    else:
        with open(source, encoding="utf-8") as fh:
            _doc = json.load(fh)
    return _doc

def _open_file(path: str): 
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".nc"):
        return xr.open_dataset(path)
    elif path.endswith(".zarr"):
        return xr.open_zarr(path)
    else: 
        return KeyError()


def list_activities(source : dict | str) -> list[str]:
    _doc = _get_source(source)
    return list(_doc.get("activity", {}).keys())


def list_entities(source : dict | str, entity_type: str | None = None) -> list[str]:
    _doc = _get_source(source)
    if entity_type is None:
        return list(_doc.get("entity", {}).keys())
    return [name for name, attrs in _doc.get("entity", {}).items() if attrs.get("prov:type") == entity_type]


def get_parameter(source : dict | str, name: str, param: str, unwrap: bool = True) -> Any:
    _doc = _get_source(source)

    record = (_doc.get("activity", {}).get(name) or _doc.get("entity", {}).get(name))
    if record is None:
        raise KeyError(
            f"No activity or entity named {name!r}. "
            f"Use list_activities() / list_entities() to see valid names."
        )

    if param in record:
        val = record[param]
        return _unwrap(val) if unwrap else val

    for prefix in ("yprov:", "prov:", "dcterms:"):
        candidate = prefix + param
        if candidate in record:
            val = record[candidate]
            return _unwrap(val) if unwrap else val

    return None


def list_parameters(data : dict | str, name: str | None = None, unwrap: bool = True) -> dict[str, Any]:
    _doc = _get_source(data)
    if name: 
        record = (_doc.get("activity", {}).get(name) or _doc.get("entity", {}).get(name))

        if record is None:
            raise KeyError(f"No activity or entity named {name!r}.")

        if not unwrap:
            return dict(record)
        return {k: _unwrap(v) for k, v in record.items()}
    else: 
        res = {}
        for record in (_doc.get("activity", {}) | _doc.get("entity", {})).values(): 
            res |= {k: _unwrap(v) for k, v in record.items()}
        return res


def list_metrics(data : dict | str, context: str | None = None, source: str | None = None) -> pd.DataFrame:
    _doc = _get_source(data)
    
    rows = {}
    for name, attrs in _doc.get("entity", {}).items():
        if not _is_metric(attrs):
            continue
        if context is not None and attrs.get("yprov:context") != context:
            continue
        if source is not None and attrs.get("yprov:source") != source:
            continue
        rows[name] = {k: _unwrap(v) for k, v in attrs.items()}

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "metric_entity"

    # Convenience: expose short metric name and context as top-level cols
    if "prov:label" in df.columns:
        df.insert(0, "label", df.pop("prov:label"))
    if "yprov:context" in df.columns:
        df.insert(1, "context", df.pop("yprov:context"))
    if "yprov:source" in df.columns:
        df.insert(2, "source", df.pop("yprov:source"))
    if "dcterms:identifier" in df.columns:
        df.insert(3, "csv_path", df.pop("dcterms:identifier"))

    return df


def list_metric_paths(data : dict | str, context: str | None = None, source : str | None = None, file_type : str | None = None) -> dict[str, str]:
    _doc = _get_source(data)
    result = {}
    for name, attrs in _doc.get("entity", {}).items():
        if not _is_metric(attrs):
            continue
        if context is not None and attrs.get("yprov:context") != context:
            continue
        if source is not None and attrs.get("yprov:source") != source:
            continue
        csv_id = attrs.get("dcterms:identifier", "")
        if file_type is None or file_type == csv_id.split(".")[-1]:  
            result[name] = csv_id
    return result


def get_metric(data : dict | str, name: str | None = None, context: str | None = None, source : str | None = None): 
    _doc = _get_source(data)
    for name, attrs in _doc.get("entity", {}).items():
        if not _is_metric(attrs):
            continue
        if context is not None and attrs.get("yprov:context") != context:
            continue
        if source is not None and attrs.get("yprov:source") != source:
            continue
        csv_id = attrs.get("dcterms:identifier", "")
        if name == csv_id: 
            return _open_file(csv_id)


def list_runs_in_proj(path: str | Path): 
    proj_dir = Path(path)
    runs = [proj_dir / f for f in os.listdir(proj_dir)]
    return runs


def list_provjson_in_proj(path : str | Path): 
    runs = list_runs_in_proj(path)
    jsons = []
    for run in runs: 
        files = [run / f for f in os.listdir(run)]
        for file in files: 
            if str(file).endswith(".json"): 
                jsons.append(file)
                break
    return jsons