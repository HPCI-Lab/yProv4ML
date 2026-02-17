
import pandas as pd
import json

def get_activities(json_file_or_path): 
    if isinstance(json_file_or_path, str): 
        json_file_or_path = json.load(open(json_file_or_path))
    return list(json_file_or_path["activity"].keys())

def get_parameters(json_file_or_path): 
    if isinstance(json_file_or_path, str): 
        json_file_or_path = json.load(open(json_file_or_path))

    parameters = []
    for ent_name, ent in json_file_or_path["entity"].items(): 
        ks = ent.keys()
        if "prov:label" in ks and "prov:value" in ks: 
            parameters.append(ent_name)
    return parameters

def get_parameter(json_file_or_path, parameter_name): 
    if isinstance(json_file_or_path, str): 
        json_file_or_path = json.load(open(json_file_or_path))

    return json_file_or_path["entity"][parameter_name]

def get_metrics(json_file_or_path): 
    if isinstance(json_file_or_path, str): 
        json_file_or_path = json.load(open(json_file_or_path))
    
    metrics = []
    for ent_name, ent in json_file_or_path["entity"].items(): 
        if "prov:type" in ent.keys():
            if ent["prov:type"] == "provml:Metric": 
                metrics.append(ent_name)

    return metrics

def get_metric_metadata(json_file_or_path, metric_name): 
    if isinstance(json_file_or_path, str): 
        json_file_or_path = json.load(open(json_file_or_path))

    return json_file_or_path["entity"][metric_name]

def get_metric_data(json_file_or_path, metric_name): 
    if isinstance(json_file_or_path, str): 
        json_file_or_path = json.load(open(json_file_or_path))

    metric_path = json_file_or_path["entity"][metric_name]["dcterms:identifier"]
    return pd.read_csv(metric_path)