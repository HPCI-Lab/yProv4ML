
import os
import torch
from typing import Optional
import time

def _safe_get_model_attr(dic, node, attr, attr_label=None):
    if attr_label is None: 
        attr_label = str(attr) 

    try: 
        if attr == "type": 
            dic[attr_label] = str(type(node))
        elif attr == "weight.dtype": 
            dic[attr_label] = str(node.weight.dtype)
        else: 
            dic[attr_label] = str(getattr(node, attr))
    except AttributeError: 
        pass

def prov4ml_experiment_matches(experiment_name : str, exp_folder : str) -> bool:
    exp_folder = "_".join(exp_folder.split("_")[:-1])
    return experiment_name == exp_folder

def get_current_time_millis() -> int:
    return int(round(time.time() * 1000))

def get_global_rank() -> Optional[int]:
    # if on torch.distributed, return the rank
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    
    # if on slurm, return the local rank
    if "SLURM_PROCID" in os.environ:
        return int(os.getenv("SLURM_PROCID", None))
    
    return 0

def get_runtime_type(): 
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return "distributed"
    return "single_core"