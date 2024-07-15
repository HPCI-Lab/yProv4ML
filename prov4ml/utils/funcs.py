
import os
import torch
from typing import Optional
import time

def prov4ml_experiment_matches(experiment_name, exp_folder):
    """Check if the experiment name matches the experiment name in the provenance data."""
    exp_folder = "_".join(exp_folder.split("_")[:-1])
    return experiment_name == exp_folder

def get_current_time_millis():
    """Get the current time in milliseconds."""
    return int(round(time.time() * 1000))

def get_global_rank() -> Optional[int]:
    # if on torch.distributed, return the rank
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    
    # if on slurm, return the local rank
    if "SLURM_PROCID" in os.environ:
        return int(os.getenv("SLURM_PROCID", None))
    
    return None