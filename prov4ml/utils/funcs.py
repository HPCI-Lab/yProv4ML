
import time

def prov4ml_experiment_matches(experiment_name, exp_folder):
    """Check if the experiment name matches the experiment name in the provenance data."""
    exp_folder = "_".join(exp_folder.split("_")[:-1])
    return experiment_name == exp_folder

def get_current_time_millis():
    """Get the current time in milliseconds."""
    return int(round(time.time() * 1000))