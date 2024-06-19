
import psutil
import torch
import sys
import warnings
if sys.platform != 'darwin':
    import pyamdgpuinfo
    import GPUtil
    import gpustat
else: 
    import apple_gpu


def get_cpu_usage() -> float:
    """
    Returns the current CPU usage percentage.
    
    Returns:
        float: The CPU usage percentage.
    """
    return psutil.cpu_percent()

def get_memory_usage() -> float:
    """
    Returns the current memory usage percentage.
    
    Returns:
        float: The memory usage percentage.
    """
    return psutil.virtual_memory().percent

def get_disk_usage() -> float:
    """
    Returns the current disk usage percentage.
    
    Returns:
        float: The disk usage percentage.
    """
    return psutil.disk_usage('/').percent

def get_gpu_memory_usage() -> float:
    """
    Returns the current GPU memory usage percentage, if GPU is available.
    
    Returns:
        float: The GPU memory usage percentage.
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / torch.cuda.memory_reserved()
    else:
        return 0
    
def get_gpu_power_usage() -> float:
    """
    Returns the current GPU power usage percentage, if GPU is available.
    
    Returns:
        float: The GPU power usage percentage.
    """
    if sys.platform != 'darwin':
        gpu_power = None
        if torch.cuda.is_available():
            gpu_power = get_gpu_metric_intel('power')

        if gpu_power is None:
            gpu_power = get_gpu_metric_nvidia('power')

        if gpu_power is None:
            gpu_power = get_gpu_metric_amd('power')
    else:
        gpu_power = get_gpu_metric_apple('power')

    return gpu_power if gpu_power is not None else 0.0
    

def get_gpu_temperature() -> float:
    """
    Returns the current GPU temperature, if GPU is available.
    
    Returns:
        float: The GPU temperature.
    """
    if sys.platform != 'darwin':
        gpu_temperature = None
        if torch.cuda.is_available():
            gpu_temperature = get_gpu_metric_intel('temperature')

        if gpu_temperature is None:
            gpu_temperature = get_gpu_metric_nvidia('temperature')

        if gpu_temperature is None:
            gpu_temperature = get_gpu_metric_amd('temperature')

    else:
        gpu_temperature = get_gpu_metric_apple('temperature')

    return gpu_temperature if gpu_temperature is not None else 0.0

def get_gpu_usage() -> float:
    """
    Returns the current GPU usage percentage, if GPU is available.
    
    Returns:
        float: The GPU usage percentage.
    """
    if sys.platform != 'darwin':
        gpu_utilization = None
        if torch.cuda.is_available():
            gpu_utilization = get_gpu_metric_intel('utilization')

        if gpu_utilization is None:
            gpu_utilization = get_gpu_metric_nvidia('utilization')

        if gpu_utilization is None:
            gpu_utilization = get_gpu_metric_amd('utilization')
    else:
        gpu_utilization = get_gpu_metric_apple('utilization')

    return gpu_utilization if gpu_utilization is not None else 0.0


def get_gpu_metric_amd(metric): 
    try: 
        first_gpu = pyamdgpuinfo.get_gpu(0)
        if metric == 'power':
            m = first_gpu.query_power()
        elif metric == 'temperature':
            m = first_gpu.query_temperature()
        elif metric == 'utilization':
            m = first_gpu.query_utilization()

        return m
    except:
        warnings.warn(f"Could not get metric: {metric}")
        return None

def get_gpu_metric_nvidia(metric):
    return None
    # try:
    #     stats = gpustat.new_query()
    #     if metric == 'power':
    #         return None
    #     elif metric == 'temperature':
    #         return None
    #     elif metric == 'utilization':
    #         return stats.gpus[0]['utilization.gpu']
    # except:
    #     warnings.warn(f"Could not get metric: {metric}")
    #     return None

def get_gpu_metric_intel(metric):
    current_gpu = torch.cuda.current_device()
    gpus = GPUtil.getGPUs()
    if current_gpu < len(gpus) and hasattr(gpus[current_gpu], metric):
        try:
            return getattr(gpus[current_gpu], metric)
        except:
            return None
    else:
        warnings.warn(f"Could not get metric: {metric}")
        return None

def get_gpu_metric_apple(metric):
    statistics = apple_gpu.accelerator_performance_statistics()
    if metric == 'power':
        return None
    elif metric == 'temperature':
        return None
    elif metric == 'utilization':
        return statistics['Device Utilization %']
    else: 
        warnings.warn(f"Could not get metric: {metric}")
        return None

