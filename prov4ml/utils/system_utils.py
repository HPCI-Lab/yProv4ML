
import psutil
import torch
import sys
import warnings
from nvitop import Device
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
    gpu_memory = get_gpu_metric_nvidia('memory_total')
    if gpu_memory is not None:
        return gpu_memory
    
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
            gpu_power = get_gpu_metric_gputil('power')

        if not gpu_power:
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
        gpu_utilization = get_gpu_metric_nvidia('temperature')

        if torch.cuda.is_available():
            gpu_temperature = get_gpu_metric_gputil('temperature')

        if not gpu_temperature:
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
        gpu_utilization = get_gpu_metric_nvidia('utilization')

        if torch.cuda.is_available():
            gpu_utilization = get_gpu_metric_gputil('utilization')

        if not gpu_utilization:
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

    devices = Device.all()
    if len(devices) == 0:  
        return None
    device = devices[0] 

    if metric == 'temperature':
        return device.temperature
    elif metric == "utilization": 
        return device.gpu_utilization
    elif metric == 'fan_speed': 
        return device.fan_speed
    elif metric == 'memory_total':
        return device.memory_total_human
    else: 
        return None

def get_gpu_metric_gputil(metric):
    current_gpu = torch.cuda.current_device()
    gpus = GPUtil.getGPUs()
    if current_gpu < len(gpus):
        if metric == 'temperature':
            return gpus[current_gpu].temperature
        elif metric == "utilization": 
            return gpus[current_gpu].load
        else: 
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