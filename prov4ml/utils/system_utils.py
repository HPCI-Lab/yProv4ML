
import psutil
import torch
import sys
import warnings
if sys.platform != 'darwin':
    import GPUtil
    import pyamdgpuinfo
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
        gpu_power = 0.0
        if torch.cuda.is_available():
            current_gpu = torch.cuda.current_device()
            gpu = pyamdgpuinfo.get_gpu(current_gpu)
            try:
                gpu_power = gpu.query_power()
            except:
                gpu_power = 0.0
                warnings.warn("Could not query GPU power usage.")   
    else:
        statistics = apple_gpu.accelerator_performance_statistics()
        if 'Power Usage' in statistics.keys():
            gpu_power = statistics['Power Usage']
        else:
            gpu_power = 0.0

    return gpu_power
    
def get_gpu_usage() -> float:
    """
    Returns the current GPU usage percentage, if GPU is available.
    
    Returns:
        float: The GPU usage percentage.
    """
    if sys.platform != 'darwin':
        gpu_utilization = 0.0
        if torch.cuda.is_available():
            current_gpu = torch.cuda.current_device()
            gpu = pyamdgpuinfo.get_gpu(current_gpu)
            try: 
                gpu_utilization = gpu.query_utilization()
            except:
                gpu_utilization = 0.0
                warnings.warn("Could not query GPU utilization.")
    else:
        statistics = apple_gpu.accelerator_performance_statistics()
        if 'Device Utilization' in statistics.keys():
            gpu_utilization = statistics['Device Utilization']
        else:
            gpu_utilization = 0.0

    return gpu_utilization