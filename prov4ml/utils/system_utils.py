
import psutil
import torch
import sys
if sys.platform != 'darwin':
    import GPUtil
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
    
def get_gpu_usage() -> float:
    """
    Returns the current GPU usage percentage, if GPU is available.
    
    Returns:
        float: The GPU usage percentage.
    """
    if sys.platform != 'darwin':
        GPUs = GPUtil.getGPUs()
        if len(GPUs) == 0:
            gpu_load = 0.0
        else: 
            gpu_load = GPUs[0].load
    else:
        statistics = apple_gpu.accelerator_performance_statistics()
        if 'Device Utilization' in statistics.keys():
            gpu_load = statistics['Device Utilization']
        else:
            gpu_load = 0.0

    return gpu_load