
import psutil
import torch
import sys
if sys.platform != 'darwin':
    import GPUtil
else: 
    import apple_gpu

def get_cpu_usage():
    """
    Get the current memory usage of the CPU in percentage.

    :return: The memory usage of the CPU in percentage.
    :rtype: float
    """
    return psutil.cpu_percent()

def get_memory_usage():
    """
    Get the current memory usage of the system in percentage.

    :return: The memory usage of the system in percentage.
    :rtype: float
    """
    return psutil.virtual_memory().percent

def get_disk_usage():
    """
    Get the current disk usage of the system in percentage.

    :return: The disk usage of the system in percentage.
    :rtype: float
    """
    return psutil.disk_usage('/').percent

def get_gpu_memory_usage():
    """
    Get the current memory usage of the GPU in percentage.

    :return: The memory usage of the GPU in percentage.
    :rtype: float
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / torch.cuda.memory_reserved()
    else:
        return 0
    
def get_gpu_usage():
    if sys.platform != 'darwin':
        GPUs = GPUtil.getGPUs()
        gpu_load = GPUs[0].load
    else:
        statistics = apple_gpu.accelerator_performance_statistics()
        if 'Device Utilization' in statistics.keys():
            gpu_load = statistics['Device Utilization']
        else:
            gpu_load = 0.0

    return gpu_load