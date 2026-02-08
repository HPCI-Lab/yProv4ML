
import psutil
import torch
import sys

get_gpu_usage = lambda: 0.0
get_gpu_memory_usage = lambda: 0.0
get_gpu_power = lambda: 0.0
get_gpu_memory_power = lambda: 0.0
get_gpu_temperature = lambda: 0.0

if sys.platform != 'darwin':
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        if "AMD" in torch.cuda.get_device_name(0): 
            import pyamdgpuinfo
            import amdsmi
            amdsmi.amdsmi_init()
            def amd_get_gpu_usage(): 
                first_gpu = pyamdgpuinfo.get_gpu(0)
                return first_gpu.query_utilization()
            def amd_get_gpu_memory_usage(): 
                return torch.cuda.memory_allocated() / torch.cuda.memory_reserved()
            def amd_get_gpu_power(): 
                first_gpu = pyamdgpuinfo.get_gpu(0)
                return first_gpu.query_power()
            def amd_get_gpu_memory_power(): 
                processors = amdsmi.amdsmi_get_processor_handles()
                power_info = amdsmi.amdsmi_get_power_info(processors[0])
                return power_info['average_socket_power']
            def amd_get_gpu_temperature(): 
                first_gpu = pyamdgpuinfo.get_gpu(0)
                return first_gpu.query_temperature()
            get_gpu_usage = amd_get_gpu_usage
            get_gpu_memory_usage = amd_get_gpu_memory_usage
            get_gpu_power = amd_get_gpu_power
            get_gpu_memory_power = amd_get_gpu_memory_power
            get_gpu_temperature = amd_get_gpu_temperature

        elif "NVIDIA" in torch.cuda.get_device_name(0): 
            import pynvml
            from nvtop import Device
            pynvml.nvmlInit()
            
            def nvidia_get_gpu_usage(): 
                devices = Device.all()
                device = devices[0] 
                return device.gpu_utilization()
            def nvidia_get_gpu_memory_usage(): 
                return torch.cuda.memory_allocated() / torch.cuda.memory_reserved()
            def nvidia_get_gpu_power(): 
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                return power_mw / 1000.0 
            def nvidia_get_gpu_memory_power(): 
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                return power_mw / 1000.0 
            def nvidia_get_gpu_temperature(): 
                devices = Device.all()
                device = devices[0] 
                return device.temperature()    
            get_gpu_usage = nvidia_get_gpu_usage
            get_gpu_memory_usage = nvidia_get_gpu_memory_usage
            get_gpu_power = nvidia_get_gpu_power
            get_gpu_memory_power = nvidia_get_gpu_memory_power
            get_gpu_temperature = nvidia_get_gpu_temperature
else: 
    import apple_gpu
    def apple_get_gpu_usage(): 
        return apple_gpu.accelerator_performance_statistics()['Device Utilization %']
    def apple_get_gpu_memory_usage(): 
        d = apple_gpu.accelerator_performance_statistics()
        used = d['In use system memory']
        tot = d['Alloc system memory']
        return 100*used / tot
    get_gpu_usage = apple_get_gpu_usage
    get_gpu_memory_usage = apple_get_gpu_memory_usage

def get_cpu_usage() -> float:
    return psutil.cpu_percent()

def get_memory_usage() -> float:
    return psutil.virtual_memory().percent

def get_disk_usage() -> float:
    return psutil.disk_usage('/').percent