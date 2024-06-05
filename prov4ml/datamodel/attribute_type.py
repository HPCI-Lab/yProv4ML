
import sys
from typing import Any
from collections import namedtuple
from enum import Enum

class LoggingItemKind(Enum): 
    METRIC = 'metric'
    FLOPS_PER_BATCH = 'flops_pb'
    FLOPS_PER_EPOCH = 'flops_pe'
    SYSTEM_METRIC = 'system'
    CARBON_METRIC = 'carbon'
    EXECUTION_TIME = 'execution_time'
    MODEL_VERSION = 'model_version'
    FINAL_MODEL_VERSION = 'model_version_final'
    PARAMETER = 'param'


class Prov4MLAttribute:
    ATTR = namedtuple('prov4ml_attr', ['value'])
    EPOCH_ATTR = namedtuple('prov4ml_epoch_attr', ['epoch', 'value'])
    ATTR_SOURCE = namedtuple('prov4ml_attr_source', ['source', 'value'])
    EPOCH_ATTR_SOURCE = namedtuple('prov4ml_epoch_attr_source', ['epoch', 'source', 'value'])

    @staticmethod
    def get_attr(value: Any) -> str:
        return str(Prov4MLAttribute.ATTR(value))

    @staticmethod
    def get_epoch_attr(epoch : int, value: Any) -> str:
        return str(Prov4MLAttribute.EPOCH_ATTR(epoch, value))
    
    @staticmethod
    def get_source_attr(source: Any, value: Any) -> str:
        return str(Prov4MLAttribute.ATTR_SOURCE(source, value))

    @staticmethod
    def get_epoch_source_attr(epoch: int, source: Any, value: Any) -> str:
        return str(Prov4MLAttribute.EPOCH_ATTR_SOURCE(epoch, source, value))
    
    @staticmethod
    def get_source_from_kind(kind: LoggingItemKind) -> str:
        if kind == LoggingItemKind.METRIC or kind == None:
            return 'custom_metric'
        elif kind == LoggingItemKind.FLOPS_PER_BATCH or kind == LoggingItemKind.FLOPS_PER_EPOCH:
            return 'fvcore.nn.FlopCountAnalysis'
        elif kind == LoggingItemKind.SYSTEM_METRIC:
            if sys.platform != 'darwin':
                return 'pyamdgpuinfo'
            else: 
                return "apple_gpu"            
        elif kind == LoggingItemKind.CARBON_METRIC:
            return 'codecarbon'
        elif kind == LoggingItemKind.EXECUTION_TIME:
            return 'std.time'
        elif kind == LoggingItemKind.MODEL_VERSION or kind == LoggingItemKind.FINAL_MODEL_VERSION:
            return 'torch'
        else:
            return ""

