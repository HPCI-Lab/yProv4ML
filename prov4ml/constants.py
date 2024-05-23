
from collections import namedtuple

MLFLOW_SUBDIR = "mlflow"
ARTIFACTS_SUBDIR = "artifacts"


class Prov4MLLOD():
    LVL_1 = "1"
    LVL_2 = "2"

    lv_attr = namedtuple('lv_attr', ['level', 'value'])
    lv_epoch_value = namedtuple('lv_step_attr', ['level', 'epoch', 'value'])

    def get_lv1_attr(value):
        return str(Prov4MLLOD.lv_attr(Prov4MLLOD.LVL_1, value))

    def get_lv2_attr(value):
        return str(Prov4MLLOD.lv_attr(Prov4MLLOD.LVL_2, value))
    
    def get_lv1_epoch_value(epoch, value):
        return str(Prov4MLLOD.lv_epoch_value(Prov4MLLOD.LVL_1, epoch, value))
    
    def get_lv2_epoch_value(epoch, value):
        return str(Prov4MLLOD.lv_epoch_value(Prov4MLLOD.LVL_2, epoch, value))


class ParameterInfo():
    def __init__(self, name, value):
        self.name = name
        self.value = value


class MetricInfo():
    def __init__(self, name, context):
        self.name = name
        self.context = context
        self.epochDataList = {} 

    def add_metric(self, value, epoch):
        if epoch not in self.epochDataList:
            self.epochDataList[epoch] = []

        self.epochDataList[epoch].append(value)


class Prov4MLData():
    def __init__(self):
        self.metrics = {} # MetricList
        self.parameters = {} # ParameterInfo

    def add_metric(self, metric, value, step, context=None):
        if metric not in self.metrics:
            self.metrics[metric] = MetricInfo(metric, context)
        self.metrics[metric].add_metric(value, step)

    def add_parameter(self, parameter, value):
        self.parameters[parameter] = ParameterInfo(parameter, value)


PROV4ML_DATA = Prov4MLData()
