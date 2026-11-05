

import torch
from typing import Optional, Any

from yprov4ml.logging_aux import log_metric

class ProvenanceTrackedFunction:
    def __init__(self, func : Any, is_input_to : Optional[str] = None, is_output_to : Optional[str] = None):
        self.fn = func
        self.is_input_to = is_input_to or type(func).__name__
        self.is_output_to = is_output_to
        self.inc = 0

    def __call__(self, *args, **kwargs):
        result = self.fn(*args, **kwargs)

        if torch.is_tensor(result) and result.shape == torch.Size([]): 
            log_metric(type(self.fn).__name__, result.item(), is_input_to=self.is_input_to, is_output_to=self.is_output_to, step=self.inc)
        else: 
            log_metric(type(self.fn).__name__, result, is_input_to=self.is_input_to, is_output_to=self.is_output_to, step=self.inc)
        self.inc += 1
        return result