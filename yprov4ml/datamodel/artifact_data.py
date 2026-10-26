
from typing import Any, Optional

from yprov4ml.utils.funcs import get_current_time_millis

class ArtifactInfo:
    def __init__(
        self, 
        name: str, 
        value: Any = None, 
        step: Optional[int] = None, 
        is_input_to: Optional[str] = None, 
        is_output_to: Optional[str] = None, 
        is_model : bool = False, 
    ) -> None:
        self.path = name
        self.value = value
        self.step = step
        self.is_input_to = is_input_to
        self.is_output_to = is_output_to
        self.creation_timestamp = get_current_time_millis()
        self.last_modified_timestamp = get_current_time_millis()

        self.is_model_version = is_model

    def update(
        self, 
        value: Any = None, 
        step: Optional[int] = None, 
        is_input_to: Optional[str] = None, 
        is_output_to: Optional[str] = None, 
    ) -> None:
        self.value = value if value is not None else self.value
        self.step = step if step is not None else self.step
        self.is_input_to = is_input_to if is_input_to is not None else self.is_input_to
        self.is_output_to = is_output_to if is_output_to is not None else self.is_output_to
        self.last_modified_timestamp = get_current_time_millis()
