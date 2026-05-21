
from typing import Any, Optional

from yprov4ml.utils.funcs import get_current_time_millis

class ArtifactInfo:
    def __init__(
        self, 
        name: str, 
        value: Any = None, 
        step: Optional[int] = None, 
        context: Optional[str] = None, 
        source: Optional[str] = None, 
        is_model : bool = False, 
    ) -> None:
        self.path = name
        self.value = value
        self.step = step
        self.context = context
        self.source = source
        self.creation_timestamp = get_current_time_millis()
        self.last_modified_timestamp = get_current_time_millis()

        self.is_model_version = is_model

    def update(
        self, 
        value: Any = None, 
        step: Optional[int] = None, 
        context: Optional[str] = None
    ) -> None:
        self.value = value if value is not None else self.value
        self.step = step if step is not None else self.step
        self.context = context if context is not None else self.context
        self.last_modified_timestamp = get_current_time_millis()
