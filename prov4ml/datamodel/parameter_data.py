
from typing import Any

class ParameterInfo:
    """
    Holds information about a parameter.

    Attributes:
        name (str): The name of the parameter.
        value (Any): The value of the parameter.
    """
    def __init__(self, name: str, value: Any) -> None:
        self.name = name
        self.value = value
