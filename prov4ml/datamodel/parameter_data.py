
from typing import Any

class ParameterInfo:
    """
    Holds information about a parameter.

    Attributes:
        name (str): The name of the parameter.
        value (Any): The value of the parameter.

    Methods:
        __init__(name: str, value: Any) -> None
            Initializes the ParameterInfo class with the given name and value.
    """
    def __init__(self, name: str, value: Any) -> None:
        self.name = name
        self.value = value
