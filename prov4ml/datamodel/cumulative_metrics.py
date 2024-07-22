
from typing import Any  

class FoldOperation:
    """
    A collection of fold operations used to update cumulative metrics.

    Attributes:
    -----------
    ADD : callable
        A lambda function to add two values.
    SUBTRACT : callable
        A lambda function to subtract the second value from the first.
    MULTIPLY : callable
        A lambda function to multiply two values.
    MIN : callable
        A lambda function to return the minimum of two values.
    MAX : callable
        A lambda function to return the maximum of two values.
    """
    ADD = lambda x, y: x + y
    SUBTRACT = lambda x, y: x - y
    MULTIPLY = lambda x, y: x * y

    MIN = lambda x, y: min(x, y)
    MAX = lambda x, y: max(x, y)


class CumulativeMetric:
    """
    A class to manage a cumulative metric, which updates its value based on a fold operation.

    Attributes:
    -----------
    label : str
        The label for the cumulative metric.
    current_value : Any
        The current value of the cumulative metric.
    fold_operation : callable
        The operation used to combine the current value with new values.

    Methods:
    --------
    __init__(label: str, initial_value: Any, fold_operation=FoldOperation.ADD) -> None
        Initializes the CumulativeMetric with a label, an initial value, and a fold operation.
    update(value: Any) -> None
        Updates the current value of the metric using the specified fold operation.
    """
    def __init__(
            self, 
            label: str, 
            initial_value: Any,
            fold_operation = FoldOperation.ADD
        ) -> None:
        """
        Initializes the CumulativeMetric with a label, an initial value, and a fold operation.

        Parameters:
        -----------
        label : str
            The label for the cumulative metric.
        initial_value : Any
            The initial value of the cumulative metric.
        fold_operation : callable, optional
            The operation to use for updating the cumulative value. Defaults to FoldOperation.ADD.

        Returns:
        --------
        None
        """
        self.label = label
        self.current_value = initial_value
        self.fold_operation = fold_operation

    def update(self, value : Any) -> None:
        """
        Updates the current value of the metric using the fold operation.

        Parameters:
        -----------
        value : Any
            The new value to be combined with the current value using the fold operation.

        Returns:
        --------
        None
        """
        self.current_value = self.fold_operation(self.current_value, value)