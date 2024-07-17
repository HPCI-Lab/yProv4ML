

class FoldOperation:
    ADD = lambda x, y: x + y
    SUBTRACT = lambda x, y: x - y
    MULTIPLY = lambda x, y: x * y

    MIN = lambda x, y: min(x, y)
    MAX = lambda x, y: max(x, y)


class CumulativeMetric:
    def __init__(self, label: str, initial_value, fold_operation = FoldOperation.ADD) -> None:
        self.label = label
        self.current_value = initial_value
        self.fold_operation = fold_operation

    def update(self, value) -> None:
        self.current_value = self.fold_operation(self.current_value, value)