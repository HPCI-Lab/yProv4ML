
from enum import Enum

class Context(Enum):
    """Enumeration class for defining the context of the metric when saved using log_metrics.

    Attributes:
        TRAINING (str): The context for training metrics.
        VALIDATION (str): The context for validation metrics.
        EVALUATION (str): The context for evaluation metrics.
    """
    TRAINING = 'training'
    EVALUATION = 'evaluation'
    VALIDATION = 'validation'

    @staticmethod
    def get_context_from_string(context: str) -> 'Context':
        """
        Returns the context enum from a string.

        Parameters:
            context (str): The context string.

        Returns:
            Context: The context enum.
        """
        if context == 'training' or context == 'Context.TRAINING':
            return Context.TRAINING
        elif context == 'evaluation' or context == 'Context.EVALUATION':
            return Context.EVALUATION
        elif context == 'validation' or context == 'Context.VALIDATION':
            return Context.VALIDATION
        else:
            raise ValueError(f"Invalid context: {context}")